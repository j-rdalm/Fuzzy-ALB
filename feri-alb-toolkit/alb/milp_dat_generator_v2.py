"""
milp_dat_generator_v2.py
=========================
Reads FERI_results.xlsx (produced by fuzzy_eri.py) and writes:
  - assembly_line.dat      → CPLEX OPL data file
  - precedence_diagram.png → visual of the task DAG (nodes coloured by station)

Precedence is read directly from the "Precedence" sheet of FERI_results.xlsx.
Station assignment is read from the "Station" column of the "FERI Results" sheet.
No group logic. Works for any generic production-line task list.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ─────────────────────────────────────────────────────────────
# 0. Configuration  ← adjust these for your problem
# ─────────────────────────────────────────────────────────────

_HERE      = Path(__file__).parent
FERI_FILE  = _HERE.parent / "feri" / "FERI_results.xlsx"
DAT_FILE   = _HERE / "assembly_line.dat"
PREC_PNG   = _HERE / "precedence_diagram.png"

N_STATIONS    = 10      # upper bound on number of operator stations K
CYCLE_TIME_UB = 120.0  # takt time in seconds (max allowed CT)
ERGO_LIMIT    = 30.0   # max ergonomic load (ERI × s) per station
W_TIME        = 0.5    # default objective weight for cycle time (0 = ergonomics-only, 1 = time-only)
W_ERI         = 0.5    # default objective weight for ERI  (W_TIME + W_ERI = 1.0)

# Weight pairs (w_time, w_eri) for sensitivity analysis: 5 × 6 α-cut scenarios = 30 runs
WEIGHT_PAIRS  = [
    (1.00, 0.00),   # time-only
    (0.75, 0.25),
    (0.50, 0.50),   # balanced (matches default above)
    (0.25, 0.75),
    (0.00, 1.00),   # ergonomics-only
]

TIME_LIMIT    = 285    # CPLEX solver time limit in seconds
CT_MARGIN     = 1.2    # CT_max  = (sum(TN_m) / n_stations) × CT_MARGIN
                       #   1.0 = ideal average (may be infeasible)
                       #   1.2 = 20% slack above ideal (recommended default)
ERI_MARGIN    = 1.2    # ERI_max = (sum(ERI)  / n_stations) × ERI_MARGIN

# α-cut parametric analysis — confidence levels to solve
# For each α ∈ ALPHA_CUTS, two .dat files are generated:
#   pessimistic (upper α-cut bounds) and optimistic (lower α-cut bounds).
# α=1.0 collapses to the crisp midpoint (same as the original single-run mode).
ALPHA_CUTS    = [0.0, 0.5, 1.0]


# ─────────────────────────────────────────────────────────────
# 1. Load FERI results and precedence
# ─────────────────────────────────────────────────────────────

def load_data(path: str):
    xl = pd.read_excel(path, sheet_name=None)

    # ── FERI Results sheet
    df = xl["FERI Results"].copy()
    df.columns = df.columns.str.strip()
    # Columns: Task ID | Station | Task Name | Time (s) | TN_l | TN_m | TN_u |
    #          Energy (kcal) | REBA | Borg | ERI_l | ERI_m | ERI_u | ERI (crisp) | Risk Level
    # All six fuzzy bounds (TN_l/m/u, ERI_l/m/u) are loaded for α-cut analysis.
    df = df.rename(columns={
        "Task ID":      "task_id",
        "Station":      "station",
        "Task Name":    "task_name",
        "TN_l":         "time_l",
        "TN_m":         "time_m",
        "TN_u":         "time_u",
        "ERI_l":        "eri_l",
        "ERI_m":        "eri_m",
        "ERI_u":        "eri_u",
        "ERI (crisp)":  "ERI",    # kept for backward-compat summary output
    })
    df = df.dropna(subset=["task_id"]).reset_index(drop=True)
    df["task_id"] = df["task_id"].astype(int)
    df["station"] = df["station"].astype(int)

    # ── Precedence sheet
    pr = xl["Precedence"].copy()
    pr.columns = pr.columns.str.strip()
    pr = pr[["Predecessor ID", "Successor ID"]].dropna().reset_index(drop=True)
    pr.columns = ["pred_id", "succ_id"]
    pr["pred_id"] = pr["pred_id"].astype(int)
    pr["succ_id"] = pr["succ_id"].astype(int)

    return df, pr


# ─────────────────────────────────────────────────────────────
# 2. Validate DAG (no cycles)
# ─────────────────────────────────────────────────────────────

def validate_dag(df, pr):
    G = nx.DiGraph()
    G.add_nodes_from(df["task_id"].tolist())
    valid_ids = set(df["task_id"])
    for _, row in pr.iterrows():
        p = int(row["pred_id"])
        s = int(row["succ_id"])
        if p not in valid_ids or s not in valid_ids:
            raise ValueError(
                f"Invalid precedence detected: {p} → {s} "
                f"(task missing in 'FERI Results')"
            )
        G.add_edge(p, s)    
    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        raise ValueError(f"Precedence graph contains cycles: {cycles}")
    print(f"  DAG validation passed — {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


# ─────────────────────────────────────────────────────────────
# 3. α-cut helper
# ─────────────────────────────────────────────────────────────

def alpha_cut_bounds(l_series, m_series, u_series, alpha):
    """Return (lower_bound, upper_bound) Series for the α-cut of a TFN column.

    For a TFN (l, m, u) at confidence level α:
      lower: l_α = l + α*(m − l)
      upper: u_α = u − α*(u − m)
    At α=1 both collapse to m (crisp midpoint).
    At α=0 they equal l and u (full uncertainty interval).
    """
    lower = l_series + alpha * (m_series - l_series)
    upper = u_series - alpha * (u_series - m_series)
    return lower, upper


# ─────────────────────────────────────────────────────────────
# 4. Write CPLEX .dat file
# ─────────────────────────────────────────────────────────────

def write_dat(df, pr, out_path: str, time_vals=None, eri_vals=None, label="",
              w_time=None, w_eri=None, scenario_tag=""):
    # Default to midpoint values when called without explicit arrays (backward compat)
    if time_vals is None:
        time_vals = df["time_m"]
    if eri_vals is None:
        eri_vals = df["ERI"]
    wt = w_time if w_time is not None else W_TIME
    we = w_eri  if w_eri  is not None else W_ERI

    # Reset index so positional access is safe
    time_vals = time_vals.reset_index(drop=True)
    eri_vals  = eri_vals.reset_index(drop=True)

    # Map task_id → sequential index (1-based, matching OPL ranges)
    id_to_idx = {tid: i+1 for i, tid in enumerate(df["task_id"].tolist())}

    n = len(df)
    K = N_STATIONS
    n_workstations = int(df["station"].max())

    # Station description comments
    station_groups = df.groupby("station")["task_id"].apply(list).to_dict()

    # Build precedence with sequential indices
    prec_pairs = []
    for _, row in pr.iterrows():
        p = id_to_idx.get(int(row["pred_id"]))
        s = id_to_idx.get(int(row["succ_id"]))
        if p is None or s is None:
            print(f"  WARNING: precedence ({row['pred_id']}→{row['succ_id']}) "
                  f"references unknown task ID — skipped.")
        else:
            prec_pairs.append((p, s))

    # ── Determine column widths for inline comments
    max_id_digits = len(str(n))

    lines = []

    # ── Header
    label_str = f" [{label}]" if label else ""
    lines.append("/*********************************************")
    lines.append(f" * OPL Data File - Assembly Line Balancing{label_str}")
    lines.append(" * Two objectives: Minimize Cycle Time and ERI")
    lines.append(f" * Source: {Path(out_path).name} ({n} tasks, {K} operators, "
                 f"{n_workstations} workstations)")
    lines.append(" *")
    lines.append(" * EXTENSIONS (v2):")
    lines.append(" *  - taskWorkstation[i]: physical workstation of each task")
    lines.append(" *  - n_workstations: number of physical workstations")
    lines.append(" *********************************************/")
    lines.append("")

    # ── n_tasks / n_stations / n_workstations
    lines.append("// Number of tasks, operators and physical workstations")
    lines.append(f"n_tasks         = {n};")
    lines.append(f"n_stations      = {K};    // number of operators")
    lines.append(f"n_workstations  = {n_workstations};    // number of physical workstations")
    lines.append("")

    # ── time_task  (multi-line, one value per line with inline comment)
    lines.append("// Task processing times — normalised (from α-cut or midpoint, depending on scenario)")
    lines.append("time_task = [")
    for i, row in df.iterrows():
        idx      = i + 1
        is_last  = (idx == n)
        comma    = "" if is_last else ","
        t_str    = f"{time_vals[i]:.4f}{comma}"
        comment  = f"// Task {idx:<{max_id_digits}} - {row['task_name']}"
        lines.append(f"    {t_str:<12}  {comment}")
    lines.append("];")
    lines.append("")

    # ── eri  (multi-line, one value per line with inline comment)
    lines.append("// ERI (Ergonomic Risk Index) per task")
    lines.append("eri = [")
    for i, row in df.iterrows():
        idx      = i + 1
        is_last  = (idx == n)
        comma    = "" if is_last else ","
        e_str    = f"{eri_vals[i]:.6f}{comma}"
        comment  = f"// Task {idx:<{max_id_digits}} - {row['task_name']}"
        lines.append(f"    {e_str:<12}  {comment}")
    lines.append("];")
    lines.append("")

    # ── taskWorkstation  (physical workstation per task, from Excel "Station" column)
    lines.append("// Physical workstation assignment per task")
    lines.append("// (read from 'Station' column of FERI_results.xlsx)")
    for ws_id, task_ids in sorted(station_groups.items()):
        idx_list = [str(id_to_idx[tid]) for tid in task_ids]
        lines.append(f"// Workstation {ws_id}: Tasks {', '.join(idx_list)}")
    lines.append("taskWorkstation = [")
    for i, row in df.iterrows():
        idx      = i + 1
        is_last  = (idx == n)
        comma    = "" if is_last else ","
        ws_str   = f"{int(row['station'])}{comma}"
        comment  = f"// Task {idx:<{max_id_digits}} - {row['task_name']}"
        lines.append(f"    {ws_str:<5}  {comment}")
    lines.append("];")
    lines.append("")

    # ── precedence  (OPL tuple-set syntax: { <p, s>, ... })
    lines.append("// Precedence constraints: task pred must precede task succ")
    lines.append("precedence = {")
    for k, (p, s) in enumerate(prec_pairs):
        is_last = (k == len(prec_pairs) - 1)
        comma   = "" if is_last else ","
        lines.append(f"    <{p}, {s}>{comma}")
    lines.append("};")
    lines.append("")

    # ── w_time / w_eri weights
    lines.append("// Objective weights for bi-objective scalarisation")
    lines.append("// w_time: weight for cycle-time term (0 = ergonomics-only, 1 = time-only)")
    lines.append("// w_eri:  weight for ERI term  (w_time + w_eri = 1.0)")
    lines.append(f"w_time = {wt:.4f};")
    lines.append(f"w_eri  = {we:.4f};")
    lines.append("")
    lines.append("// Run identifier — echoed by CPLEX DISPLAY_RESULTS for result parsing")
    tag = scenario_tag if scenario_tag else label
    lines.append(f'scenario_tag = "{tag}";')
    lines.append("")

    # ── CPLEX time limit
    lines.append("// CPLEX time limit (in seconds)")
    lines.append(f"timeLimit = {TIME_LIMIT};")
    lines.append("")

    # ── CT_max and ERI_max  (caps for OBJ-2 Dual SI)
    # Computed as the ideal balanced load = total / n_stations (normalised units).
    # This is the tightest meaningful upper bound: a perfectly balanced line would
    # have every operator at exactly this load. Setting CT_max/ERI_max to this
    # value forces OBJ-2 to find the most balanced feasible assignment without
    # allowing the solver to stack all tasks on one operator.
    ct_max_val  = (time_vals.sum() / K) * CT_MARGIN
    eri_max_val = (eri_vals.sum()  / K) * ERI_MARGIN
    lines.append("// Upper-bound caps for OBJ-2 (Dual Smoothness Index).")
    lines.append(f"// CT_max  = (sum(TN_m) / n_stations) x {CT_MARGIN}  — ideal time load + {int((CT_MARGIN-1)*100)}% margin")
    lines.append(f"// ERI_max = (sum(ERI)  / n_stations) x {ERI_MARGIN}  — ideal ERI  load + {int((ERI_MARGIN-1)*100)}% margin")
    lines.append(f"CT_max  = {ct_max_val:.6f};")
    lines.append(f"ERI_max = {eri_max_val:.6f};")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    suffix = f" [{label}]" if label else ""
    print(f"\n  DAT written : {out_path}{suffix}")
    print(f"  Tasks       : {n}")
    print(f"  Precedences : {len(prec_pairs)}")
    print(f"  Workstations: {n_workstations}")
    print(f"  Stations UB : {K}")
    print(f"  Takt time   : {CYCLE_TIME_UB} s")
    print(f"  Ergo limit  : {ERGO_LIMIT}")
    print(f"  W_TIME      : {wt}")
    print(f"  W_ERI       : {we}")
    print(f"  Time limit  : {TIME_LIMIT} s")

    return id_to_idx, prec_pairs


# ─────────────────────────────────────────────────────────────
# 4. Plot precedence diagram  (nodes coloured by workstation)
# ─────────────────────────────────────────────────────────────

# Palette for up to 8 workstations (visually distinct, colour-blind friendly)
# Palette for up to 20 workstations (distinct & reasonably color-blind friendly)
STATION_PALETTE = [
    "#2980B9",  # WS1 – blue
    "#27AE60",  # WS2 – green
    "#E67E22",  # WS3 – orange
    "#8E44AD",  # WS4 – purple
    "#C0392B",  # WS5 – red
    "#16A085",  # WS6 – teal
    "#F39C12",  # WS7 – amber
    "#2C3E50",  # WS8 – dark slate

    "#7F8C8D",  # WS9 – gray
    "#D35400",  # WS10 – dark orange
    "#1ABC9C",  # WS11 – light teal
    "#9B59B6",  # WS12 – lavender purple
    "#34495E",  # WS13 – navy gray
    "#E74C3C",  # WS14 – bright red
    "#2ECC71",  # WS15 – light green
    "#3498DB",  # WS16 – light blue

    "#F1C40F",  # WS17 – yellow
    "#95A5A6",  # WS18 – light gray
    "#A93226",  # WS19 – deep red
    "#117864",  # WS20 – deep teal
]

def station_color(station_id: int) -> str:
    idx = (station_id - 1) % len(STATION_PALETTE)
    return STATION_PALETTE[idx]

def eri_fill(eri: float) -> str:
    """Fill colour encodes ERI level."""
    if   eri < 0.2: return "#1E8449"
    elif eri < 0.4: return "#239B56"
    elif eri < 0.6: return "#D4AC0D"
    elif eri < 0.8: return "#CA6F1E"
    else:           return "#922B21"

def plot_precedence(df, pr, G, out_path: str):
    # Build display graph
    DG = nx.DiGraph()
    for _, row in df.iterrows():
        DG.add_node(int(row["task_id"]),
                    label=f"T{row['task_id']}",
                    eri=row["ERI"],
                    time=row["time_m"],
                    station=int(row["station"]),
                    name=row["task_name"])
    for _, row in pr.iterrows():
        DG.add_edge(int(row["pred_id"]), int(row["succ_id"]))

    # Use topological generations for layered layout
    try:
        layers = list(nx.topological_generations(DG))
    except Exception:
        layers = [list(DG.nodes())]

    pos = {}
    for layer_idx, layer in enumerate(layers):
        layer = sorted(layer)
        n_in_layer = len(layer)
        for ni, node in enumerate(layer):
            pos[node] = (layer_idx * 2.8,
                         -(ni - (n_in_layer - 1) / 2) * 1.6)

    fig, ax = plt.subplots(figsize=(max(16, len(layers) * 1.2), 12))
    ax.set_facecolor("#F4F6F7")
    fig.patch.set_facecolor("#F4F6F7")

    # ── Draw station background bands ──
    station_groups = df.groupby("station")["task_id"].apply(list).to_dict()

    for ws_id, task_ids in station_groups.items():
        xs = [pos[tid][0] for tid in task_ids if tid in pos]
        ys = [pos[tid][1] for tid in task_ids if tid in pos]
        if not xs:
            continue
        pad_x, pad_y = 0.9, 0.7
        x0, x1 = min(xs) - pad_x, max(xs) + pad_x
        y0, y1 = min(ys) - pad_y, max(ys) + pad_y
        color = station_color(ws_id)
        rect = mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.1",
            linewidth=2, edgecolor=color,
            facecolor=color + "22")
        ax.add_patch(rect)
        ax.text(x0 + 0.15, y1 - 0.15,
                f"WS {ws_id}",
                ha="left", va="top",
                fontsize=9, fontweight="bold",
                color=color)

    # ── Edges
    nx.draw_networkx_edges(DG, pos, ax=ax,
                           edge_color="#7F8C8D", arrows=True,
                           arrowsize=20, width=1.8,
                           connectionstyle="arc3,rad=0.08",
                           node_size=2000,
                           min_source_margin=20,
                           min_target_margin=20)

    # ── Nodes  (fill = station colour, edge ring = ERI level)
    node_list    = list(DG.nodes)
    node_fills   = [eri_fill(DG.nodes[n]["eri"]) for n in node_list]
    node_borders = [station_color(DG.nodes[n]["station"]) for n in node_list]

    nx.draw_networkx_nodes(DG, pos, ax=ax,
                           nodelist=node_list,
                           node_color=node_fills,
                           node_size=2000,
                           edgecolors=node_borders,
                           linewidths=5)

    # ── Node labels: Task ID + ERI value
    node_labels = {n: f"T{n}\n{DG.nodes[n]['eri']:.2f}" for n in DG.nodes}
    nx.draw_networkx_labels(DG, pos, labels=node_labels,
                            ax=ax, font_size=7.5,
                            font_weight="bold", font_color="white")

    # ── Layer headers
    for layer_idx, layer in enumerate(layers):
        x_coord = layer_idx * 2.8
        ys = [pos[n][1] for n in layer]
        y_top = max(ys) + 1.35
        ax.text(x_coord, y_top, f"Layer {layer_idx+1}",
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="#1A5276",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#D6EAF8", edgecolor="#1A5276", linewidth=1.2))

    # ── Legend – Station colours
    station_patches = [
        mpatches.Patch(color=station_color(ws), label=f"Workstation {ws}")
        for ws in sorted(station_groups.keys())
    ]
    legend_ws = ax.legend(handles=station_patches,
                          loc="upper right", fontsize=9,
                          framealpha=0.92, title="Workstation (node border)",
                          title_fontsize=9)
    ax.add_artist(legend_ws)

    # ── Legend – ERI fill colours
    eri_patches = [
        mpatches.Patch(color="#1E8449", label="ERI < 0.2  Low"),
        mpatches.Patch(color="#239B56", label="ERI 0.2–0.4  Moderate"),
        mpatches.Patch(color="#D4AC0D", label="ERI 0.4–0.6  Medium"),
        mpatches.Patch(color="#CA6F1E", label="ERI 0.6–0.8  High"),
        mpatches.Patch(color="#922B21", label="ERI > 0.8  Very High"),
    ]
    ax.legend(handles=eri_patches,
              loc="lower right", fontsize=9,
              framealpha=0.92, title="ERI risk (node fill)",
              title_fontsize=9)

    ax.set_title("Production Line — Task Precedence Diagram\n"
                 "(node fill = ERI risk  |  node border = workstation  |  label = Task ID / ERI)",
                 fontsize=13, fontweight="bold", pad=18)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    #plt.savefig('output.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Diagram     : {out_path}")


# ─────────────────────────────────────────────────────────────
# 5. Print summary table
# ─────────────────────────────────────────────────────────────

def print_summary(df, pr):
    print("\n" + "="*85)
    print("  MILP INPUT SUMMARY  (midpoint values — TN_m, ERI crisp)")
    print("="*85)
    print(f"  {'Idx':>4}  {'ID':>5}  {'WS':>3}  {'Task':<30}  {'TN_m':>8}  "
          f"{'ERI':>7}  {'ERI×TN_m':>9}")
    print("  " + "-"*83)
    for i, row in df.iterrows():
        print(f"  {i+1:>4}  {row['task_id']:>5}  {row['station']:>3}  "
              f"{row['task_name']:<30}  "
              f"{row['time_m']:>8.4f}  {row['ERI']:>7.4f}  "
              f"{row['ERI']*row['time_m']:>8.3f}")
    print("  " + "-"*83)
    print(f"  {'TOTAL':>4}  {'':>5}  {'':>3}  {'':30}  {df['time_m'].sum():>8.4f}  "
          f"{'':>7}  {(df['ERI']*df['time_m']).sum():>8.3f}")
    print(f"\n  Precedence pairs: {len(pr)}")
    print("="*85)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading FERI results …")
    df, pr = load_data(FERI_FILE)
    print(f"  {len(df)} tasks  |  {len(pr)} precedence pairs  |  "
          f"{df['station'].max()} workstations")

    print("Validating precedence graph …")
    G = validate_dag(df, pr)

    print_summary(df, pr)

    # ── Backward-compatible single .dat file (fuzzy cut=1.0 midpoint, w_time=0.5)
    print("\nWriting base CPLEX .dat file (fuzzy cut=1.0 midpoint) …")
    write_dat(df, pr, DAT_FILE,
              time_vals=df["time_m"], eri_vals=df["ERI"],
              label="cut=1.0 midpoint (backward-compat)")

    # ── 25-run sensitivity experiment
    # For each weight pair and fuzzy cut level:
    #   cut < 1.0 → two files (pessimistic = upper bounds, optimistic = lower bounds)
    #   cut = 1.0 → one file  (upper and lower both collapse to midpoint — identical)
    # File naming: assembly_line_wt{aaa}_we{bbb}_cut{cc}_{scenario}.dat
    #   wt{aaa} = w_time × 100, zero-padded (e.g. 100, 075, 050, 025, 000)
    #   we{bbb} = w_eri  × 100, zero-padded
    #   cut{cc} = cut level × 10, zero-padded (e.g. 00, 05, 10)
    n_partial = sum(1 for c in ALPHA_CUTS if c < 1.0)
    n_full    = sum(1 for c in ALPHA_CUTS if c == 1.0)
    n_files   = len(WEIGHT_PAIRS) * (n_partial * 2 + n_full)
    print(f"\nWriting {n_files} .dat files "
          f"({len(WEIGHT_PAIRS)} weight pairs × "
          f"{n_partial} partial cuts × 2 scenarios + {n_full} midpoint) …")
    generated = []
    for (wt, we) in WEIGHT_PAIRS:
        aaa = f"{round(wt * 100):03d}"
        bbb = f"{round(we * 100):03d}"
        for cut in ALPHA_CUTS:
            t_lo, t_hi = alpha_cut_bounds(df["time_l"], df["time_m"], df["time_u"], cut)
            e_lo, e_hi = alpha_cut_bounds(df["eri_l"],  df["eri_m"],  df["eri_u"],  cut)
            cut_str = f"{cut:.1f}".replace(".", "")
            if cut == 1.0:
                fname = _HERE / f"assembly_line_wt{aaa}_we{bbb}_cut{cut_str}.dat"
                tag   = f"wt={wt:.2f} we={we:.2f} cut={cut:.1f} midpoint"
                write_dat(df, pr, fname,
                          time_vals=t_hi, eri_vals=e_hi,
                          label=tag, w_time=wt, w_eri=we, scenario_tag=tag)
                generated.append(fname.name)
            else:
                for scenario, t_vals, e_vals in [
                    ("pessimistic", t_hi, e_hi),
                    ("optimistic",  t_lo, e_lo),
                ]:
                    fname = _HERE / f"assembly_line_wt{aaa}_we{bbb}_cut{cut_str}_{scenario}.dat"
                    tag   = f"wt={wt:.2f} we={we:.2f} cut={cut:.1f} {scenario}"
                    write_dat(df, pr, fname,
                              time_vals=t_vals, eri_vals=e_vals,
                              label=tag, w_time=wt, w_eri=we, scenario_tag=tag)
                    generated.append(fname.name)

    print("\nPlotting precedence diagram …")
    plot_precedence(df, pr, G, PREC_PNG)

    print("\n" + "="*70)
    print("  GENERATED FILES")
    print("="*70)
    print("  assembly_line.dat  ← base (cut=1.0 midpoint, wt=0.5 we=0.5)")
    print()
    for (wt, we) in WEIGHT_PAIRS:
        aaa = f"{round(wt * 100):03d}"
        bbb = f"{round(we * 100):03d}"
        print(f"  wt={wt:.2f} we={we:.2f}:")
        for cut in ALPHA_CUTS:
            cut_str = f"{cut:.1f}".replace(".", "")
            if cut == 1.0:
                print(f"    assembly_line_wt{aaa}_we{bbb}_cut{cut_str}.dat             ← cut={cut} midpoint (crisp)")
            else:
                print(f"    assembly_line_wt{aaa}_we{bbb}_cut{cut_str}_pessimistic.dat  ← cut={cut} worst-case")
                print(f"    assembly_line_wt{aaa}_we{bbb}_cut{cut_str}_optimistic.dat   ← cut={cut} best-case")
        print()
    print(f"  Total: {len(generated)} .dat files + 1 base file")
    print()
    print("  NEXT STEPS")
    print("="*70)
    print("  1. Open CPLEX IDE (OPL Studio)")
    print("  2. Create a new OPL project")
    print("  3. Add  assembly_line_new_model.mod  as model file")
    print("  4. Add one .dat file at a time and run")
    print("  5. Compare objective ranges across cut-levels per weight pair")
    print("     to recover the membership function of the fuzzy optimal solution")
    print("  6. Adjust W_TIME/W_ERI/WEIGHT_PAIRS/N_STATIONS/ALPHA_CUTS")
    print("     at the top of this file and re-run as needed")
    print("="*70)
