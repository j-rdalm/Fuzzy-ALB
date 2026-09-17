"""
fuzzy_eri.py
================
Fuzzy Ergonomic Risk Index (FERI) 

Changes from v2:
  - Input file now has two sheets: "Tasks" and "Precedence"
  - No group/prefix logic — tasks are generic
  - Reads columns: Task ID | Task Name | Time (s) | Energy (kcal) | REBA | Borg
  - Exports FERI_results.xlsx with same two-sheet structure for downstream use

Run order:
  1. python fuzzy_eri.py          → FERI_results.xlsx
  2. python milp_dat_generator_v2.py → assembly_line.dat + precedence_diagram.png
  3. CPLEX OPL: assembly_line.mod + assembly_line.dat
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────────────────────

_AHP_FILE = Path(__file__).parent / "experts_answer.xlsx"
try:
    from ahp_weights import get_weights as _ahp_get_weights
    WEIGHTS = _ahp_get_weights(str(_AHP_FILE), verbose=False)
    _WEIGHTS_SOURCE = f"AHP ({_AHP_FILE.name})"
except Exception as _ahp_err:
    WEIGHTS = {"energy": 0.35, "reba": 0.35, "borg": 0.30}
    _WEIGHTS_SOURCE = "default (hardcoded)"

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"

SPREAD = {
    "time":   0.05,   # ±5% of measured value
    "energy": 0.05,   # ±5% of measured value
    "reba":   1.0,    # ±1 scale points
    "borg":   1.0,    # ±1 scale point
}

RISK_LABELS = {
    (0.0,  0.2):  ("Low",       "#27AE60"),
    (0.2,  0.4):  ("Moderate",  "#2ECC71"),
    (0.4,  0.6):  ("Medium",    "#F4A261"),
    (0.6,  0.8):  ("High",      "#E76F51"),
    (0.8,  1.01): ("Very High", "#C0392B"),
}


# ─────────────────────────────────────────────────────────────
# 1. Load data from two-sheet Excel
# ─────────────────────────────────────────────────────────────

def load_data(path: str):

    # ── Tasks sheet (has title row on row 0 → real headers on row 1)
    df_tasks = pd.read_excel(path, sheet_name="Tasks", header=1)
    df_tasks.columns = df_tasks.columns.str.strip()

    df_tasks = df_tasks[["Task ID", "Station", "Task Name", "Time (s)",
                         "Energy (kcal)", "REBA", "Borg"]].copy()
    df_tasks.columns = ["task_id", "station", "task_name", "time", "energy", "reba", "borg"]
    df_tasks = df_tasks.dropna(subset=["task_id"]).reset_index(drop=True)
    df_tasks["task_id"] = df_tasks["task_id"].astype(int)

    # ⚠️ Fix decimal commas (VERY IMPORTANT for Portugal data)
    df_tasks["time"]   = df_tasks["time"].astype(str).str.replace(",", ".").astype(float)
    df_tasks["energy"] = df_tasks["energy"].astype(str).str.replace(",", ".").astype(float)

    # ── Precedence sheet (has title row on row 0 → real headers on row 1)
    df_prec = pd.read_excel(path, sheet_name="Precedence", header=1)
    df_prec.columns = df_prec.columns.str.strip()

    df_prec = df_prec[["Predecessor ID", "Successor ID"]].copy()
    df_prec.columns = ["pred_id", "succ_id"]
    df_prec = df_prec.dropna().reset_index(drop=True)
    df_prec["pred_id"] = df_prec["pred_id"].astype(int)
    df_prec["succ_id"] = df_prec["succ_id"].astype(int)
    return df_tasks, df_prec


# ─────────────────────────────────────────────────────────────
# 2–6. FERI pipeline
# ─────────────────────────────────────────────────────────────

def fuzzify(df):
    delta_t     = df["time"] * SPREAD["time"]
    df["t_l"]   = df["time"] - delta_t
    df["t_m"]   = df["time"]
    df["t_u"]   = df["time"] + delta_t
    delta_e     = df["energy"] * SPREAD["energy"]
    df["e_l"]   = df["energy"] - delta_e
    df["e_m"]   = df["energy"]
    df["e_u"]   = df["energy"] + delta_e
    df["r_l"]   = (df["reba"] - SPREAD["reba"]).clip(lower=0)
    df["r_m"]   = df["reba"].astype(float)
    df["r_u"]   = df["reba"] + SPREAD["reba"]
    df["b_l"]   = (df["borg"] - SPREAD["borg"]).clip(lower=0)
    df["b_m"]   = df["borg"].astype(float)
    df["b_u"]   = df["borg"] + SPREAD["borg"]
    return df

def normalize(df):
    max_t = df["t_u"].max()
    max_e = df["e_u"].max()
    max_r = df["r_u"].max()
    max_b = df["b_u"].max()
    for s in ["l", "m", "u"]:
        df[f"tn_{s}"] = df[f"t_{s}"] / max_t
        df[f"en_{s}"] = df[f"e_{s}"] / max_e
        df[f"rn_{s}"] = df[f"r_{s}"] / max_r
        df[f"bn_{s}"] = df[f"b_{s}"] / max_b
    return df

def aggregate(df):
    wE, wR, wB = WEIGHTS["energy"], WEIGHTS["reba"], WEIGHTS["borg"]
    for s in ["l", "m", "u"]:
        df[f"eri_{s}"] = (wE * df[f"en_{s}"]
                        + wR * df[f"rn_{s}"]
                        + wB * df[f"bn_{s}"])
    return df

def defuzzify(df):
    df["ERI"] = (df["eri_l"] + df["eri_m"] + df["eri_u"]) / 3
    return df

def classify(eri):
    for (lo, hi), (label, color) in RISK_LABELS.items():
        if lo <= eri < hi:
            return label, color
    return "Very High", "#C0392B"

def add_classification(df):
    df[["risk_level", "color"]] = df["ERI"].apply(
        lambda v: pd.Series(classify(v))
    )
    return df

def compute_feri(df_tasks: pd.DataFrame) -> pd.DataFrame:
    df = df_tasks.copy()
    df = fuzzify(df)
    df = normalize(df)
    df = aggregate(df)
    df = defuzzify(df)
    df = add_classification(df)
    return df


# ─────────────────────────────────────────────────────────────
# 7. Console report
# ─────────────────────────────────────────────────────────────

def print_report(df):
    cols = ["task_id", "station", "task_name", "time", "tn_l", "tn_m", "tn_u", "energy", "reba", "borg",
            "eri_l", "eri_m", "eri_u", "ERI", "risk_level"]
    out = df[cols].copy()
    out.columns = ["ID", "Station", "Task", "Time(s)", "TN_l", "TN_m", "TN_u", "Energy", "REBA", "Borg",
                   "ERI_l", "ERI_m", "ERI_u", "ERI", "Risk"]
    print("\n" + "="*125)
    print("  FUZZY ERGONOMIC RISK INDEX (FERI) — Results")
    print("="*125)
    print(f"  Weights : Energy={WEIGHTS['energy']:.4f}  REBA={WEIGHTS['reba']:.4f}  "
          f"Borg={WEIGHTS['borg']:.4f}   [source: {_WEIGHTS_SOURCE}]")
    print(f"  Spreads : Time±{SPREAD['time']*100:.0f}%  Energy±{SPREAD['energy']*100:.0f}%  "
          f"REBA±{SPREAD['reba']}  Borg±{SPREAD['borg']}")
    print()
    print(out.to_string(index=False, float_format="{:.4f}".format))
    print()
    counts = df["risk_level"].value_counts()
    order  = ["Low", "Moderate", "Medium", "High", "Very High"]
    print("  Overall risk distribution:")
    for lvl in order:
        n = counts.get(lvl, 0)
        print(f"    {lvl:<12} {'█'*n} ({n})")
    print()

    # ── Per-station summary
    print("  ── Station summary ──")
    station_grp = df.groupby("station")
    for st, grp in station_grp:
        avg_eri = grp["ERI"].mean()
        risk_label, _ = classify(avg_eri)
        dist = grp["risk_level"].value_counts()
        dist_str = "  ".join(f"{lvl}:{dist.get(lvl,0)}" for lvl in order if dist.get(lvl,0) > 0)
        print(f"    Station {st:>2} | Tasks: {len(grp):>2} | Avg ERI: {avg_eri:.4f} "
              f"[{risk_label}] | {dist_str}")
    print("="*125)


# ─────────────────────────────────────────────────────────────
# 8. Visualisation
# ─────────────────────────────────────────────────────────────

def plot_results(df, out_path=None):
    n_tasks   = len(df)
    fig_width = max(22, n_tasks * 0.22)   # give the per-task row enough room to breathe
    fig = plt.figure(figsize=(fig_width, 16))
    gs  = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1], hspace=0.75, wspace=0.3)
    fig.suptitle("Fuzzy Ergonomic Risk Index (FERI)",
                 fontsize=16, fontweight="bold")
    palette = {l: c for (_, (l, c)) in RISK_LABELS.items()}
    labels  = [f"T{r.task_id}" for _, r in df.iterrows()]

    # 8.1 ERI bar + fuzzy interval — full-width row so every task label is legible
    ax = fig.add_subplot(gs[0, :])
    ax.bar(labels, df["ERI"], color=df["color"], edgecolor="white", linewidth=0.5,
           yerr=[df["ERI"]-df["eri_l"], df["eri_u"]-df["ERI"]], capsize=3,
           error_kw={"elinewidth": 1.2, "ecolor": "gray", "alpha": 0.7})
    ax.set_ylim(0, 1.05)
    ax.axhline(0.6, color="#E76F51", lw=1, ls="--", alpha=0.6)
    ax.axhline(0.4, color="#F4A261", lw=1, ls="--", alpha=0.6)
    ax.set_title("ERI per Task (with fuzzy interval)", fontsize=11)
    ax.set_xlabel("Task"); ax.set_ylabel("ERI")
    rotation = 90 if n_tasks > 30 else 55
    fontsize = 7 if n_tasks > 60 else 8
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation,
                        ha="center" if rotation == 90 else "right", fontsize=fontsize)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[mpatches.Patch(color=c, label=l)
                        for (_, (l, c)) in RISK_LABELS.items()], fontsize=7)

    # 8.2 Risk distribution
    ax = fig.add_subplot(gs[1, 0])
    order  = ["Low", "Moderate", "Medium", "High", "Very High"]
    counts = df["risk_level"].value_counts().reindex(order, fill_value=0)
    ax.bar(counts.index, counts.values,
           color=[palette[l] for l in counts.index], edgecolor="white")
    ax.set_title("Distribution by Risk Level", fontsize=11)
    ax.set_xlabel("Level"); ax.set_ylabel("Tasks")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 8.3 Scatter REBA vs Energy, size ∝ Borg
    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(df["reba"], df["energy"], s=df["borg"]*25,
                    c=df["ERI"], cmap="RdYlGn_r", vmin=0, vmax=1,
                    edgecolors="gray", linewidth=0.5, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="ERI")
    ax.set_title("REBA vs Energy  (size ∝ Borg)", fontsize=11)
    ax.set_xlabel("REBA"); ax.set_ylabel("Energy (kcal)")
    for _, r in df.iterrows():
        ax.annotate(f"T{r.task_id}", (r["reba"], r["energy"]),
                    fontsize=7, ha="left", va="bottom", alpha=0.7)
    ax.grid(alpha=0.3)

    # 8.4 Top-10 highest ERI
    ax = fig.add_subplot(gs[1, 2])
    top = df.nlargest(min(10, len(df)), "ERI").copy()
    top["label"] = top["task_id"].apply(lambda i: f"T{i}")
    hbars = ax.barh(top["label"][::-1], top["ERI"][::-1],
                    color=top["color"][::-1], edgecolor="white")
    ax.set_xlim(0, 1.0)
    ax.set_title("Top Tasks — Highest ERI", fontsize=11)
    ax.set_xlabel("ERI")
    for bar, val in zip(hbars, top["ERI"][::-1]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.axvline(0.6, color="#E76F51", lw=1, ls="--", alpha=0.6)
    ax.grid(axis="x", alpha=0.3)

    # 8.5 Average ERI per station (bar chart)
    ax = fig.add_subplot(gs[2, 0])
    st_summary = df.groupby("station")["ERI"].mean().sort_index()
    st_labels  = [f"S{s}" for s in st_summary.index]
    st_colors  = [classify(v)[1] for v in st_summary.values]
    bars = ax.bar(st_labels, st_summary.values, color=st_colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.6, color="#E76F51", lw=1, ls="--", alpha=0.6)
    ax.axhline(0.4, color="#F4A261", lw=1, ls="--", alpha=0.6)
    ax.set_title("Average ERI per Station", fontsize=11)
    ax.set_xlabel("Station"); ax.set_ylabel("Avg ERI")
    for bar, val in zip(bars, st_summary.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    st_rotation = 45 if len(st_labels) <= 20 else 90
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(st_rotation)
        lbl.set_ha("right" if st_rotation < 90 else "center")
        lbl.set_fontsize(8)
    ax.grid(axis="y", alpha=0.3)

    # 8.6 Stacked risk distribution per station
    ax = fig.add_subplot(gs[2, 1])
    stations = sorted(df["station"].unique())
    bottoms  = np.zeros(len(stations))
    for lvl in order:
        heights = [
            (df[df["station"] == s]["risk_level"] == lvl).sum()
            for s in stations
        ]
        ax.bar([f"S{s}" for s in stations], heights,
               bottom=bottoms, color=palette[lvl],
               label=lvl, edgecolor="white", linewidth=0.5)
        bottoms += np.array(heights, dtype=float)
    ax.set_title("Risk Distribution per Station", fontsize=11)
    ax.set_xlabel("Station"); ax.set_ylabel("Tasks")
    ax.legend(fontsize=8, loc="upper right")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(st_rotation)
        lbl.set_ha("right" if st_rotation < 90 else "center")
        lbl.set_fontsize(8)
    ax.grid(axis="y", alpha=0.3)

    fig.add_subplot(gs[2, 2]).axis("off")   # unused cell

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved : {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# 9. Export FERI_results.xlsx  (Tasks + Precedence sheets)
# ─────────────────────────────────────────────────────────────

def export_excel(df_feri: pd.DataFrame, df_prec: pd.DataFrame, out_path: str):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb  = Workbook()
    ws  = wb.active
    ws.title = "FERI Results"

    thin   = Side(style="thin", color="AAAAAA")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")

    def hdr(ws, row, col, value, bg="0D1B2A", fg="00B4D8"):
        c = ws.cell(row=row, column=col, value=value)
        c.font      = Font(bold=True, color=fg, size=10)
        c.fill      = PatternFill("solid", start_color=bg)
        c.alignment = center
        c.border    = border

    def dat(ws, row, col, value, fmt=None, bold=False, fill=None):
        c = ws.cell(row=row, column=col, value=value)
        c.alignment = center
        c.border    = border
        c.font      = Font(bold=bold, size=10)
        if fmt:   c.number_format = fmt
        if fill:  c.fill = PatternFill("solid", start_color=fill)

    # ── FERI Results sheet
    headers = ["Task ID", "Station", "Task Name", "Time (s)", "TN_l", "TN_m", "TN_u", "Energy (kcal)", "REBA", "Borg",
               "ERI_l", "ERI_m", "ERI_u", "ERI (crisp)", "Risk Level"]
    for ci, h in enumerate(headers, 1):
        hdr(ws, 1, ci, h)

    risk_fills = {
        "Low":       "ABEBC6",
        "Moderate":  "D5F5E3",
        "Medium":    "FAD7A0",
        "High":      "F0B27A",
        "Very High": "E74C3C",
    }

    cols = ["task_id", "station", "task_name", "time", "tn_l", "tn_m", "tn_u", "energy", "reba", "borg",
            "eri_l", "eri_m", "eri_u", "ERI", "risk_level"]
    fmts = [None, None, None, "0.00", "0.0000", "0.0000", "0.0000", "0.0000", None, None,
            "0.0000", "0.0000", "0.0000", "0.0000", None]

    for ri, row in enumerate(df_feri[cols].itertuples(index=False), 2):
        for ci, (val, fmt) in enumerate(zip(row, fmts), 1):
            fill = None
            if ci == 16:
                fill = risk_fills.get(val)
            align = Alignment(horizontal="left", vertical="center") if ci == 3 else center
            c = ws.cell(row=ri, column=ci, value=val)
            c.alignment = align; c.border = border
            c.font = Font(bold=(ci == 16), size=10)
            if fmt:  c.number_format = fmt
            if fill: c.fill = PatternFill("solid", start_color=fill)

    col_widths = [9, 10, 32, 10, 10, 10, 10, 14, 8, 8, 10, 10, 10, 12, 12]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"

    # ── Station Summary sheet
    ws4 = wb.create_sheet("Station Summary")
    st_headers = ["Station", "Num Tasks", "Total Time (s)", "Avg Energy (kcal)",
                  "Avg REBA", "Avg Borg", "Avg ERI", "Max ERI", "Risk Level"]
    for ci, h in enumerate(st_headers, 1):
        hdr(ws4, 1, ci, h, bg="1A5276")

    order = ["Low", "Moderate", "Medium", "High", "Very High"]
    for ri, (st, grp) in enumerate(df_feri.groupby("station"), 2):
        avg_eri = grp["ERI"].mean()
        risk_label, _ = classify(avg_eri)
        fill = risk_fills.get(risk_label)
        row_vals = [
            int(st),
            len(grp),
            grp["time"].sum(),
            grp["energy"].mean(),
            grp["reba"].mean(),
            grp["borg"].mean(),
            avg_eri,
            grp["ERI"].max(),
            risk_label,
        ]
        row_fmts = [None, None, "0.00", "0.0000", "0.00", "0.00", "0.0000", "0.0000", None]
        for ci, (val, fmt) in enumerate(zip(row_vals, row_fmts), 1):
            c = ws4.cell(row=ri, column=ci, value=val)
            c.alignment = center; c.border = border
            c.font = Font(bold=(ci == 9), size=10)
            if fmt: c.number_format = fmt
            if ci == 9 and fill:
                c.fill = PatternFill("solid", start_color=fill)

    st_col_widths = [10, 12, 16, 18, 12, 12, 12, 12, 14]
    for i, w in enumerate(st_col_widths, 1):
        ws4.column_dimensions[get_column_letter(i)].width = w
    ws4.freeze_panes = "A2"

    # ── Precedence sheet (pass-through)
    ws2 = wb.create_sheet("Precedence")
    for ci, h in enumerate(["Predecessor ID", "Successor ID"], 1):
        hdr(ws2, 1, ci, h, bg="1A5276")
    for ri, row in enumerate(df_prec.itertuples(index=False), 2):
        dat(ws2, ri, 1, int(row.pred_id))
        dat(ws2, ri, 2, int(row.succ_id))
    ws2.column_dimensions["A"].width = 16
    ws2.column_dimensions["B"].width = 14

    # ── Parameters sheet
    ws3 = wb.create_sheet("Parameters")
    for ci, h in enumerate(["Parameter", "Value"], 1):
        hdr(ws3, 1, ci, h, bg="2C3E50")
    params = [
        ("Weight Energy",        WEIGHTS["energy"]),
        ("Weight REBA",          WEIGHTS["reba"]),
        ("Weight Borg",          WEIGHTS["borg"]),
        ("Spread Energy (±%)",   SPREAD["energy"]),
        ("Spread REBA (±pts)",   SPREAD["reba"]),
        ("Spread Borg (±pts)",   SPREAD["borg"]),
        ("Total tasks",          len(df_feri)),
        ("Total precedences",    len(df_prec)),
        ("CT_max (sum of times)", df_feri["time"].sum()),
        ("EL_max (sum ERI*time)", (df_feri["ERI"] * df_feri["time"]).sum()),
    ]
    for i, (k, v) in enumerate(params, 2):
        c = ws3.cell(row=i, column=1, value=k)
        c.font = Font(bold=True, size=10); c.border = border
        c.alignment = Alignment(horizontal="left", vertical="center")
        dat(ws3, i, 2, round(v, 6) if isinstance(v, float) else v)
    ws3.column_dimensions["A"].width = 28
    ws3.column_dimensions["B"].width = 18

    wb.save(out_path)
    print(f"  Excel saved : {out_path}")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _here   = Path(__file__).parent
    INPUT   = _here / "example_dataset.xlsx"
    OUT_XL  = _here / "FERI_results.xlsx"
    OUT_FIG = _here / "FERI_charts.png"

    print("Loading data …")
    df_tasks, df_prec = load_data(INPUT)
    print(f"  {len(df_tasks)} tasks  |  {len(df_prec)} precedence pairs")

    print("Computing FERI …")
    df_feri = compute_feri(df_tasks)

    print_report(df_feri)
    plot_results(df_feri, out_path=OUT_FIG)
    export_excel(df_feri, df_prec, out_path=OUT_XL)
    print("\nDone. Next: run milp_dat_generator_v2.py")
