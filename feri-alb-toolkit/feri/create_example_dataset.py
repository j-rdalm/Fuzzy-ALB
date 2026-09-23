"""
create_example_dataset.py
==========================
Generates example_dataset.xlsx with:
  Sheet 1 — "Tasks"           : task data (ID, Station, Name, Time, Energy, REBA, Borg)
  Sheet 2 — "Precedence"      : (predecessor_id, successor_id) pairs
  Sheet 3 — "Station Summary" : tasks per station with aggregated metrics
  Sheet 4 — "Precedence Chart": embedded precedence DAG diagram (PNG)

Usage:
  Set N_TASKS and N_STATIONS below, then run:
    python create_example_dataset.py

Parameters:
  N_TASKS    — total number of tasks to generate (e.g. 20, 50, 70, 100)
  N_STATIONS — number of workstations to assign tasks to
  SEED       — random seed for reproducibility
"""

import math
import random
import io
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImage
PILImage.MAX_IMAGE_PIXELS = None   # disable decompression bomb check for our own output
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as XLImage

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  ← change these
# ─────────────────────────────────────────────────────────────
N_TASKS    = 200       #total tasks  (e.g. 20, 50, 70, 100)
N_STATIONS = 10        #number of workstations
SEED       = 42       #reproducibility
OUT        = Path(__file__).parent / "example_dataset.xlsx"
# ─────────────────────────────────────────────────────────────
# 1. Task name fragments (combinable for variety)
# ─────────────────────────────────────────────────────────────
VERBS = [
    "Mount", "Attach", "Install", "Secure", "Connect",
    "Route", "Align", "Tighten", "Insert", "Apply",
    "Fit", "Place", "Fix", "Fasten", "Assemble",
    "Position", "Clamp", "Seal", "Wire", "Test",
]
OBJECTS = [
    "base frame", "side panel", "support bracket", "cable conduit",
    "power cable", "terminal block", "fan unit", "motor assembly",
    "motor shaft", "control panel", "cover plate", "wiring harness",
    "grounding strap", "sensor unit", "cooling duct", "retaining clip",
    "junction box", "front bezel", "rear cover", "safety guard",
    "drive belt", "sprocket", "gasket", "insulation pad", "heat sink",
    "relay module", "fuse holder", "signal cable", "data connector",
    "output shaft", "bearing housing", "oil seal", "pressure valve",
    "limit switch", "proximity sensor", "encoder disc", "coupling flange",
    "bus bar", "circuit board", "display module", "keypad unit",
    "emergency stop", "indicator light", "power supply", "transformer",
    "filter element", "lubricant port", "drain plug", "inspection hatch",
]
def generate_task_names(n, seed):
    rng = random.Random(seed)
    used, names = set(), []
    verbs  = VERBS  * math.ceil(n / len(VERBS))
    objs   = OBJECTS * math.ceil(n / len(OBJECTS))
    rng.shuffle(verbs)
    rng.shuffle(objs)
    for v, o in zip(verbs, objs):
        name = f"{v} {o}"
        if name not in used:
            used.add(name)
            names.append(name)
        if len(names) == n:
            break
    # fill remaining if needed
    extra = 0
    while len(names) < n:
        names.append(f"Task operation {extra+1}")
        extra += 1
    return names
# ─────────────────────────────────────────────────────────────
# 2. Generate task data
# ─────────────────────────────────────────────────────────────
def generate_tasks(n, n_stations, seed):
    rng = np.random.default_rng(seed)
    names = generate_task_names(n, seed)

    # Realistic distributions
    times   = rng.uniform(10, 75, n).round(1)
    energy  = (times * rng.uniform(0.006, 0.012, n)).round(4)
    reba    = rng.integers(1, 8, n)          # 1–7
    borg    = (reba + rng.integers(-1, 2, n)).clip(1, 10)

    # Assign stations: distribute tasks roughly evenly, then shuffle
    station_ids = np.array(sorted(
        [i % n_stations + 1 for i in range(n)]
    ))
    rng.shuffle(station_ids)  # randomise assignment

    rows = []
    for i in range(n):
        rows.append((
            i + 1,
            int(station_ids[i]),
            names[i],
            float(times[i]),
            float(energy[i]),
            int(reba[i]),
            int(borg[i]),
        ))
    return rows

# ─────────────────────────────────────────────────────────────
# 3. Generate a realistic DAG (precedence graph)
#    Strategy: layered DAG — tasks organised in L layers,
#    edges flow forward, density controlled by EDGE_PROB
# ─────────────────────────────────────────────────────────────
def generate_dag(n, seed, edge_prob=0.25, min_edges_per_node=1):
    rng = random.Random(seed + 1)
    # Split tasks into layers (like a production flow)
    n_layers = max(4, n // 6)
    layers   = [[] for _ in range(n_layers)]
    for i in range(1, n + 1):
        layers[i % n_layers].append(i)
    # Shuffle within layers for variety
    for l in layers:
        rng.shuffle(l)

    edges = set()
    layer_list = [l for l in layers if l]

    # Connect consecutive layers
    for li in range(len(layer_list) - 1):
        src_layer = layer_list[li]
        dst_layer = layer_list[li + 1]
        for dst in dst_layer:
            # Each node in next layer gets at least one predecessor
            pred = rng.choice(src_layer)
            edges.add((pred, dst))
        # Additional random cross-edges within reachable range
        for src in src_layer:
            for dst in dst_layer:
                if (src, dst) not in edges and rng.random() < edge_prob:
                    edges.add((src, dst))

    # Optionally add skip-layer edges for realism
    for li in range(len(layer_list) - 2):
        for src in layer_list[li]:
            for dst in layer_list[li + 2]:
                if rng.random() < edge_prob * 0.3:
                    edges.add((src, dst))

    return sorted(edges)

# ─────────────────────────────────────────────────────────────
# 4. Precedence diagram (matplotlib → PNG bytes)
# ─────────────────────────────────────────────────────────────
def build_precedence_diagram(tasks, edges, n_stations):
    """Return PNG bytes of the precedence graph."""
    n = len(tasks)
    station_map = {t[0]: t[1] for t in tasks}  # task_id → station

    # Station colour palette
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(n_stations)
    st_colors = {s: cmap(s - 1) for s in range(1, n_stations + 1)}

    # ── Layered layout (Sugiyama-style via longest-path ranking)
    # Compute rank (layer) for each node = longest path from a source
    succ = {t[0]: [] for t in tasks}
    pred = {t[0]: [] for t in tasks}
    for (p, s) in edges:
        succ[p].append(s)
        pred[s].append(p)

    rank = {}
    def get_rank(node):
        if node in rank:
            return rank[node]
        if not pred[node]:
            rank[node] = 0
        else:
            rank[node] = max(get_rank(p) for p in pred[node]) + 1
        return rank[node]
    for t in tasks:
        get_rank(t[0])

    max_rank  = max(rank.values()) if rank else 0
    layers    = [[] for _ in range(max_rank + 1)]
    for tid, r in rank.items():
        layers[r].append(tid)

    # Node positions
    pos = {}
    x_gap = 3.5
    for r, layer in enumerate(layers):
        y_gap = 1.8
        for j, tid in enumerate(layer):
            x = r * x_gap
            y = -(j - (len(layer) - 1) / 2) * y_gap
            pos[tid] = (x, y)

    # Figure size — cap to avoid PIL decompression bomb at high node counts
    fig_w = min(28, max(14, (max_rank + 1) * 2.2))
    fig_h = min(18, max(8,  max(len(l) for l in layers) * 1.4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("equal")
    ax.axis("off")
    title = f"Precedence Graph  ({n} tasks, {n_stations} stations)"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # Draw edges
    for (p, s) in edges:
        x1, y1 = pos[p]
        x2, y2 = pos[s]
        ax.annotate("",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#999999",
                            lw=0.9, mutation_scale=10))

    # Draw nodes
    node_r = 0.55
    for t in tasks:
        tid = t[0]
        st  = station_map[tid]
        x, y = pos[tid]
        col = st_colors[st]
        circle = plt.Circle((x, y), node_r, color=col, ec="white",
                             lw=1.2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(tid), ha="center", va="center",
                fontsize=7 if n > 50 else 8,
                fontweight="bold", color="white", zorder=4)

    # Legend (stations)
    handles = [mpatches.Patch(color=st_colors[s], label=f"Station {s}")
               for s in range(1, n_stations + 1)]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.9, title="Stations", title_fontsize=8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=72, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────────────────────
# 5. Excel helpers
# ─────────────────────────────────────────────────────────────
_thin = Side(style="thin", color="AAAAAA")
_border = Border(left=_thin, right=_thin, top=_thin, bottom=_thin)
_center = Alignment(horizontal="center", vertical="center")

def hdr(ws, row, col, value, bg="0D1B2A", fg="00B4D8"):
    c = ws.cell(row=row, column=col, value=value)
    c.font      = Font(bold=True, color=fg, size=10, name="Arial")
    c.fill      = PatternFill("solid", start_color=bg)
    c.alignment = _center
    c.border    = _border

def dat(ws, row, col, value, fmt=None, bold=False, bg=None, left=False):
    c = ws.cell(row=row, column=col, value=value)
    c.alignment = Alignment(horizontal="left" if left else "center",
                            vertical="center")
    c.border = _border
    c.font   = Font(bold=bold, size=10, name="Arial")
    if fmt: c.number_format = fmt
    if bg:  c.fill = PatternFill("solid", start_color=bg)

# ─────────────────────────────────────────────────────────────
# 6. Write Excel
# ─────────────────────────────────────────────────────────────
def write_excel(tasks, edges, n_stations, out_path):
    from collections import defaultdict

    wb = Workbook()

    # task lookup
    task_map = {t[0]: t for t in tasks}
    n = len(tasks)

    reba_fill = lambda v: ("ABEBC6" if v <= 2 else ("FAD7A0" if v <= 4 else "F0B27A"))

    # ── Sheet 1: Tasks
    ws1 = wb.active
    ws1.title = "Tasks"
    note = (f"Auto-generated dataset  |  {n} tasks  |  {n_stations} stations  |  "
            f"{len(edges)} precedence constraints")
    ws1["A1"] = note
    ws1["A1"].font = Font(italic=True, color="555555", size=9, name="Arial")
    ws1.merge_cells(f"A1:G1")

    hdrs1 = ["Task ID", "Station", "Task Name", "Time (s)",
             "Energy (kcal)", "REBA", "Borg"]
    for ci, h in enumerate(hdrs1, 1):
        hdr(ws1, 2, ci, h)

    fmts1 = [None, None, None, "0.00", "0.0000", None, None]
    for ri, t in enumerate(tasks, 3):
        vals = list(t)
        for ci, (v, f) in enumerate(zip(vals, fmts1), 1):
            bg = None
            if ci == 6:  bg = reba_fill(v)    # REBA colour
            left = (ci == 3)
            dat(ws1, ri, ci, v, fmt=f, bg=bg, left=left)

    col_w1 = [10, 10, 36, 10, 14, 8, 8]
    for i, w in enumerate(col_w1, 1):
        ws1.column_dimensions[get_column_letter(i)].width = w
    ws1.freeze_panes = "A3"
    ws1.auto_filter.ref = f"A2:{get_column_letter(len(hdrs1))}2"
    ws1.row_dimensions[1].height = 20
    ws1.row_dimensions[2].height = 18

    # ── Sheet 2: Precedence
    ws2 = wb.create_sheet("Precedence")
    note2 = "Each row defines one precedence constraint (predecessor must finish before successor starts)."
    ws2["A1"] = note2
    ws2["A1"].font = Font(italic=True, color="555555", size=9, name="Arial")
    ws2.merge_cells("A1:C1")

    hdrs2 = ["Predecessor ID", "Successor ID", "Description (optional)"]
    for ci, h in enumerate(hdrs2, 1):
        hdr(ws2, 2, ci, h, bg="1A5276")

    for ri, (pred_id, succ_id) in enumerate(edges, 3):
        dat(ws2, ri, 1, pred_id)
        dat(ws2, ri, 2, succ_id)
        pname = task_map[pred_id][2]
        sname = task_map[succ_id][2]
        c = ws2.cell(row=ri, column=3, value=f"{pname}  →  {sname}")
        c.border    = _border
        c.font      = Font(color="555555", size=9, italic=True, name="Arial")
        c.alignment = Alignment(horizontal="left", vertical="center")

    ws2.column_dimensions["A"].width = 16
    ws2.column_dimensions["B"].width = 14
    ws2.column_dimensions["C"].width = 60
    ws2.freeze_panes = "A3"
    ws2.row_dimensions[1].height = 20

    # ── Sheet 3: Station Summary
    ws3 = wb.create_sheet("Station Summary")
    ws3["A1"] = "Station-level summary (auto-generated)"
    ws3["A1"].font = Font(italic=True, color="555555", size=9, name="Arial")
    ws3.merge_cells("A1:G1")

    hdrs3 = ["Station", "Num Tasks", "Total Time (s)",
             "Avg Energy (kcal)", "Avg REBA", "Avg Borg", "Task IDs"]
    for ci, h in enumerate(hdrs3, 1):
        hdr(ws3, 2, ci, h, bg="154360")

    station_groups = defaultdict(list)
    for t in tasks:
        station_groups[t[1]].append(t)

    for ri, st in enumerate(sorted(station_groups.keys()), 3):
        grp = station_groups[st]
        tids     = sorted(t[0] for t in grp)
        tot_time = sum(t[3] for t in grp)
        avg_en   = sum(t[4] for t in grp) / len(grp)
        avg_reba = sum(t[5] for t in grp) / len(grp)
        avg_borg = sum(t[6] for t in grp) / len(grp)
        dat(ws3, ri, 1, st, bold=True)
        dat(ws3, ri, 2, len(grp))
        dat(ws3, ri, 3, tot_time,  fmt="0.00")
        dat(ws3, ri, 4, avg_en,    fmt="0.0000")
        dat(ws3, ri, 5, avg_reba,  fmt="0.00")
        dat(ws3, ri, 6, avg_borg,  fmt="0.00")
        c = ws3.cell(row=ri, column=7,
                     value=", ".join(str(i) for i in tids))
        c.border    = _border
        c.font      = Font(size=9, name="Arial")
        c.alignment = Alignment(horizontal="left", vertical="center",
                                wrap_text=True)

    col_w3 = [10, 12, 16, 18, 12, 12, 50]
    for i, w in enumerate(col_w3, 1):
        ws3.column_dimensions[get_column_letter(i)].width = w
    ws3.freeze_panes = "A3"
    ws3.row_dimensions[1].height = 20

    # ── Sheet 4: Precedence Chart
    ws4 = wb.create_sheet("Precedence Chart")
    ws4["A1"] = "Precedence Graph (nodes coloured by station)"
    ws4["A1"].font = Font(bold=True, size=12, name="Arial", color="0D1B2A")
    ws4.row_dimensions[1].height = 22

    print("  Building precedence diagram …")
    img_buf = build_precedence_diagram(tasks, edges, n_stations)
    img     = XLImage(img_buf)
    # Scale to fit nicely (openpyxl uses EMU; 1 cm ≈ 360000 EMU)
    img.anchor = "A2"
    ws4.add_image(img)

    wb.save(out_path)
    print(f"  Excel saved : {out_path}")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating dataset: {N_TASKS} tasks, {N_STATIONS} stations …")

    tasks = generate_tasks(N_TASKS, N_STATIONS, SEED)
    edges = generate_dag(N_TASKS, SEED)

    print(f"  Tasks       : {len(tasks)}")
    print(f"  Precedences : {len(edges)}")

    write_excel(tasks, edges, N_STATIONS, OUT)
    print(f"\nDone.  →  {OUT}")
    print("Next: run fuzzy_eri.py")
