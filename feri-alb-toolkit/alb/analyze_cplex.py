#!/usr/bin/env python3
"""
analyze_cplex.py
================
Parse CPLEX output (outputCPLEX.rtf) from the sensitivity experiment and produce:
  - Fuzzy membership function charts per weight pair
  - Weight sensitivity line plots
  - CT vs max-ERI Pareto scatter
  - Objective heatmaps (weight pairs × scenarios)
  - Operator-level heatmaps for the balanced scenario
  - Excel summary with pivot tables

Run metadata is read automatically from the run filename header when present,
with fallback to the echoed scenario_tag field.
The parser accepts both the new filename-led format and the legacy OBJ-1/OBJ-2
block format.

Usage:
    python analyze_cplex.py [path/to/outputCPLEX.rtf]
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
RTF_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "outputCPLEX.rtf"
OUT_DIR  = BASE_DIR / "cplex_analysis"
OUT_DIR.mkdir(exist_ok=True)

# ─── Colour scheme ────────────────────────────────────────────────────────────

WT_PALETTE = {
    1.00: "#1f77b4",   # blue     — time-only
    0.75: "#ff7f0e",   # orange
    0.50: "#2ca02c",   # green    — balanced
    0.25: "#d62728",   # red
    0.00: "#9467bd",   # purple   — ergonomics-only
}
SC_MARKER = {"optimistic": "v", "midpoint": "D", "pessimistic": "^"}
SC_LS     = {"optimistic": "--", "midpoint": "-", "pessimistic": ":"}
CUT_LS    = {0.0: "-", 0.5: "--", 1.0: ":"}


# ─── 1. RTF → plain text ──────────────────────────────────────────────────────

def strip_rtf(rtf: str) -> str:
    s = re.sub(r"\\'([0-9a-fA-F]{2})",
               lambda m: chr(int(m.group(1), 16)), rtf)
    s = re.sub(r'\\[a-zA-Z]+\*?\-?\d*[ ]?', '', s)
    s = s.replace('{', '').replace('}', '')
    s = re.sub(r'\\\s*$', '', s, flags=re.MULTILINE)
    s = s.replace('\\', '')
    clean = []
    for ln in s.splitlines():
        ln = re.sub(r'[\xa0 \t]+', ' ', ln).strip()
        if ln:
            clean.append(ln)
    return '\n'.join(clean)


# ─── 2. Split into solution blocks ────────────────────────────────────────────

RUN_HDR = re.compile(r'^(assembly_line_[A-Za-z0-9_]+)$', re.IGNORECASE | re.MULTILINE)
LEGACY_BLOCK_HDR = re.compile(r'(?:// *[-=]+ *)?(OBJ-[12])\s*:', re.IGNORECASE)


def _parse_run_name(run_name: str):
    """Parse 'assembly_line_wt000_we100_cut00_optimistic' into metadata."""
    m = re.match(
        r'^assembly_line_wt(?P<wt>\d{3})_we(?P<we>\d{3})_cut(?P<cut>\d{2})'
        r'(?:_(?P<scenario>optimistic|pessimistic|midpoint))?$',
        run_name,
        re.IGNORECASE,
    )
    if not m:
        return None, None, None, None

    wt = int(m.group('wt')) / 100.0
    we = int(m.group('we')) / 100.0
    cut_code = m.group('cut')
    cut_map = {'00': 0.0, '05': 0.5, '10': 1.0}
    cut = cut_map.get(cut_code, int(cut_code) / 10.0)
    scenario = m.group('scenario').lower() if m.group('scenario') else None
    if scenario is None and cut == 1.0:
        scenario = 'midpoint'   # alpha = 1: single midpoint instance (no scenario suffix in the file name)
    return wt, we, cut, scenario

def split_blocks(text: str):
    hits = [(m.start(), m.group(1)) for m in RUN_HDR.finditer(text)]
    if hits:
        for i, (pos, lbl) in enumerate(hits):
            end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
            yield lbl, text[pos:end]
        return

    hits = [(m.start(), m.group(1)) for m in LEGACY_BLOCK_HDR.finditer(text)]
    for i, (pos, lbl) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        yield lbl, text[pos:end]


# ─── 3. Parse one block ───────────────────────────────────────────────────────

NUM = r'-?[\d.]+(?:[eE][+\-]?\d+)?'

def _f(pat, text, g=1):
    m = re.search(pat, text, re.IGNORECASE)
    return float(m.group(g)) if m else None


def _parse_scenario_tag(tag: str):
    """Parse 'wt=0.50 we=0.50 cut=0.5 pessimistic' → (0.5, 0.5, 0.5, 'pessimistic')."""
    wt = re.search(r'wt=([\d.]+)', tag)
    we = re.search(r'we=([\d.]+)', tag)
    ct = re.search(r'cut=([\d.]+)', tag)
    sc = re.search(r'\b(pessimistic|optimistic|midpoint)\b', tag, re.IGNORECASE)
    return (
        float(wt.group(1)) if wt else None,
        float(we.group(1)) if we else None,
        float(ct.group(1)) if ct else None,
        sc.group(1).lower() if sc else None,
    )


def parse_block(run_lbl: str, text: str) -> dict:
    if "OBJECTIVE VALUES" not in text:
        return None

    d = {"run_name": run_lbl, "obj_type": None}

    if run_lbl.startswith("assembly_line_"):
        d["source_file"] = f"{run_lbl}.dat"
        d["w_time"], d["w_eri"], d["cut"], d["scenario"] = _parse_run_name(run_lbl)

    # ── Solver status and objective
    # The model's own "SOLVER STATUS:" line (proven optimal within gap 1e-6, or not) takes
    # precedence over OPL's built-in "// solution (...)" line, which does not check the gap.
    m = re.search(r'SOLVER STATUS:\s*([^(\n]+?)\s*\(cplexStatus[^)]*\)\s*objective\s*=\s*(' + NUM + r')', text)
    if not m:
        m = re.search(r'solution \(([^)]+)\) with objective (' + NUM + r')', text)
    d["status"]     = m.group(1) if m else "—"
    d["solver_obj"] = float(m.group(2)) if m else None

    # ── scenario_tag (echoed by CPLEX DISPLAY_RESULTS — primary metadata source)
    m = re.search(r'scenario_tag\s*=\s*(.+)', text)
    raw_tag = m.group(1).strip().strip('"') if m else ""
    d["scenario_tag"] = raw_tag
    if d["w_time"] is None:
        d["w_time"], d["w_eri"], d["cut"], d["scenario"] = _parse_scenario_tag(raw_tag)

    # ── Fall back to body fields if scenario_tag absent (legacy / backward compat)
    if d["w_time"] is None:
        m2 = re.search(r'w_time\s*=\s*([\d.]+)\s+w_eri\s*=\s*([\d.]+)', text)
        if m2:
            d["w_time"], d["w_eri"] = float(m2.group(1)), float(m2.group(2))
        else:
            m3 = re.search(r'alpha\s*=\s*([\d.]+)\s+beta\s*=\s*([\d.]+)', text)
            if m3:
                d["w_time"], d["w_eri"] = float(m3.group(1)), float(m3.group(2))

    # ── Capacity caps
    d["CT_max"]  = _f(r'CT_max\s*=\s*([\d.]+)', text)
    d["ERI_max"] = _f(r'ERI_max\s*=\s*([\d.]+)', text)

    # ── Objective values section
    m = re.search(r'\[OBJ-1\][^\n]*=\s*([\d.]+)', text)
    d["OBJ1"]       = float(m.group(1)) if m else None
    d["cycleTime"]  = _f(r'cycleTime\s*=\s*([\d.]+)', text)
    # NOTE: must not match the "[OBJ-1] ...*maxEriLoad = X" formula echo above —
    # that "X" is the blended objective, not the real ergonomic load. Only the
    # standalone "maxEriLoad = X" line (not preceded by "*") holds the true value.
    d["maxEriLoad"] = _f(r'(?<!\*)\bmaxEriLoad\s*=\s*([\d.]+)', text)

    m = re.search(r'\[OBJ-2\][^\n]*=\s*([\d.]+)', text)
    d["OBJ2"] = float(m.group(1)) if m else None

    m = re.search(
        r'SI_time\s*=\s*sqrt\(SI2_time\)\s*=\s*([\d.]+)\s+SI2_time\s*=\s*([\d.]+)', text)
    d["SI_time"],  d["SI2_time"]  = (float(m.group(1)), float(m.group(2))) if m else (None, None)

    m = re.search(
        r'SI_eri\s*=\s*sqrt\(SI2_eri\)\s*=\s*([\d.]+)\s+SI2_eri\s*=\s*([\d.]+)', text)
    d["SI_eri"],   d["SI2_eri"]   = (float(m.group(1)), float(m.group(2))) if m else (None, None)

    # ── Operator assignments
    ops = []
    for om in re.finditer(
        r'Operator\s+(\d+):(.*?)(?=Operator\s+\d+:|PERFORMANCE METRICS|$)',
        text, re.DOTALL | re.IGNORECASE
    ):
        ot = om.group(2)
        m_tn = re.search(
            r'TN_m\s*=\s*([\d.]+)\s+idle\s*=\s*(' + NUM + r')\s+util%\s*=\s*([\d.]+)', ot)
        m_er = re.search(r'\bERI\b\s*=\s*([\d.]+)\s+idle\s*=\s*(' + NUM + r')', ot)
        ops.append({
            "op":       int(om.group(1)),
            "TN_m":     float(m_tn.group(1)) if m_tn else None,
            "idle_TN":  float(m_tn.group(2)) if m_tn else None,
            "util_pct": float(m_tn.group(3)) if m_tn else None,
            "ERI":      float(m_er.group(1)) if m_er else None,
            "ERI_idle": float(m_er.group(2)) if m_er else None,
        })
    d["operators"] = ops

    # ── Performance metrics block
    pm = re.search(r'PERFORMANCE METRICS(.*?)(?:={5,}|$)', text, re.DOTALL | re.IGNORECASE)
    if pm:
        pt = pm.group(1)
        d["line_eff"]     = _f(r'Line Efficiency\s*:\s*([\d.]+)', pt)
        d["total_TNm"]    = _f(r'Total TN_m\s*:\s*([\d.]+)', pt)
        d["perf_SI_time"] = _f(r'SI_time\s*:\s*([\d.]+)', pt)
        d["perf_SI_eri"]  = _f(r'SI_eri\s*:\s*([\d.]+)', pt)

    # ── Fill gaps
    if d["cycleTime"] is None and ops:
        vals = [o["TN_m"] for o in ops if o["TN_m"] is not None]
        d["cycleTime"] = max(vals) if vals else None
    if d["SI_time"]  is None: d["SI_time"]  = d.get("perf_SI_time")
    if d["SI_eri"]   is None: d["SI_eri"]   = d.get("perf_SI_eri")
    if d["SI2_time"] is None and d["SI_time"] is not None: d["SI2_time"] = d["SI_time"] ** 2
    if d["SI2_eri"]  is None and d["SI_eri"]  is not None: d["SI2_eri"]  = d["SI_eri"]  ** 2
    if d["maxEriLoad"] is None and ops:
        eris = [o["ERI"] for o in ops if o["ERI"] is not None]
        d["maxEriLoad"] = max(eris) if eris else None
    wt = d.get("w_time") or 0
    we = d.get("w_eri")  or 0
    if d["OBJ1"] is None and d["cycleTime"] is not None:
        d["OBJ1"] = wt * d["cycleTime"] + we * (d["maxEriLoad"] or 0)
    if d["OBJ2"] is None and d["SI2_time"] is not None:
        d["OBJ2"] = wt * d["SI2_time"] + we * (d["SI2_eri"] or 0)

    if d["obj_type"] is None:
        d["obj_type"] = "OBJ-1" if d.get("OBJ1") is not None else "OBJ-2"

    return d


# ─── 4. Build flat results DataFrame ─────────────────────────────────────────

def build_results_df(solutions: list) -> pd.DataFrame:
    rows = []
    for s in solutions:
        obj_val = s.get("OBJ1") if s["obj_type"] == "OBJ-1" else s.get("OBJ2")
        rows.append({
            "run_name":     s.get("run_name"),
            "source_file":  s.get("source_file"),
            "obj_type":    s["obj_type"],
            "w_time":      s.get("w_time"),
            "w_eri":       s.get("w_eri"),
            "cut":         s.get("cut"),
            "scenario":    s.get("scenario"),
            "scenario_tag":s.get("scenario_tag"),
            "status":      s.get("status"),
            "solver_obj":  s.get("solver_obj"),
            "obj_value":   obj_val,
            "cycleTime":   s.get("cycleTime"),
            "maxEriLoad":  s.get("maxEriLoad"),
            "SI_time":     s.get("SI_time"),
            "SI_eri":      s.get("SI_eri"),
            "line_eff":    s.get("line_eff"),
            "OBJ1":        s.get("OBJ1"),
            "OBJ2":        s.get("OBJ2"),
        })
    return pd.DataFrame(rows)


# ─── 5. Fuzzy membership function chart ──────────────────────────────────────

def plot_membership_functions(df, path):
    """
    For each (w_time, w_eri) pair, reconstruct the triangular membership function
    of the fuzzy optimal objective from the three α-cut levels:
        α=0 optimistic  → left foot  (μ=0)
        α=0 pessimistic → right foot (μ=0)
        α=0.5 opt/pess  → intermediate interval (μ=0.5)
        α=1 midpoint    → apex        (μ=1)
    One subplot per (weight pair × objective type).
    """
    wps = sorted(df[["w_time", "w_eri"]].dropna().drop_duplicates()
                 .itertuples(index=False), key=lambda r: -r.w_time)
    obj_types = sorted(df["obj_type"].dropna().unique())
    if not wps or not obj_types:
        return

    n_rows, n_cols = len(wps), len(obj_types)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 3.5 * n_rows),
                             sharex=False, sharey=True)
    if n_rows == 1: axes = np.array([axes])
    if n_cols == 1: axes = axes[:, np.newaxis]

    fig.suptitle("Fuzzy Optimal Objective — Membership Functions\n"
                 "(reconstructed from α-cut parametric analysis)",
                 fontsize=13, fontweight="bold")

    mu_map = {0.0: 0.0, 0.5: 0.5, 1.0: 1.0}

    for ri, wp in enumerate(wps):
        wt, we = wp.w_time, wp.w_eri
        color  = WT_PALETTE.get(round(wt, 2), "grey")

        for ci, obj in enumerate(obj_types):
            ax  = axes[ri][ci]
            sub = df[(df["w_time"] == wt) & (df["w_eri"] == we) & (df["obj_type"] == obj)]

            # Collect (lower_bound, upper_bound) per cut level
            cuts = sorted(df["cut"].dropna().unique())
            pts  = {}
            for cut in cuts:
                cs    = sub[sub["cut"] == cut]
                pess  = cs[cs["scenario"].isin(["pessimistic", "midpoint"])]["obj_value"]
                opt   = cs[cs["scenario"].isin(["optimistic",  "midpoint"])]["obj_value"]
                hi = float(pess.max()) if not pess.empty else None
                lo = float(opt.min())  if not opt.empty  else hi
                if hi is not None:
                    pts[cut] = (lo, hi)

            if not pts:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="grey")
                ax.set_title(f"wt={wt:.2f} we={we:.2f} | {obj}", fontsize=9, fontweight="bold")
                continue

            sorted_cuts = sorted(pts)
            lo_vals = [pts[c][0] for c in sorted_cuts]
            hi_vals = [pts[c][1] for c in sorted_cuts]
            mus     = [mu_map[c]  for c in sorted_cuts]

            ax.fill_betweenx(mus, lo_vals, hi_vals, alpha=0.15, color=color)
            ax.plot(lo_vals, mus, color=color, lw=2, ls="--", label="optimistic bound")
            ax.plot(hi_vals, mus, color=color, lw=2, ls="-",  label="pessimistic bound")

            for cut, (lo, hi) in pts.items():
                mu = mu_map[cut]
                mk = {0.0: "o", 0.5: "s", 1.0: "^"}.get(cut, "o")
                ax.hlines(mu, lo, hi, colors=color, lw=1.0, alpha=0.5)
                for x, ha in [(lo, "right"), (hi, "left")]:
                    ax.scatter(x, mu, marker=mk, s=60, color=color,
                               edgecolors="black", linewidth=0.7, zorder=5)
                    ax.annotate(f"{x:.3f}", (x, mu),
                                xytext=(-5 if ha == "right" else 5, 4),
                                textcoords="offset points", fontsize=7, ha=ha)

            ax.set_ylim(-0.1, 1.2)
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(["0  (α=0)", "0.5  (α=0.5)", "1  (α=1)"], fontsize=7)
            ax.set_xlabel("Objective value", fontsize=8)
            ax.set_ylabel("Membership μ", fontsize=8)
            ax.set_title(f"wt={wt:.2f}  we={we:.2f}  |  {obj}", fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─── 6. Weight sensitivity line plots ─────────────────────────────────────────

def plot_weight_sensitivity(df, path):
    """Objective (and CT/maxERI) vs w_time for each cut×scenario combination."""
    obj_types  = sorted(df["obj_type"].dropna().unique())
    cuts       = sorted(df["cut"].dropna().unique())
    scenarios  = [s for s in ["optimistic", "midpoint", "pessimistic"]
                  if s in df["scenario"].fillna("").unique()]

    metrics = [
        ("obj_value",  "Objective value"),
        ("cycleTime",  "Cycle Time (CT)"),
        ("maxEriLoad", "Max FERI"),
    ]

    fig, axes = plt.subplots(len(obj_types), len(metrics),
                             figsize=(6 * len(metrics), 4.2 * len(obj_types)),
                             squeeze=False)
    fig.suptitle("Sensitivity Analysis — Objective vs Objective Weight (w_time)",
                 fontsize=13, fontweight="bold")

    for ci, obj in enumerate(obj_types):
        sub = df[df["obj_type"] == obj]
        for ri, (metric, ylabel) in enumerate(metrics):
            ax = axes[ci][ri]
            for cut in cuts:
                for sc in scenarios:
                    grp = sub[(sub["cut"] == cut) & (sub["scenario"] == sc)]
                    if grp.empty:
                        continue
                    grp = grp.sort_values("w_time")
                    ax.plot(grp["w_time"], grp[metric],
                            ls=CUT_LS.get(cut, "-"),
                            marker=SC_MARKER.get(sc, "o"),
                            lw=1.8, ms=6,
                            label=f"cut={cut} {sc}")
            ax.set_xlabel("w_time  (0 = ergo-only → 1 = time-only)", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f"{obj}  —  {ylabel}", fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(fontsize=6.5, ncol=2)
            ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─── 7. CT vs max-ERI Pareto scatter ─────────────────────────────────────────

def plot_pareto_scatter(df, path):
    """Cycle time vs max ERI for all runs, coloured by w_time."""
    obj_types = sorted(df["obj_type"].dropna().unique())
    fig, axes = plt.subplots(1, len(obj_types), figsize=(7 * len(obj_types), 6))
    if len(obj_types) == 1: axes = [axes]
    fig.suptitle("Cycle Time vs Max FERI  (Pareto view — all runs)",
                 fontsize=12, fontweight="bold")

    for ax, obj in zip(axes, obj_types):
        sub = df[(df["obj_type"] == obj)].dropna(subset=["cycleTime", "maxEriLoad"])
        # Runs that reach exactly the same (CT, max ERI) point are drawn as nested markers of
        # decreasing size (largest = first run of the group, smallest on top), so that every
        # run stays visible instead of being hidden behind the last one drawn.
        rows = sub.assign(_k=list(zip(sub["cycleTime"].round(4), sub["maxEriLoad"].round(4))))
        for _, grp in rows.groupby("_k", sort=False):
            grp = grp.sort_values("w_time", ascending=True)
            n = len(grp)
            for j, (_, row) in enumerate(grp.iterrows()):
                wt = row.get("w_time")
                sc = row.get("scenario", "midpoint")
                color  = WT_PALETTE.get(round(wt, 2) if wt is not None else -1, "grey")
                marker = SC_MARKER.get(sc, "o")
                ax.scatter(row["cycleTime"], row["maxEriLoad"],
                           c=color, marker=marker, s=90 + 200 * (n - 1 - j), zorder=4 + j,
                           edgecolors="black", linewidth=0.6)
        ax.set_xlabel("Cycle Time (CT)", fontsize=9)
        ax.set_ylabel("Max FERI",    fontsize=9)
        ax.set_title(obj, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    # Legends
    wt_legend = [Patch(facecolor=c, label=f"w_time={wt:.2f}")
                 for wt, c in sorted(WT_PALETTE.items(), reverse=True)]
    sc_legend  = [Line2D([0],[0], marker=mk, color="grey", lw=0, ms=8, label=sc)
                  for sc, mk in SC_MARKER.items()]
    fig.legend(handles=wt_legend + sc_legend, loc="lower center",
               ncol=len(wt_legend) + len(sc_legend), fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.text(0.5, 0.925, "Runs with identical (CT, max FERI) are drawn as nested markers",
             ha="center", fontsize=8, style="italic", color="dimgrey")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─── 8. Objective heatmap (weight pair × run scenario) ───────────────────────

def plot_objective_heatmap(df, metric, title, path, cmap="YlOrRd"):
    """Rows = weight pairs, columns = cut×scenario, one file per objective type."""
    df2 = df.copy()
    df2["weights"] = df2.apply(
        lambda r: f"wt={r['w_time']:.2f} we={r['w_eri']:.2f}"
        if pd.notna(r["w_time"]) else "—", axis=1)
    df2["run"] = df2.apply(
        lambda r: f"cut={r['cut']} {r['scenario']}"
        if pd.notna(r["cut"]) else "—", axis=1)

    for obj in df2["obj_type"].dropna().unique():
        sub   = df2[df2["obj_type"] == obj]
        pivot = sub.pivot_table(index="weights", columns="run",
                                values=metric, aggfunc="first")
        if pivot.empty:
            continue
        data = pivot.values.astype(float)
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.8),
                                        max(3, len(pivot) * 0.9)))
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                v = data[r, c]
                if np.isnan(v):
                    continue
                br = (v - vmin) / (vmax - vmin + 1e-9)
                ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                        fontsize=7.5, color="white" if br > 0.6 else "black",
                        fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.set_title(f"{title}  [{obj}]", fontsize=11, fontweight="bold", pad=10)
        fig.tight_layout()
        fname = path.parent / f"{path.stem}_{obj.replace('-', '')}{path.suffix}"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname.name}")


# ─── 9. Operator-level heatmaps (selected subset) ────────────────────────────

def build_operator_df(solutions, metric):
    data = {}
    for s in solutions:
        col = s.get("run_name") or s.get("scenario_tag") or f"{s['obj_type']} wt={s.get('w_time')} cut={s.get('cut')} {s.get('scenario')}"
        col = f"{s['obj_type']} | {col}"
        data[col] = {o["op"]: o.get(metric) for o in s["operators"]}
    df = pd.DataFrame(data)
    df.index.name = "Operator"
    return df


def plot_heatmap(df, title, path, cmap="YlOrRd", fmt=".2f"):
    if df.empty:
        return
    data = df.values.astype(float)
    fig, ax = plt.subplots(figsize=(max(9, len(df.columns) * 1.8),
                                    max(3, len(df) * 0.7)))
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([f"Op {i}" for i in df.index], fontsize=9)
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            br = (v - vmin) / (vmax - vmin + 1e-9)
            ax.text(c, r, f"{v:{fmt}}", ha="center", va="center",
                    fontsize=7.5, color="white" if br > 0.6 else "black",
                    fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─── 10. Excel report ─────────────────────────────────────────────────────────

def save_excel(df, solutions, path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # All results flat
        export_cols = [c for c in df.columns if c != "operators"]
        df[export_cols].to_excel(writer, sheet_name="Results", index=False)

        # Pivot per objective
        df2 = df.copy()
        df2["weights"] = df2.apply(
            lambda r: f"wt={r['w_time']:.2f} we={r['w_eri']:.2f}"
            if pd.notna(r["w_time"]) else "—", axis=1)
        df2["run"] = df2.apply(
            lambda r: f"cut={r['cut']} {r['scenario']}"
            if pd.notna(r["cut"]) else "—", axis=1)
        for obj in df2["obj_type"].dropna().unique():
            sub   = df2[df2["obj_type"] == obj]
            pivot = sub.pivot_table(index="weights", columns="run",
                                    values="obj_value", aggfunc="first")
            pivot.to_excel(writer, sheet_name=f"Pivot_{obj.replace('-','')}")

        # Operator data
        for metric, sheet in [("TN_m", "Op_TN_m"), ("ERI", "Op_ERI"),
                               ("util_pct", "Op_Utilisation")]:
            op_df = build_operator_df(solutions, metric)
            if not op_df.empty:
                op_df.to_excel(writer, sheet_name=sheet)

    print(f"  Saved: {path.name}")


# ─── 11. Console summary ─────────────────────────────────────────────────────

def print_summary(df):
    cols = ["run_name", "obj_type", "w_time", "w_eri", "cut", "scenario",
            "status", "obj_value", "cycleTime", "maxEriLoad",
            "SI_time", "SI_eri", "line_eff"]
    cols = [c for c in cols if c in df.columns]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n" + "=" * 120)
    print("  RESULTS SUMMARY")
    print("=" * 120)
    print(df[cols].to_string(index=False))
    print("=" * 120 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not RTF_FILE.exists():
        sys.exit(f"File not found: {RTF_FILE}")

    print(f"Reading {RTF_FILE.name} …")
    raw  = RTF_FILE.read_text(encoding="latin-1", errors="replace")
    text = strip_rtf(raw)

    print("Parsing solution blocks …")
    solutions = []
    for lbl, blk in split_blocks(text):
        sol = parse_block(lbl, blk)
        if sol is None:
            continue
        solutions.append(sol)
        display_name = sol.get('run_name') or lbl
        print(f"  [{display_name}]  tag={sol.get('scenario_tag')!r:45s}  "
              f"CT={sol.get('cycleTime')}  obj={sol.get('solver_obj')}")

    if not solutions:
        sys.exit("No solution blocks found.\n"
                 "Expected filename-led blocks like assembly_line_wt000_we100_cut00_optimistic.")

    df = build_results_df(solutions)
    has_metadata = df["cut"].notna().any()

    print_summary(df)
    print(f"\nGenerating charts → {OUT_DIR}/")

    if has_metadata:
        plot_membership_functions(df, OUT_DIR / "fig1_membership_functions.png")
        plot_weight_sensitivity(df,   OUT_DIR / "fig2_weight_sensitivity.png")
        plot_pareto_scatter(df,       OUT_DIR / "fig3_pareto_ct_vs_eri.png")
        plot_objective_heatmap(df, "obj_value", "Objective Value",
                               OUT_DIR / "fig4_heatmap_objective.png", cmap="YlOrRd")
        plot_objective_heatmap(df, "cycleTime", "Cycle Time",
                               OUT_DIR / "fig4_heatmap_cycletime.png", cmap="Blues")
        plot_objective_heatmap(df, "maxEriLoad", "Max ERI Load",
                               OUT_DIR / "fig4_heatmap_eri.png",       cmap="Oranges")

        # Operator-level heatmaps for balanced scenario only (avoid 50-column plots)
        balanced = [s for s in solutions
                    if abs((s.get("w_time") or 0) - 0.5) < 1e-6
                    and abs((s.get("w_eri")  or 0) - 0.5) < 1e-6]
        if balanced:
            df_tnm  = build_operator_df(balanced, "TN_m")
            df_eri  = build_operator_df(balanced, "ERI")
            df_util = build_operator_df(balanced, "util_pct")
            plot_heatmap(df_tnm,  "Task-Time Load (TN_m) — wt=0.5 we=0.5",
                         OUT_DIR / "fig5a_op_TNm_balanced.png",  cmap="Blues")
            plot_heatmap(df_eri,  "Ergonomic Load (ERI)  — wt=0.5 we=0.5",
                         OUT_DIR / "fig5b_op_ERI_balanced.png",  cmap="Oranges")
            plot_heatmap(df_util, "Utilisation (%)       — wt=0.5 we=0.5",
                         OUT_DIR / "fig5c_op_util_balanced.png", cmap="Greens")
    else:
        # Legacy mode: only two runs, no cut/scenario structure
        print("  (no filename headers found — running in legacy 2-block mode)")
        plot_pareto_scatter(df, OUT_DIR / "fig3_pareto_ct_vs_eri.png")
        df_tnm  = build_operator_df(solutions, "TN_m")
        df_eri  = build_operator_df(solutions, "ERI")
        df_util = build_operator_df(solutions, "util_pct")
        plot_heatmap(df_tnm,  "Task-Time Load (TN_m)", OUT_DIR / "fig5a_op_TNm.png",  cmap="Blues")
        plot_heatmap(df_eri,  "Ergonomic Load (ERI)",  OUT_DIR / "fig5b_op_ERI.png",  cmap="Oranges")
        plot_heatmap(df_util, "Utilisation (%)",        OUT_DIR / "fig5c_op_util.png", cmap="Greens")

    save_excel(df, solutions, OUT_DIR / "cplex_results.xlsx")
    print(f"\nDone. All outputs saved to  {OUT_DIR}/\n")


if __name__ == "__main__":
    main()
