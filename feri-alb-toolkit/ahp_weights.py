"""
ahp_weights.py
==============
AHP (Analytic Hierarchy Process) for FERI ergonomic criteria weights.

Criteria (3×3 pairwise problem):
    Energy  — physical energy cost / metabolic load
    REBA    — postural risk (REBA score)
    Borg    — perceived exertion (Borg CR-10)

Pipeline:
    1. Read expert pairwise comparisons from experts_answer.xlsx
    2. Map verbal descriptors → Saaty 1–9 scale
    3. Build a 3×3 comparison matrix per expert
    4. Compute priority vector (eigenvector method) + Consistency Ratio
    5. Exclude experts whose CR > 0.10
    6. Aggregate valid matrices via element-wise geometric mean
    7. Return final weight dict for use in fuzzy_eri.py

Run standalone:   python ahp_weights.py
Import:           from ahp_weights import get_weights
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────

CRITERIA       = ["energy", "reba", "borg"]
CRITERIA_LABEL = {"energy": "Energy", "reba": "REBA", "borg": "Borg"}
CR_THRESHOLD   = 0.10

# Saaty (1980) Random Consistency Index for n = 1..10
RI_VALUES = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
             6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# Default fallback (equal weights)
DEFAULT_WEIGHTS = {"energy": 1/3, "reba": 1/3, "borg": 1/3}

AHP_WEIGHTS_PATH = Path(__file__).parent / "experts_answer.xlsx"

# ─── Verbal → Saaty scale ─────────────────────────────────────────────────────
# Longer entries must appear before shorter ones so longest-match logic works
# when both a compound phrase and a sub-phrase are present.

VERBAL_SCALE = {
    # Intermediate (compound) values — 2, 4, 6, 8
    "between equal and slightly":        2,
    "between slightly and moderately":   4,
    "between moderately and strongly":   6,
    "between strongly and extremely":    8,
    "between strongly and absolutely":   8,
    # Standard Saaty levels (longer strings must precede shorter sub-strings)
    "absolutely":    9,
    "extremely":     9,
    "very strongly": 7,
    "strongly":      5,
    "moderately":    5,
    "slightly":      3,
    "equally":       1,
    "equal":         1,
    # Plain directional preference (3-option scale: Favor A / Equal / Favor B)
    # No adverb → treated as the weakest non-equal preference (Saaty 2)
    # so the answer encodes direction, not a strong intensity claim.
    "favor":         2,
}

# Criterion name aliases searched (lower-case) inside the answer text.
# Order within each list: most specific first (longer strings → fewer false matches).
CRITERION_ALIASES = {
    "energy": ["energy expenditure", "metabolic load", "energy", "metabolic"],
    "reba":   ["postural risk", "reba score", "reba"],
    "borg":   ["perceived exertion", "borg cr-10", "borg"],
}

# ─── Column auto-detection ────────────────────────────────────────────────────

# Each entry: (crit_A, crit_B, must_contain_all, must_exclude)
# Used to identify the correct Excel column for each pairwise comparison.
COMPARISON_DEFS = [
    ("energy", "reba",
     ["metabolic", "reba"],
     ["borg", "perceived"]),
    ("energy", "borg",
     ["metabolic", "borg"],
     []),
    ("reba", "borg",
     ["reba", "borg"],
     ["metabolic", "energy"]),
]


def _find_column(columns, must_contain, must_exclude):
    """Return the first column name matching all must_contain and none of must_exclude."""
    for col in columns:
        if not isinstance(col, str):
            continue
        cl = col.lower()
        if all(kw in cl for kw in must_contain):
            if not any(kw in cl for kw in must_exclude):
                return col
    return None


def _detect_comparison_columns(df_columns):
    """
    Return dict {(crit_a, crit_b): column_name} for all 3 comparisons.
    Raises ValueError if any comparison column cannot be found.
    """
    col_map = {}
    for crit_a, crit_b, must, excl in COMPARISON_DEFS:
        col = _find_column(df_columns, must, excl)
        if col is None:
            raise ValueError(
                f"Cannot find column for '{crit_a}' vs '{crit_b}'. "
                f"Must contain: {must}, must exclude: {excl}. "
                f"Available columns: {list(df_columns)}"
            )
        col_map[(crit_a, crit_b)] = col
    return col_map


# ─── Verbal answer parser ─────────────────────────────────────────────────────

def _parse_answer(text):
    """
    Parse one verbal answer.

    Returns
    -------
    (scale: float, favored_crit: str | None)
        favored_crit is None when the answer means "equally important".
    Returns None if the text is missing or unrecognisable.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    t = text.strip().lower()

    # "Equally important" case
    if t.startswith("equal"):
        return 1.0, None

    # Determine strength: longest matching key wins
    scale_value = None
    matched_len = 0
    for keyword, value in VERBAL_SCALE.items():
        if keyword in t and len(keyword) > matched_len:
            scale_value = float(value)
            matched_len = len(keyword)

    if scale_value is None:
        return None

    # Determine favored criterion: longest alias match wins
    favored = None
    matched_len = 0
    for crit, aliases in CRITERION_ALIASES.items():
        for alias in aliases:
            if alias in t and len(alias) > matched_len:
                favored = crit
                matched_len = len(alias)

    if favored is None:
        return None

    return scale_value, favored


def _logical_consistency(row, col_map):
    """
    Check whether the three pairwise answers can coexist without contradiction.

    Equal answers merge criteria into the same equivalence class. Any directional
    preference inside the same class, or any cycle between classes, is treated as
    a logical contradiction.
    """
    parsed = {}
    for crit_a, crit_b, *_ in COMPARISON_DEFS:
        col = col_map.get((crit_a, crit_b))
        if col is None or col not in row.index:
            return False, "missing comparison column"

        result = _parse_answer(row[col])
        if result is None:
            return False, "incomplete or unparseable answers"

        _, favored = result
        parsed[(crit_a, crit_b)] = favored

    parent = {crit: crit for crit in CRITERIA}

    def find(crit):
        while parent[crit] != crit:
            parent[crit] = parent[parent[crit]]
            crit = parent[crit]
        return crit

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    # First merge all equalities.
    for (crit_a, crit_b), favored in parsed.items():
        if favored is None:
            union(crit_a, crit_b)

    # Then add directional edges between the resulting equivalence classes.
    graph = {}
    for crit_a, crit_b, *_ in COMPARISON_DEFS:
        favored = parsed[(crit_a, crit_b)]
        if favored is None:
            continue

        source = crit_a if favored == crit_a else crit_b
        target = crit_b if favored == crit_a else crit_a
        source_root = find(source)
        target_root = find(target)

        if source_root == target_root:
            return False, (
                f"logical contradiction: {CRITERIA_LABEL[crit_a]} and "
                f"{CRITERIA_LABEL[crit_b]} are treated as equal and ordered at the same time"
            )

        graph.setdefault(source_root, set()).add(target_root)
        graph.setdefault(target_root, set())

    # Detect cycles between classes.
    indegree = {node: 0 for node in graph}
    for source, targets in graph.items():
        for target in targets:
            indegree[target] = indegree.get(target, 0) + 1

    queue = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for target in graph.get(node, ()): 
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if visited != len(graph):
        return False, "logical contradiction: pairwise answers contain a preference cycle"

    return True, None


# ─── Matrix construction ──────────────────────────────────────────────────────

def _build_matrix(row, col_map):
    """
    Build a 3×3 pairwise comparison matrix for one expert row.

    Returns np.ndarray (3×3) or None if any comparison is missing/unparseable.
    """
    n = len(CRITERIA)
    A = np.ones((n, n))

    for crit_a, crit_b, *_ in COMPARISON_DEFS:
        ia = CRITERIA.index(crit_a)
        ib = CRITERIA.index(crit_b)

        col = col_map.get((crit_a, crit_b))
        if col is None or col not in row.index:
            return None

        result = _parse_answer(row[col])
        if result is None:
            return None

        scale, favored = result

        if favored is None:           # equally important
            A[ia][ib] = 1.0
            A[ib][ia] = 1.0
        elif favored == crit_a:
            A[ia][ib] = scale
            A[ib][ia] = 1.0 / scale
        else:                         # favored == crit_b
            A[ia][ib] = 1.0 / scale
            A[ib][ia] = scale

    return A


# ─── AHP mathematics ──────────────────────────────────────────────────────────

def _priority_vector(A):
    """Principal right eigenvector (normalised) of A."""
    vals, vecs = np.linalg.eig(A)
    idx = int(np.argmax(vals.real))
    w = np.abs(vecs[:, idx].real)
    return w / w.sum()


def _consistency(A, w):
    """Returns (CI, CR) for matrix A and priority vector w."""
    n = A.shape[0]
    lam_max = float(np.mean((A @ w) / w))
    CI = (lam_max - n) / (n - 1)
    CR = CI / RI_VALUES.get(n, 1.49)
    return CI, CR


def _geometric_mean_aggregate(matrices):
    """Element-wise geometric mean of a list of matrices (Saaty aggregation)."""
    stacked = np.stack(matrices, axis=0)
    return np.exp(np.mean(np.log(stacked), axis=0))


# ─── Public API ───────────────────────────────────────────────────────────────

def get_weights(excel_path=None, verbose=True):
    """
    Run the full AHP pipeline.

    Parameters
    ----------
    excel_path : str or Path, optional
        Path to experts_answer.xlsx. Defaults to AHP_WEIGHTS_PATH.
    verbose : bool
        Print the full AHP report to stdout.

    Returns
    -------
    dict  {"energy": float, "reba": float, "borg": float}
        Normalised priority weights summing to 1.
        Falls back to equal weights (1/3 each) if no valid experts are found.
    """
    path = Path(excel_path) if excel_path else AHP_WEIGHTS_PATH

    if not path.exists():
        if verbose:
            print(f"  [AHP] File not found: {path}. Using equal weights.")
        return dict(DEFAULT_WEIGHTS)

    df = pd.read_excel(path, header=0)

    try:
        col_map = _detect_comparison_columns(df.columns)
    except ValueError as e:
        if verbose:
            print(f"  [AHP] Column detection failed: {e}\n  Using equal weights.")
        return dict(DEFAULT_WEIGHTS)

    if verbose:
        _sep = "=" * 72
        print(f"\n{_sep}")
        print("  AHP — Analytic Hierarchy Process")
        print("  Criteria: Energy  |  REBA  |  Borg")
        print(_sep)
        print(f"\n  Expert responses in file : {len(df)}")
        print(f"  Comparison columns detected:")
        for (ca, cb), col in col_map.items():
            print(f"    {CRITERIA_LABEL[ca]:6s} vs {CRITERIA_LABEL[cb]:6s} → "
                  f"'{col[:55].strip()}...'")

    valid_matrices = []

    for idx, row in df.iterrows():
        name = str(row.get("Name", "")).strip()
        if not name or name.lower() == "nan":
            name = f"Expert {idx + 1}"
        institution = str(row.get("Institution / Company", "")).strip()
        if institution.lower() == "nan":
            institution = ""
        label = f"{name}" + (f" ({institution})" if institution else "")

        logical_ok, logical_reason = _logical_consistency(row, col_map)
        if not logical_ok:
            if verbose:
                print(f"\n  [{label}]  → Skipped ({logical_reason})")
            continue

        A = _build_matrix(row, col_map)
        if A is None:
            if verbose:
                print(f"\n  [{label}]  → Skipped (incomplete or unparseable answers)")
            continue

        w  = _priority_vector(A)
        CI, CR = _consistency(A, w)

        if verbose:
            print(f"\n  [{label}]")
            print(f"    Pairwise comparison matrix:")
            hdr = "           " + "  ".join(f"{CRITERIA_LABEL[c]:>8}" for c in CRITERIA)
            print(f"    {hdr}")
            for i, c in enumerate(CRITERIA):
                row_s = "  ".join(f"{A[i][j]:>8.4f}" for j in range(len(CRITERIA)))
                print(f"    {CRITERIA_LABEL[c]:>8}   {row_s}")
            print(f"    Priority weights:")
            for c, wi in zip(CRITERIA, w):
                bar = "█" * int(wi * 36)
                print(f"      {CRITERIA_LABEL[c]:>8}: {wi:.4f}  ({wi*100:5.1f}%)  {bar}")
            cr_tag = "✓ consistent" if CR <= CR_THRESHOLD else f"✗ inconsistent (> {CR_THRESHOLD})"
            print(f"    CI = {CI:.4f}  |  CR = {CR:.4f}  [{cr_tag}]")

        if CR <= CR_THRESHOLD:
            valid_matrices.append(A)
        else:
            if verbose:
                print(f"    → Excluded from aggregation.")

    # ── Aggregation ────────────────────────────────────────────────────────────
    if not valid_matrices:
        if verbose:
            print("\n  WARNING: No consistent expert matrices. Using equal weights.")
        return dict(DEFAULT_WEIGHTS)

    if len(valid_matrices) == 1:
        agg_A = valid_matrices[0]
    else:
        agg_A = _geometric_mean_aggregate(valid_matrices)

    agg_w        = _priority_vector(agg_A)
    agg_CI, agg_CR = _consistency(agg_A, agg_w)
    n_valid      = len(valid_matrices)
    n_total      = len(df)

    if verbose:
        _dsep = "─" * 72
        print(f"\n  {_dsep}")
        method = ("direct (1 expert)" if n_valid == 1
                  else f"geometric mean of {n_valid} experts")
        print(f"  AGGREGATED RESULT  [{method}]  ({n_valid}/{n_total} valid)")
        print(f"  {_dsep}")
        print(f"  Aggregated pairwise matrix:")
        hdr = "           " + "  ".join(f"{CRITERIA_LABEL[c]:>8}" for c in CRITERIA)
        print(f"  {hdr}")
        for i, c in enumerate(CRITERIA):
            row_s = "  ".join(f"{agg_A[i][j]:>8.4f}" for j in range(len(CRITERIA)))
            print(f"  {CRITERIA_LABEL[c]:>8}   {row_s}")
        print(f"\n  Final priority weights:")
        for c, wi in zip(CRITERIA, agg_w):
            bar = "█" * int(wi * 36)
            print(f"    {CRITERIA_LABEL[c]:>8}: {wi:.4f}  ({wi*100:5.1f}%)  {bar}")
        cr_tag = "✓ Acceptable" if agg_CR <= CR_THRESHOLD else "⚠ Above 0.10 — review judgements"
        print(f"\n  CI = {agg_CI:.4f}  |  CR = {agg_CR:.4f}  [{cr_tag}]")
        print(f"\n  → Paste into fuzzy_eri.py (for reference):")
        print(f"    WEIGHTS = {{")
        for c, wi in zip(CRITERIA, agg_w):
            print(f'        "{c}": {wi:.4f},')
        print(f"    }}")
        print("=" * 72)

    return {c: float(wi) for c, wi in zip(CRITERIA, agg_w)}


# ─── Standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    weights = get_weights(verbose=True)
    print(f"\nWeights returned: {weights}")
    total = sum(weights.values())
    print(f"Sum of weights  : {total:.6f}  {'✓' if abs(total - 1.0) < 1e-6 else '✗'}")
