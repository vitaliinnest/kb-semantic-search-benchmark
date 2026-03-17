"""Multi-criteria model selection: Pareto front + weighted scoring.

Criteria that are maximised (higher is better):
  ndcg_at_k, mrr_at_k, recall_at_k, precision_at_k

Criteria that are minimised (lower is better):
  avg_latency_ms

Workflow:
  1. Extract raw values for each model.
  2. Min-max normalise every criterion into [0, 1] so that 1 is always
     "best" (normalisation inverts min-criteria automatically).
  3. Compute a weighted composite score (weighted mean of normalised values).
  4. Identify the Pareto-optimal set on the **raw** values.
  5. Return ranked list + Pareto set + the single best model by composite score.
"""
from __future__ import annotations

CRITERIA: list[dict] = [
    {"key": "ndcg_at_k",      "label": "nDCG@k",     "maximize": True,  "default_weight": 0.35},
    {"key": "mrr_at_k",       "label": "MRR@k",      "maximize": True,  "default_weight": 0.20},
    {"key": "recall_at_k",    "label": "Recall@k",   "maximize": True,  "default_weight": 0.25},
    {"key": "precision_at_k", "label": "P@k",        "maximize": True,  "default_weight": 0.05},
    {"key": "avg_latency_ms", "label": "Latency ms", "maximize": False, "default_weight": 0.15},
]

DEFAULT_WEIGHTS: dict[str, float] = {c["key"]: c["default_weight"] for c in CRITERIA}


# ── helpers ───────────────────────────────────────────────────────────────────

def _raw_row(model: dict) -> dict[str, float]:
    return {c["key"]: float(model.get(c["key"]) or 0.0) for c in CRITERIA}


def _normalize(raw_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """Min-max normalise; for minimise-criteria invert so 1 = best."""
    keys = [c["key"] for c in CRITERIA]
    maximize = {c["key"]: c["maximize"] for c in CRITERIA}
    mins = {k: min(r[k] for r in raw_rows) for k in keys}
    maxs = {k: max(r[k] for r in raw_rows) for k in keys}

    result = []
    for row in raw_rows:
        norm: dict[str, float] = {}
        for k in keys:
            lo, hi = mins[k], maxs[k]
            span = hi - lo
            if span < 1e-12:
                norm[k] = 1.0
            elif maximize[k]:
                norm[k] = (row[k] - lo) / span
            else:
                norm[k] = (hi - row[k]) / span
        result.append(norm)
    return result


def _composite_score(norm_row: dict[str, float], weights: dict[str, float]) -> float:
    total_w = sum(weights.get(k, 0.0) for k in norm_row)
    if total_w < 1e-12:
        return 0.0
    return sum(norm_row[k] * weights.get(k, 0.0) for k in norm_row) / total_w


def _dominates(b_raw: dict[str, float], a_raw: dict[str, float]) -> bool:
    """Return True iff b dominates a (b ≥ a on all criteria, b > a on at least one).

    For maximise-criteria: b dominates if b[k] >= a[k].
    For minimise-criteria: b dominates if b[k] <= a[k].
    """
    maxim = {c["key"]: c["maximize"] for c in CRITERIA}
    strictly_better = False
    for k, is_max in maxim.items():
        va, vb = a_raw[k], b_raw[k]
        if is_max:
            if vb < va - 1e-9:   # b is worse
                return False
            if vb > va + 1e-9:
                strictly_better = True
        else:
            if vb > va + 1e-9:   # b is worse (higher latency)
                return False
            if vb < va - 1e-9:
                strictly_better = True
    return strictly_better


def _pareto_names(raw_rows: list[dict[str, float]], names: list[str]) -> list[str]:
    n = len(raw_rows)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i != j and _dominates(raw_rows[j], raw_rows[i]):
                dominated[i] = True
                break
    return [names[i] for i in range(n) if not dominated[i]]


# ── public API ────────────────────────────────────────────────────────────────

def run_selection(models: list[dict], weights: dict[str, float] | None = None) -> dict:
    """Run multi-criteria selection on a list of model benchmark dicts.

    Args:
        models:  list of model-level dicts from a benchmark JSON file.
        weights: optional override of criterion weights (0..1 each).

    Returns a dict:
        {
            "criteria":   list[dict],         # CRITERIA metadata
            "weights":    dict[str, float],   # effective weights used
            "ranked":     list[dict],         # models sorted by composite score
            "pareto_set": list[str],          # model names in Pareto-optimal set
            "best_model": str | None,         # model with highest composite score
        }

    Each item in "ranked":
        {
            "model_name": str,
            "raw":        dict[str, float],   # original metric values
            "normalized": dict[str, float],   # normalised values (0..1, 1=best)
            "score":      float,              # weighted composite score
            "pareto":     bool,
            "rank":       int,
        }
    """
    if not models:
        return {
            "criteria": CRITERIA,
            "weights": DEFAULT_WEIGHTS,
            "ranked": [],
            "pareto_set": [],
            "best_model": None,
        }

    eff_weights: dict[str, float] = {**DEFAULT_WEIGHTS, **(weights or {})}
    # Clamp to [0, 1]
    eff_weights = {k: max(0.0, min(1.0, v)) for k, v in eff_weights.items()}

    names = [
        m.get("model_name") or m.get("artifacts_dir", "?")
        for m in models
    ]
    raw_rows = [_raw_row(m) for m in models]
    norm_rows = _normalize(raw_rows)
    scores = [_composite_score(nr, eff_weights) for nr in norm_rows]
    pareto = _pareto_names(raw_rows, names)

    order = sorted(range(len(names)), key=lambda i: scores[i], reverse=True)
    ranked = [
        {
            "model_name": names[i],
            "raw":        raw_rows[i],
            "normalized": norm_rows[i],
            "score":      round(scores[i], 6),
            "pareto":     names[i] in pareto,
            "rank":       rank,
        }
        for rank, i in enumerate(order, start=1)
    ]

    return {
        "criteria":   CRITERIA,
        "weights":    eff_weights,
        "ranked":     ranked,
        "pareto_set": pareto,
        "best_model": ranked[0]["model_name"] if ranked else None,
    }
