"""Quarter-decade LR grid with edge extension and refinement (7.4). Pure functions of the records
already on disk, so the planner can be re-run after any interruption."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

COARSE = [1e-4, 1.8e-4, 3.2e-4, 5.6e-4, 1e-3, 1.8e-3, 3.2e-3]
STEP = 1.78
REFINE_RATIOS = [0.76, 0.87, 1.15, 1.32]
MAX_EXT = 3


def coarse_grid() -> List[float]:
    return list(COARSE)


def _r2(x: float) -> float:
    return float(f"{x:.2g}")


def extension_lr(base: float, direction: int, k: int) -> float:
    """k-th extension beyond the coarse edge (k>=1), 2 s.f."""
    return _r2(base * (STEP ** (k * direction)))


def refine_lrs(best: float) -> List[float]:
    return [_r2(best * r) for r in REFINE_RATIOS]


def score_table(medians: Dict[float, float]) -> Dict[float, float]:
    return {lr: (v if (v is not None and math.isfinite(v)) else float("inf")) for lr, v in medians.items()}


def select(medians: Dict[float, float]) -> Optional[float]:
    """Lowest median validation loss; diverged (inf/NaN) ignored; ties -> smaller LR."""
    s = score_table(medians)
    if not s or all(math.isinf(v) for v in s.values()):
        return None
    best = min(s.values())
    return min(lr for lr, v in s.items() if v == best)


def lattice() -> set:
    """Coarse grid plus the (up to MAX_EXT per side) extension points. Refinement points are NOT on it."""
    ext = [extension_lr(COARSE[0], -1, k) for k in range(1, MAX_EXT + 1)]
    ext += [extension_lr(COARSE[-1], +1, k) for k in range(1, MAX_EXT + 1)]
    return set(COARSE) | set(ext)


def plan_next(medians: Dict[float, float], present: set = frozenset()) -> dict:
    """Which LRs of this arm to run next, given the *complete* LRs in ``medians`` (lr -> median val loss).
    Stages: coarse -> edge extension (up to MAX_EXT per side, one LR at a time) -> ONE refinement round
    around the best lattice point -> done.  Final selection is over every evaluated LR."""
    coarse_missing = [lr for lr in COARSE if lr not in medians]
    if coarse_missing:
        return {"todo": coarse_missing, "stage": "coarse", "edge_warning": False, "done": False}
    lat = {lr: v for lr, v in medians.items() if lr in lattice()}
    best = select(lat)
    if best is None:
        return {"todo": [], "stage": "all-diverged", "edge_warning": True, "done": True}
    lo, hi = min(lat), max(lat)
    n_lo = sum(1 for lr in lat if lr < COARSE[0])
    n_hi = sum(1 for lr in lat if lr > COARSE[-1])
    if best == lo and n_lo < MAX_EXT:
        return {"todo": [extension_lr(COARSE[0], -1, n_lo + 1)], "stage": "extend-low", "edge_warning": False, "done": False}
    if best == hi and n_hi < MAX_EXT:
        return {"todo": [extension_lr(COARSE[-1], +1, n_hi + 1)], "stage": "extend-high", "edge_warning": False, "done": False}
    ref = [lr for lr in refine_lrs(best) if lr not in medians]
    final = select(medians)
    edge = best in (lo, hi)
    if ref:
        return {"todo": ref, "stage": "refine", "edge_warning": edge, "done": False}
    return {"todo": [], "stage": "done", "edge_warning": edge, "done": True, "best": final}


def median_over(vals: List[float]) -> float:
    if not vals:
        return float("inf")
    v = [x if (x is not None and math.isfinite(x)) else float("inf") for x in vals]
    return float(np.median(v))
