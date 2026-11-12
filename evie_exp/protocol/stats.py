"""Exact paired tests, Holm, bootstrap (7.9). Implemented here, tested against scipy."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy import stats as sps


def _avg_ranks(a: np.ndarray) -> np.ndarray:
    return sps.rankdata(a, method="average")


def paired_wilcoxon_exact(a, b, mc_flips: int = 200_000, seed: int = 0) -> dict:
    """Two-sided Wilcoxon signed-rank on paired samples. Zero differences are dropped, ties get
    average ranks. Exact by sign enumeration for n<=20, else normal approximation (tie + continuity
    corrected) plus a sign-flip Monte-Carlo p."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return {"n": len(d), "n_nonzero": 0, "W": float("nan"), "p": 1.0, "method": "degenerate"}
    r = _avg_ranks(np.abs(nz))
    w_plus = float(r[nz > 0].sum())
    total = float(r.sum())
    if n <= 20:
        r2 = np.rint(r * 2).astype(np.int64)  # half-integer ranks -> integers
        sums = np.zeros(1, dtype=np.int64)
        for ri in r2:  # DP over subset sums keeps memory O(sum), not 2^n
            sums = np.concatenate([sums, sums + ri])
        obs2, tot2 = int(round(w_plus * 2)), int(round(total * 2))
        dev = np.abs(sums - tot2 / 2.0)
        p = float(np.mean(dev >= abs(obs2 - tot2 / 2.0) - 1e-9))
        return {"n": len(d), "n_nonzero": n, "W": w_plus, "p": min(1.0, p), "method": "exact"}
    mu = total / 2.0
    _, cnt = np.unique(np.abs(nz), return_counts=True)
    sigma2 = n * (n + 1) * (2 * n + 1) / 24.0 - (cnt ** 3 - cnt).sum() / 48.0
    z = (abs(w_plus - mu) - 0.5) / math.sqrt(sigma2)
    p_norm = float(2 * sps.norm.sf(max(z, 0.0)))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(mc_flips, n)).astype(np.int8)
    wm = signs @ r
    p_mc = float(np.mean(np.abs(wm - mu) >= abs(w_plus - mu) - 1e-9))
    return {"n": len(d), "n_nonzero": n, "W": w_plus, "p": min(1.0, p_norm), "p_mc": p_mc, "method": "normal+mc"}


def sign_flip_exact(d) -> dict:
    """Paired permutation test on the mean difference; exact for n<=20 else Monte-Carlo."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    obs = abs(d.mean()) if n else float("nan")
    if n == 0:
        return {"n": 0, "p": float("nan"), "method": "empty"}
    if n <= 20:
        sums = np.zeros(1)
        for x in d:
            sums = np.concatenate([sums + x, sums - x])
        return {"n": n, "p": float(np.mean(np.abs(sums) / n >= obs - 1e-12)), "method": "exact"}
    rng = np.random.default_rng(0)
    s = rng.choice([-1.0, 1.0], size=(200_000, n)) @ d / n
    return {"n": n, "p": float(np.mean(np.abs(s) >= obs - 1e-12)), "method": "mc"}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(((k, p) for k, p in pvals.items() if p == p), key=lambda kv: kv[1])
    m = len(items)
    out = {k: float("nan") for k in pvals}
    run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def bootstrap_ci(d, stat: str = "mean", B: int = 20_000, seed: int = 0, alpha: float = 0.05) -> tuple:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    f = np.mean if stat == "mean" else np.median
    vals = f(d[idx], axis=1)
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def win_count(a, b, higher_is_better: bool) -> int:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return int(np.sum((a > b) if higher_is_better else (a < b)))


def min_exact_p(n: int) -> float:
    """Smallest attainable two-sided exact Wilcoxon p for n pairs (sanity print)."""
    return 2.0 / (2 ** n)
