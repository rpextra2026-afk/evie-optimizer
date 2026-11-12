"""Shared contract of every R1x script: --smoke --out --seed, deterministic PCG64, summary.json with claims[]."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np


def claim(claim_id: str, statement: str, criterion: str, observed: dict, ok: bool) -> dict:
    return {"claim_id": claim_id, "statement": statement, "criterion": criterion,
            "observed": _clean(observed), "pass": bool(ok)}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else str(f)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def rng_for(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def write_summary(out: Path, rid: str, runtime_s: float, params: dict, claims: List[dict], files: List[str], smoke: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    s = {"id": rid, "runtime_s": round(runtime_s, 2), "smoke": bool(smoke), "params": _clean(params),
         "claims": claims, "files": files}
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    return s


def main_for(rid: str, fn: Callable, argv=None) -> int:
    """fn(rng, smoke, outdir) -> (params, claims, files). Exit code 0 even if a claim fails (the verdict is data)."""
    ap = argparse.ArgumentParser(description=rid)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/theory")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out) / rid
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    params, claims, files = fn(rng_for(a.seed), a.smoke, out)
    s = write_summary(out, rid, time.time() - t0, {**params, "seed": a.seed}, claims, files, a.smoke)
    for c in claims:
        print(f"[{rid}] {c['claim_id']:<8s} {'PASS' if c['pass'] else 'FALSIFIED':9s} {c['criterion']} | {json.dumps(c['observed'])[:300]}")
    print(f"[{rid}] done in {s['runtime_s']} s -> {out/'summary.json'}")
    return 0


# ---- gates (numpy, float64): Appendix A -----------------------------------------------------------------------------
def s_E(x):
    return (1.0 + np.asarray(x, float)) ** -0.5


def m_0(x):
    x = np.asarray(x, float)
    return np.sqrt((1.0 + x) / (1.0 + 2.0 * x))


def m_A(x, w=0.85):
    return m_0(x) / np.sqrt(w)


def golden_min(f, lo, hi, iters=80):
    """Elementwise golden-section minimiser of a vectorised f on [lo, hi] (arrays), returns the argmin."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    lo, hi = np.array(lo, float), np.array(hi, float)
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        left = fc < fd
        hi = np.where(left, d, hi)
        lo = np.where(left, lo, c)
        c_new, d_new = hi - gr * (hi - lo), lo + gr * (hi - lo)
        fc_n, fd_n = f(c_new), f(d_new)
        fc, fd, c, d = fc_n, fd_n, c_new, d_new
    return 0.5 * (lo + hi)
