"""LR tuning: planning, medians, selection.  All pure functions of the run records on disk, so the
planner can be re-run after any interruption (resumable by construction)."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .cells import Cell, tune_seeds, tune_steps, tune_units
from .protocol import lrsearch
from .protocol.runlog import atomic_write_json, is_final, load_valid, run_path
from .runner import Job

log = logging.getLogger("evie_exp")


def _rec(dirs: List[Path], cell: str, arm: str, unit: str, seed: int, lr: float) -> Optional[dict]:
    for d in dirs:
        r = load_valid(run_path(d, cell, "tune", arm, unit, seed, lr))
        if is_final(r):
            return r
    return None


def lr_medians(dirs: List[Path], cell: Cell, profile: dict, arm: str, candidate_lrs: List[float]) -> Dict[float, float]:
    """Complete LRs only: every (tune unit, tune seed) has a final record. Diverged/errored -> +inf."""
    units, seeds = tune_units(cell, profile), tune_seeds(cell, profile)
    out: Dict[float, float] = {}
    for lr in candidate_lrs:
        vals, complete = [], True
        for u in units:
            for s in seeds:
                r = _rec(dirs, cell.name, arm, u, s, lr)
                if r is None:
                    complete = False
                    break
                m = r.get("metrics", {})
                v = m.get("val_loss_best", float("nan")) if r["status"] == "ok" else float("inf")
                vals.append(v if v == v else float("inf"))
            if not complete:
                break
        if complete:
            out[lr] = lrsearch.median_over(vals)
    return out


def known_lrs(dirs: List[Path], cell: Cell, arm: str) -> List[float]:
    seen = set()
    for d in dirs:
        for p in (Path(d) / cell.name / "tune").glob(f"{arm}__*.json"):
            try:
                seen.add(float(p.stem.rsplit("__lr", 1)[1]))
            except Exception:
                pass
    return sorted(seen | set(lrsearch.COARSE))


def plan(dirs: List[Path], cell: Cell, profile: dict, arms: List[str]) -> Tuple[Dict[str, dict], List[Tuple[str, float]]]:
    """Returns (per-arm plan info, [(arm, lr)] still to run in the next batch)."""
    info, todo = {}, []
    for arm in arms:
        med = lr_medians(dirs, cell, profile, arm, known_lrs(dirs, cell, arm))
        p = lrsearch.plan_next(med)
        info[arm] = p
        todo += [(arm, lr) for lr in p["todo"]]
    return info, todo


def jobs_for(cell: Cell, profile: dict, todo: List[Tuple[str, float]], fixed_gamma: Optional[float] = None) -> List[Job]:
    steps = tune_steps(cell, profile)
    jobs = []
    for arm, lr in todo:
        for u in tune_units(cell, profile):
            for s in tune_seeds(cell, profile):
                jobs.append(Job(cell.name, cell.domain, "tune", arm, u, s, lr, steps, cell.batch_size, cell.eval_every,
                                fixed_gamma=fixed_gamma if arm == "FixedGamma" else None))
    jobs.sort(key=lambda j: (j.unit, j.seed, j.arm, j.lr))
    return jobs


def select(dirs: List[Path], cell: Cell, profile: dict, arms: List[str], out_dir: Path) -> dict:
    """Write results/<cell>/lr_selected.json (merging arms selected earlier)."""
    path = Path(out_dir) / cell.name / "lr_selected.json"
    cur = json.load(open(path)) if path.exists() else {}
    for arm in arms:
        med = lr_medians(dirs, cell, profile, arm, known_lrs(dirs, cell, arm))
        p = lrsearch.plan_next(med)
        if not p["done"]:
            raise RuntimeError(f"{arm}: tuning incomplete (stage {p['stage']}, still need {p['todo']}); run tune_cell first")
        best = lrsearch.select(med)
        if best is None:
            raise RuntimeError(f"{arm}: every tuning run diverged; nothing to select")
        cur[arm] = {"lr": best, "curve": {f"{k:.6g}": (v if math.isfinite(v) else None) for k, v in sorted(med.items())},
                    "edge_warning": bool(p["edge_warning"]), "tune_steps": tune_steps(cell, profile)}
    atomic_write_json(path, cur)
    return cur


def load_selected(dirs: List[Path], cell: Cell) -> dict:
    for d in dirs:
        p = Path(d) / cell.name / "lr_selected.json"
        if p.exists():
            return json.load(open(p))
    return {}
