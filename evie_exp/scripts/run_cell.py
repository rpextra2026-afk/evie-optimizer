"""Evaluation runs.  Phase A: AdamW, Evie, FixedGamma, EvieAmplify at their tuned LRs.
Phase B: controls M1, M2 and FixedGammaOwn (derived LRs, no tuning)."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .. import cells as C
from .. import tuning
from ..engine import run_jobs
from ..protocol.runlog import atomic_write_json, is_final, load_valid, run_path
from ..runner import Job
from ._common import add_common, make_ctx

log = logging.getLogger("evie_exp")


def _jobs(cell, profile, units, seeds, arm, lr, **kw):
    return [Job(cell.name, cell.domain, "eval", arm, u, s, lr, cell.steps, cell.batch_size, cell.eval_every, **kw)
            for u in units for s in seeds]


def phase_a(cell, profile, ctx, units, seeds):
    sel = tuning.load_selected([ctx.results] + ctx.read_dirs, cell)
    arms = [a for a in profile["arms"]]
    miss = [a for a in arms if a not in sel]
    if miss:
        raise SystemExit(f"missing tuned LR for {miss}: finish tune_cell/--select (and calibrate_fixedgamma) first")
    fg = json.load(open(ctx.results / cell.name / "fixedgamma.json"))["gamma_F"] if "FixedGamma" in arms else None
    jobs = []
    for arm in arms:
        jobs += _jobs(cell, profile, units, seeds, arm, sel[arm]["lr"], fixed_gamma=fg if arm == "FixedGamma" else None)
    jobs.sort(key=lambda j: (j.unit, j.seed, j.arm))
    return jobs


def rho_bar(dirs, cell, sel, units, seeds, calib_step):
    """Mean over all post-calibration eval intervals, units and seeds of Evie's realised rho_rms."""
    vals = []
    for u in units:
        for s in seeds:
            for d in dirs:
                r = load_valid(run_path(d, cell.name, "eval", "Evie", u, s, sel["Evie"]["lr"]))
                if r and r["status"] == "ok":
                    ts = r["diagnostics"]["timeseries"]
                    vals += [v for st, v in zip(ts["eval_step"], ts["rho_rms"]) if st > calib_step and v == v]
                    break
    if not vals:
        raise RuntimeError("no post-calibration rho_rms found in Evie eval runs: finish phase A first")
    return float(np.mean(vals)), len(vals)


def phase_b_generic(cell, profile, ctx, units, seeds):
    """Derived arms: each runs at the selected LR of its source arm (no tuning of its own)."""
    sel = tuning.load_selected([ctx.results] + ctx.read_dirs, cell)
    jobs = []
    for arm, src in profile["derived"].items():
        if src not in sel:
            raise SystemExit(f"{arm} needs the tuned LR of {src}: finish tune_cell/--select first")
        log.info("derived arm %s runs at the LR of %s = %g", arm, src, sel[src]["lr"])
        jobs += _jobs(cell, profile, units, seeds, arm, sel[src]["lr"], lr_source=f"derived:{src}")
    jobs.sort(key=lambda j: (j.unit, j.seed, j.arm))
    return jobs


def phase_b(cell, profile, ctx, units, seeds):
    if "derived" in profile:
        return phase_b_generic(cell, profile, ctx, units, seeds)
    dirs = [ctx.results] + ctx.read_dirs
    sel = tuning.load_selected(dirs, cell)
    rb, n = rho_bar(dirs, cell, sel, units, seeds, ctx.cfg["calib_step"])
    atomic_write_json(ctx.results / cell.name / "controls.json", {"rho_bar": rb, "n_intervals": n, "lr_M1": sel["EvieAmplify"]["lr"],
                                                                   "lr_M2": sel["Evie"]["lr"]})
    log.info("controls: rho_bar=%.6f from %d post-calibration intervals; M1 lr=%g, M2 lr=%g", rb, n,
             sel["EvieAmplify"]["lr"], sel["Evie"]["lr"])
    jobs = _jobs(cell, profile, units, seeds, "M1", sel["EvieAmplify"]["lr"], lr_source="derived:M1")
    jobs += _jobs(cell, profile, units, seeds, "M2", sel["Evie"]["lr"], lr_source="derived:M2", rho_bar=rb)
    if "FixedGammaOwn" in profile.get("extra_arms", []):
        for u in units:
            for s in seeds:
                er = None
                for d in dirs:
                    er = load_valid(run_path(d, cell.name, "eval", "Evie", u, s, sel["Evie"]["lr"]))
                    if er:
                        break
                g = er and er["diagnostics"]["calibration"]["gamma0"]
                if not g or er["status"] != "ok":
                    log.warning("FixedGammaOwn %s s%d skipped: no Evie eval record with calib.gamma0 (never guessed)", u, s)
                    continue
                jobs.append(Job(cell.name, cell.domain, "eval", "FixedGammaOwn", u, s, sel["Evie"]["lr"], cell.steps,
                                cell.batch_size, cell.eval_every, lr_source="derived:FixedGammaOwn", fixed_gamma=float(g)))
    jobs.sort(key=lambda j: (j.unit, j.seed, j.arm))
    return jobs


def main(argv=None) -> int:
    ap = add_common(argparse.ArgumentParser())
    ap.add_argument("--phase", required=True, choices=["A", "B"])
    a = ap.parse_args(argv)
    cell, profile, ctx = make_ctx(a, f"run_cell_{a.phase}")
    units = C.eval_units(cell, a.max_units, a.fake_data)
    seeds = C.eval_seeds(cell, profile)
    if a.smoke:
        units, seeds = units[:2], seeds[:1]
        ctx.cfg["smoke_steps"] = True
    jobs = phase_a(cell, profile, ctx, units, seeds) if a.phase == "A" else phase_b(cell, profile, ctx, units, seeds)
    if a.smoke:
        for j in jobs:
            j.steps, j.eval_every = 200, 100
    r = run_jobs(jobs, ctx, f"eval {cell.name} phase {a.phase}")
    return 0 if r["reason"] == "finished" else (10 if r["reason"] in ("budget", "signal") else 3)


if __name__ == "__main__":
    raise SystemExit(main())
