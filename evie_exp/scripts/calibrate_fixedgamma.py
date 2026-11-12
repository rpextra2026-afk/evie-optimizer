"""gamma_F for the FixedGamma arm (7.5): first 126 Evie steps at Evie's selected LR on the tuning units,
calibration seed 9001 (disjoint from tuning seeds 1001.. and evaluation seeds 42..).  Median over units."""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from .. import cells as C
from .. import tuning
from ..protocol.runlog import atomic_write_json
from ..train import train_one
from ._common import add_common, make_ctx

log = logging.getLogger("evie_exp")


def main(argv=None) -> int:
    ap = add_common(argparse.ArgumentParser())
    a = ap.parse_args(argv)
    cell, profile, ctx = make_ctx(a, "calibrate_fixedgamma")
    dirs = [ctx.results] + ctx.read_dirs
    sel = tuning.load_selected(dirs, cell)
    if "Evie" not in sel:
        raise SystemExit("Evie's LR is not selected yet: run tune_cell --arms AdamW,Evie,EvieAmplify (and --select) first")
    lr = sel["Evie"]["lr"]
    path = ctx.results / cell.name / "fixedgamma.json"
    cur = json.load(open(path)) if path.exists() else {"per_unit": {}, "clip_status": {}}
    for u in C.tune_units(cell, profile):
        if u in cur["per_unit"]:
            log.info("fixedgamma: unit %s already calibrated (checkpoint): gamma0=%.4g", u, cur["per_unit"][u])
            continue
        if ctx.stop.requested or not ctx.deadline.can_start(cell.est_run_s):
            log.info("BUDGET REACHED / stop requested - rerun to resume")
            return 10
        rec = train_one(cell=cell.name, domain=cell.domain, phase="calib", arm="Evie", unit=u, seed=C.CALIB_SEED, lr=lr,
                        steps=C.tune_steps(cell, profile), batch_size=cell.batch_size, eval_every=cell.eval_every,
                        cfg=ctx.cfg, device=ctx.device, fake_data=ctx.fake_data, stop_after=ctx.cfg["calib_step"] + 1)
        cal = rec["diagnostics"]["calibration"]
        if cal["gamma0"] is None:
            raise RuntimeError(f"calibration produced no gamma0 on {u}")
        cur["per_unit"][u], cur["clip_status"][u] = cal["gamma0"], cal["clip_status"]
        atomic_write_json(path, cur)
        log.info("fixedgamma: unit %s gamma0=%.4g (%s)", u, cal["gamma0"], cal["clip_status"])
    cur["gamma_F"] = float(np.median(list(cur["per_unit"].values())))
    cur.update({"lr_evie": lr, "calib_seed": C.CALIB_SEED})
    atomic_write_json(path, cur)
    log.info("gamma_F = %.6g (median of %d units)", cur["gamma_F"], len(cur["per_unit"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
