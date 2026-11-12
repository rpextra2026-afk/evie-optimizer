"""Tune LRs for the given arms (resumable, shardable), or select them.

  python -m evie_exp.scripts.tune_cell --cell AV --arms AdamW,Evie,EvieAmplify          # run (loops stages)
  python -m evie_exp.scripts.tune_cell --cell AV --arms AdamW,Evie,EvieAmplify --select # CPU, seconds
"""
from __future__ import annotations

import argparse
import json
import logging
import time

from .. import tuning
from ..engine import run_jobs
from ._common import add_common, make_ctx

log = logging.getLogger("evie_exp")
BARRIER_MAX_S = 1800


def _barrier(ctx, dirs, cell, profile, arms, todo_jobs) -> bool:
    """Wait until every job of this round (all shards) has a final record."""
    from ..protocol.runlog import find_existing
    t0 = time.time()
    while True:
        miss = [j for j in todo_jobs if find_existing(dirs, j.cell, j.phase, j.arm, j.unit, j.seed, j.lr) is None]
        if not miss:
            return True
        if ctx.stop.requested or not ctx.deadline.can_start(60) or time.time() - t0 > BARRIER_MAX_S:
            log.info("barrier: %d jobs of the other shard(s) not finished; stopping (rerun to resume)", len(miss))
            return False
        time.sleep(15)


def main(argv=None) -> int:
    ap = add_common(argparse.ArgumentParser())
    ap.add_argument("--arms", default="AdamW,Evie,EvieAmplify")
    ap.add_argument("--select", action="store_true", help="write lr_selected.json from finished tuning")
    ap.add_argument("--plan", action="store_true", help="print the next batch and exit")
    a = ap.parse_args(argv)
    arms = a.arms.split(",")
    cell, profile, ctx = make_ctx(a, "tune_cell")
    dirs = [ctx.results] + ctx.read_dirs
    if a.select:
        sel = tuning.select(dirs, cell, profile, arms, ctx.results)
        for arm in arms:
            log.info("selected %s: lr=%g edge_warning=%s", arm, sel[arm]["lr"], sel[arm]["edge_warning"])
        return 0
    fg = None
    if "FixedGamma" in arms:
        fp = ctx.results / cell.name / "fixedgamma.json"
        if not fp.exists():
            raise SystemExit("FixedGamma needs results/<cell>/fixedgamma.json: run calibrate_fixedgamma first")
        fg = json.load(open(fp))["gamma_F"]
    rounds = 0
    while True:
        info, todo = tuning.plan(dirs, cell, profile, arms)
        for arm, p in info.items():
            log.info("plan %s: stage=%s todo=%s", arm, p["stage"], p["todo"])
        if a.plan or not todo:
            if not todo:
                log.info("tuning complete for %s (rounds run this session: %d)", arms, rounds)
            return 0
        jobs = tuning.jobs_for(cell, profile, todo, fg)
        r = run_jobs(jobs, ctx, f"tune {cell.name} round {rounds}")
        rounds += 1
        if r["reason"] != "finished":
            return 10 if r["reason"] in ("budget", "signal") else 3
        if ctx.shard[1] > 1 and not _barrier(ctx, dirs, cell, profile, arms, jobs):
            return 10


if __name__ == "__main__":
    raise SystemExit(main())
