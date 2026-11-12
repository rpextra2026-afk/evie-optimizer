"""One command per cell: tune -> select -> calibrate gamma_F -> tune FixedGamma -> select -> phase A -> phase B,
on ``--gpus`` GPUs in parallel, with the 9h55 budget, and a resume pack written at the end (or at the budget).

    python -m evie_exp.scripts.run_all --cell AV --gpus 2

Every step is idempotent; after a cutoff, download results_pack.zip, attach it to the next session and rerun
the SAME command: finished runs are skipped and the log says it resumed from a checkpoint."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from ..domains.base import data_dir
from ..logsetup import results_dir, setup
from ..protocol import resume
from ..protocol.budget import DEFAULT_BUDGET_MIN, Deadline, GracefulStop, Heartbeat
from .pack_results import default_out

log = logging.getLogger("evie_exp")


def _launch(module: str, base: list, gpus: int, cpu: bool) -> int:
    procs = []
    n = 1 if cpu else gpus
    for i in range(n):
        cmd = [sys.executable, "-m", f"evie_exp.scripts.{module}", *base, "--shard", f"{i}/{n}"]
        if not cpu:
            cmd += ["--device", f"cuda:{i}"]
        log.info("launch: %s", " ".join(cmd))
        procs.append(subprocess.Popen(cmd))
    return max(p.wait() for p in procs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["AV", "AF", "AL"])
    ap.add_argument("--profile", default="ea")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--results", default=None)
    ap.add_argument("--results-in", action="append", default=[])
    ap.add_argument("--time-budget-min", type=float, default=DEFAULT_BUDGET_MIN)
    ap.add_argument("--elapsed-offset-min", type=float, default=0.0)
    ap.add_argument("--max-units", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    res = results_dir(a.results)
    setup(res, f"run_all_{a.cell}")
    stats = resume.restore_from_packs(res, [Path(p) for p in a.results_in], data_dir())
    resume.log_resume(stats, res)
    dl = Deadline(a.time_budget_min, None, a.elapsed_offset_min)
    stop = GracefulStop().install()
    common = ["--cell", a.cell, "--profile", a.profile, "--results", str(res), "--deadline-epoch", str(dl.deadline)]
    for r in a.results_in:
        common += ["--results-in", r]
    if a.smoke:
        common += ["--smoke"]
    if a.max_units:
        common += ["--max-units", str(a.max_units)]
    from .. import cells as C
    prof = C.load_profile(a.profile)
    tuned = [x for x in prof["arms"] if x != "FixedGamma"]
    arms_csv = ",".join(tuned)
    steps = [("tune_cell", ["--arms", arms_csv]), ("tune_cell", ["--arms", arms_csv, "--select"])]
    if "FixedGamma" in prof["arms"]:
        steps += [("calibrate_fixedgamma", []), ("tune_cell", ["--arms", "FixedGamma"]), ("tune_cell", ["--arms", "FixedGamma", "--select"])]
    steps += [("run_cell", ["--phase", "A"]), ("run_cell", ["--phase", "B"])]
    rc_final, completed_all = 0, True
    with Heartbeat(lambda: f"run_all {a.cell}: budget left {dl.remaining_s() / 60:.0f} min"):
        for name, extra in steps:
            if stop.requested or dl.remaining_s() < 120:
                completed_all = False
                log.info("BUDGET/SIGNAL: stopping before step %s %s", name, extra)
                break
            log.info("=== STEP %s %s ===", name, " ".join(extra))
            rc = _launch(name, common + extra, a.gpus, a.cpu or a.smoke)
            if rc == 10:        # clean stop: budget or signal. Everything is checkpointed; rerun resumes.
                completed_all = False
                log.info("BUDGET REACHED (or signal) during step %s: stopping cleanly, nothing is lost", name)
                break
            if rc != 0:
                log.error("step %s exited with %d - stopping; fix the cause and rerun the same command", name, rc)
                rc_final, completed_all = rc, False
                break
            if dl.remaining_s() < 120:
                completed_all = False
                break
    info = resume.write_pack(res, default_out() if not a.smoke else res.parent / "results_pack_smoke.zip", data_dir())
    log.info("PACK written: %s (%d result files, %.1f MB)", info["zip"], info["n_result_files"], info["bytes"] / 2 ** 20)
    if completed_all:
        log.info("ALL STEPS COMPLETE for cell %s", a.cell)
    else:
        log.info("NOT COMPLETE: download %s, attach it to the next session and rerun the same command to resume.", info["zip"])
    return rc_final


if __name__ == "__main__":
    raise SystemExit(main())
