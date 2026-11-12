"""Run a list of jobs: sharding, deadline, graceful stop, circuit breaker, 10-min heartbeat."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .protocol.budget import Deadline, GracefulStop, Heartbeat
from .protocol.runlog import find_existing
from .runner import Job, execute

log = logging.getLogger("evie_exp")
BREAKER = 5


@dataclass
class Ctx:
    results: Path
    read_dirs: List[Path]
    cfg: dict
    device: torch.device
    deadline: Deadline
    stop: GracefulStop
    shard: Tuple[int, int] = (0, 1)
    fake_data: bool = False
    diag: bool = True
    est_run_s: float = 90.0
    summary: dict = field(default_factory=dict)


def parse_shard(s: str) -> Tuple[int, int]:
    i, n = s.split("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise ValueError(f"--shard {s}: need 0 <= i < n")
    return i, n


def shard_slice(jobs: List[Job], shard: Tuple[int, int]) -> List[Job]:
    """Contiguous chunks keep unit locality (data cache) and are stable across resumes."""
    i, n = shard
    a, b = (len(jobs) * i) // n, (len(jobs) * (i + 1)) // n
    return jobs[a:b]


def run_jobs(jobs: List[Job], ctx: Ctx, label: str) -> dict:
    mine = shard_slice(jobs, ctx.shard)
    pending = [j for j in mine if find_existing([ctx.results] + ctx.read_dirs, j.cell, j.phase, j.arm, j.unit, j.seed, j.lr) is None]
    n_done_before = len(mine) - len(pending)
    log.info("[%s] shard %d/%d: %d jobs, %d already complete (checkpoint), %d to run", label, ctx.shard[0],
             ctx.shard[1], len(mine), n_done_before, len(pending))
    st = {"ok": 0, "diverged": 0, "error": 0, "skipped": 0}
    durs: List[float] = []
    consec_err, reason = 0, "finished"
    t_start = time.time()

    def status() -> str:
        return (f"{label} shard {ctx.shard[0]}/{ctx.shard[1]}: {st['ok'] + st['diverged'] + st['error']}/{len(pending)} run "
                f"(ok={st['ok']} div={st['diverged']} err={st['error']}), "
                f"mean {sum(durs) / max(len(durs), 1):.1f}s/run, budget left {ctx.deadline.remaining_s() / 60:.0f} min")

    with Heartbeat(status):
        for i, job in enumerate(pending):
            est = 1.25 * max(durs[-20:]) if durs else ctx.est_run_s
            if ctx.stop.requested:
                reason = "signal"
                break
            if not ctx.deadline.can_start(est):
                reason = "budget"
                break
            r = execute(job, ctx.results, ctx.read_dirs, ctx.cfg, ctx.device, ctx.fake_data, ctx.diag)
            st[r["state"]] = st.get(r["state"], 0) + 1
            if r["state"] != "skipped":
                durs.append(r["wall_s"] or 0.0)
            consec_err = consec_err + 1 if r["state"] == "error" else 0
            if consec_err >= BREAKER:
                reason = "circuit-breaker"
                log.error("[%s] %d consecutive run errors: stopping (likely a code/data problem, not a flaky run)", label, BREAKER)
                break
            if (i + 1) % 10 == 0 or i + 1 == len(pending):
                log.info("[%s] progress %d/%d | %s", label, i + 1, len(pending), status())
    ran = st["ok"] + st["diverged"] + st["error"]
    left = len(pending) - ran
    if reason == "budget":
        log.info("BUDGET REACHED - rerun the same command to resume. %d runs left in this shard (%s)", left, label)
    elif reason == "signal":
        log.info("STOPPED by signal after the current run. %d runs left; rerun the same command to resume (%s)", left, label)
    elif reason == "circuit-breaker":
        log.info("STOPPED by circuit breaker. %d runs left (%s)", left, label)
    else:
        log.info("[%s] shard finished: ok=%d diverged=%d error=%d in %.1f min", label, st["ok"], st["diverged"], st["error"],
                 (time.time() - t_start) / 60)
    out = {"label": label, "reason": reason, "left": left, "ran": ran, "already_complete": n_done_before, **st}
    ctx.summary[label] = out
    return out
