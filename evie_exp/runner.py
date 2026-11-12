"""Run one (arm, unit, seed, lr) job with skip-if-final, error handling and atomic JSON output."""
from __future__ import annotations

import gc
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from .protocol.runlog import MAX_ATTEMPTS, atomic_write_json, find_existing, load_valid, run_path
from .train import train_one

log = logging.getLogger("evie_exp")


@dataclass
class Job:
    cell: str
    domain: str
    phase: str
    arm: str
    unit: str
    seed: int
    lr: float
    steps: int
    batch_size: int
    eval_every: int
    lr_source: str = "tuned"
    fixed_gamma: Optional[float] = None
    rho_bar: Optional[float] = None

    def key(self) -> str:
        return f"{self.cell}/{self.phase}/{self.arm}/{self.unit}/s{self.seed}/lr{self.lr:.6g}"


def execute(job: Job, results: Path, read_dirs: List[Path], cfg: dict, device: torch.device,
            fake_data: bool = False, diag: bool = True) -> dict:
    """Returns {"state": "skipped"|"ok"|"diverged"|"error", "path", "wall_s"}.  Never raises for a
    failing run: the failure is written to the record (retried in later sessions up to MAX_ATTEMPTS)."""
    existing = find_existing([results] + list(read_dirs), job.cell, job.phase, job.arm, job.unit, job.seed, job.lr)
    if existing is not None:
        return {"state": "skipped", "path": existing, "wall_s": 0.0}
    out = run_path(results, job.cell, job.phase, job.arm, job.unit, job.seed, job.lr)
    prev = load_valid(out)
    attempts = (prev or {}).get("attempts", 0)
    last_exc = None
    for tries in range(2):  # one in-session retry (CUDA OOM / transient)
        t0 = time.time()
        try:
            rec = train_one(cell=job.cell, domain=job.domain, phase=job.phase, arm=job.arm, unit=job.unit,
                            seed=job.seed, lr=job.lr, steps=job.steps, batch_size=job.batch_size,
                            eval_every=job.eval_every, cfg=cfg, device=device, fake_data=fake_data,
                            lr_source=job.lr_source, fixed_gamma=job.fixed_gamma, rho_bar=job.rho_bar, diag=diag)
            rec["attempts"] = attempts + tries + 1
            atomic_write_json(out, rec)
            return {"state": rec["status"], "path": out, "wall_s": time.time() - t0}
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            last_exc = e
            log.error("run failed (%s): %r", job.key(), e)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)
        except Exception as e:  # data missing, bad config, bug: do not hammer, record and move on
            last_exc = e
            log.error("run failed (%s): %r\n%s", job.key(), e, traceback.format_exc())
            break
    rec = {"schema_version": 1, "cell": job.cell, "domain": job.domain, "phase": job.phase, "arm": job.arm,
           "unit": job.unit, "seed": job.seed, "lr": job.lr, "status": "error", "error": repr(last_exc),
           "traceback": traceback.format_exception_only(type(last_exc), last_exc)[-1].strip(),
           "attempts": attempts + 2, "metrics": {"primary_name": "none", "primary": float("nan")}}
    atomic_write_json(out, rec)
    final = rec["attempts"] >= MAX_ATTEMPTS
    log.error("run %s recorded as ERROR (attempt %d/%d)%s", job.key(), rec["attempts"], MAX_ATTEMPTS,
              " - PERMANENT, will be reported as failed" if final else " - will be retried next session")
    return {"state": "error", "path": out, "wall_s": 0.0}
