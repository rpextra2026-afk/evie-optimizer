"""stdout + appended log file (results/logs/<name>.log).  Appending means a resumed session's log
continues the earlier one, and the RESUME lines say whether it started from a checkpoint."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def results_dir(arg: str | None = None) -> Path:
    if arg:
        return Path(arg)
    env = os.environ.get("EVIE_RESULTS")
    if env:
        return Path(env)
    return Path("/kaggle/working/results") if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else Path("./results")


def setup(res: Path, name: str) -> logging.Logger:
    log = logging.getLogger("evie_exp")
    if getattr(log, "_evie_setup", None) == (str(res), name):
        return log
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    logdir = Path(res) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logdir / f"{name}.log", mode="a")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.propagate = False
    log._evie_setup = (str(res), name)  # type: ignore[attr-defined]
    return log
