"""One JSON file per run, written atomically; corrupt or partial files are detected and re-run.

Status semantics for resume: ``ok`` and ``diverged`` are final. ``error`` is retried until
``MAX_ATTEMPTS`` attempts have been made (counted across sessions), after which it is final and
reported as a failed run (never silently dropped).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1
MAX_ATTEMPTS = 3
REQUIRED = ("schema_version", "cell", "phase", "arm", "unit", "seed", "lr", "status", "metrics")


def run_filename(arm: str, unit: str, seed: int, lr: float) -> str:
    return f"{arm}__{unit}__s{seed}__lr{lr:.6g}.json"


def run_path(results: Path, cell: str, phase: str, arm: str, unit: str, seed: int, lr: float) -> Path:
    return Path(results) / cell / phase / run_filename(arm, unit, seed, lr)


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, allow_nan=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def validate(rec: dict) -> bool:
    return isinstance(rec, dict) and all(k in rec for k in REQUIRED) and rec["schema_version"] == SCHEMA_VERSION


def load_valid(path: Path) -> Optional[dict]:
    """Return the record if the file exists and validates; quarantine it as ``.corrupt`` otherwise."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            rec = json.load(f)
        if validate(rec):
            return rec
        reason = "schema"
    except Exception as e:  # truncated / garbage
        reason = repr(e)
    try:
        os.replace(path, str(path) + ".corrupt")
    except OSError:
        pass
    print(f"[WARN] corrupt run record quarantined ({reason}): {path}", flush=True)
    return None


def is_final(rec: Optional[dict]) -> bool:
    if rec is None:
        return False
    if rec["status"] in ("ok", "diverged"):
        return True
    return rec["status"] == "error" and rec.get("attempts", 1) >= MAX_ATTEMPTS


def find_existing(dirs: list[Path], cell: str, phase: str, arm: str, unit: str, seed: int, lr: float) -> Optional[Path]:
    """First directory (writable results dir first, then ``--results-in`` dirs) holding a final record."""
    for d in dirs:
        p = run_path(d, cell, phase, arm, unit, seed, lr)
        if is_final(load_valid(p)):
            return p
    return None


def git_info(repo: Path | None = None) -> dict:
    repo = Path(repo or Path(__file__).resolve().parents[2])

    def _g(*a: str) -> str:
        try:
            return subprocess.check_output(["git", *a], cwd=repo, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return ""
    return {"git_commit": _g("rev-parse", "HEAD") or "unknown", "git_dirty": bool(_g("status", "--porcelain"))}


def env_info(device: str) -> dict:
    import numpy
    import torch
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": numpy.__version__,
            "cuda": torch.version.cuda or "none", "device": device, "platform": platform.platform(),
            "tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False))}
    if torch.cuda.is_available() and device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(torch.device(device))
    info.update(git_info())
    return info
