"""Checkpoint = the set of finished run JSONs. A session can end at any time (budget / kill); the
user downloads ``results_pack.zip``, attaches it to the next session, and the same command resumes.

``restore_from_packs`` scans ``/kaggle/input`` (and any ``--results-in`` directory/zip) for packs,
copies every *valid* run record that is not already present, and reports exactly what it did so the
log says whether the run started from a checkpoint.
"""
from __future__ import annotations

import json
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Iterable, List

from .runlog import atomic_write_json, validate

log = logging.getLogger("evie_exp")
PACK_NAME = "results_pack.zip"
MANIFEST = "PACK_MANIFEST.json"


def kaggle_roots() -> List[Path]:
    r = Path("/kaggle/input")
    return [r] if r.exists() else []


def _walk_packs(root: Path, max_depth: int = 5) -> Iterable[Path]:
    """Yield *.zip files and directories named 'results' under root (bounded depth)."""
    root = Path(root)
    if root.is_file() and root.suffix == ".zip":
        yield root
        return
    if not root.exists():
        return
    base = len(root.parts)
    for dp, dn, fn in os.walk(root):
        if len(Path(dp).parts) - base > max_depth:
            dn[:] = []
            continue
        for f in fn:
            if f.endswith(".zip"):
                yield Path(dp) / f
        if Path(dp).name == "results":
            yield Path(dp)


def _copy_if_missing(name: str, data: bytes, dest: Path, stats: dict) -> None:
    if dest.exists():
        stats["already_present"] += 1
        return
    try:
        rec = json.loads(data)
    except Exception:
        stats["corrupt_skipped"] += 1
        return
    if name.endswith(".json") and ("/tune/" in name or "/eval/" in name):
        if not validate(rec):
            stats["corrupt_skipped"] += 1
            return
    dest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest, rec)
    stats["restored"] += 1


def restore_from_packs(results_dir: Path, extra: Iterable[Path] = (), data_dir: Path | None = None) -> dict:
    """Merge records from previous sessions into ``results_dir``. Never overwrites an existing file."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stats = {"packs": [], "restored": 0, "already_present": 0, "corrupt_skipped": 0, "data_files": 0}
    roots = [Path(p) for p in extra] + kaggle_roots()
    seen = set()
    for root in roots:
        for pack in _walk_packs(root):
            if pack.resolve() in seen or pack.resolve() == results_dir.resolve():
                continue
            seen.add(pack.resolve())
            n0 = stats["restored"]
            if pack.is_dir():
                for f in pack.rglob("*.json"):
                    rel = f.relative_to(pack)
                    _copy_if_missing(str(rel), f.read_bytes(), results_dir / rel, stats)
            else:
                try:
                    with zipfile.ZipFile(pack) as z:
                        names = z.namelist()
                        if MANIFEST not in names:
                            continue  # not one of ours
                        for nm in names:
                            if nm.startswith("results/") and nm.endswith(".json"):
                                _copy_if_missing(nm, z.read(nm), results_dir / nm[len("results/"):], stats)
                            elif nm.startswith("data/") and data_dir is not None and not nm.endswith("/"):
                                dest = Path(data_dir) / nm[len("data/"):]
                                if not dest.exists():
                                    dest.parent.mkdir(parents=True, exist_ok=True)
                                    dest.write_bytes(z.read(nm))
                                    stats["data_files"] += 1
                except zipfile.BadZipFile:
                    log.warning("bad zip skipped: %s", pack)
                    continue
            stats["packs"].append({"path": str(pack), "restored": stats["restored"] - n0})
    return stats


def log_resume(stats: dict, results_dir: Path) -> None:
    if stats["restored"] or stats["already_present"]:
        log.info("RESUME: started from checkpoint. restored %d run records from %d pack(s) "
                 "(%d already present, %d corrupt skipped, %d data files) into %s",
                 stats["restored"], len(stats["packs"]), stats["already_present"],
                 stats["corrupt_skipped"], stats["data_files"], results_dir)
        for p in stats["packs"]:
            log.info("RESUME:   pack %s -> %d records", p["path"], p["restored"])
    else:
        log.info("RESUME: no previous checkpoint found; starting FRESH in %s", results_dir)


def write_pack(results_dir: Path, out_zip: Path, data_dir: Path | None = None, include_data: bool = True,
               extra_files: Iterable[Path] = (), note: str = "") -> dict:
    """Zip results (+ small data caches + logs) atomically."""
    results_dir, out_zip = Path(results_dir), Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_zip.with_suffix(".zip.tmp")
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(results_dir.rglob("*")):
            if f.is_file() and not f.name.endswith((".tmp", ".corrupt")) and f.resolve() != out_zip.resolve():
                z.write(f, "results/" + str(f.relative_to(results_dir)))
                n += 1
        if include_data and data_dir and Path(data_dir).exists():
            for sub in ("finance", "language"):  # small caches; CIFAR is re-downloaded
                for f in sorted((Path(data_dir) / sub).rglob("*")) if (Path(data_dir) / sub).exists() else []:
                    if f.is_file():
                        z.write(f, "data/" + str(f.relative_to(data_dir)))
        for f in extra_files:
            if Path(f).is_file():
                z.write(f, "logs/" + Path(f).name)
        z.writestr(MANIFEST, json.dumps({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_result_files": n,
                                         "note": note}))
    os.replace(tmp, out_zip)
    return {"zip": str(out_zip), "n_result_files": n, "bytes": out_zip.stat().st_size}
