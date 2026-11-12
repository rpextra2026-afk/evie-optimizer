"""Zip everything needed to resume (or analyse) -> /kaggle/working/results_pack.zip."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from ..domains.base import data_dir
from ..logsetup import results_dir, setup
from ..protocol.resume import PACK_NAME, write_pack

log = logging.getLogger("evie_exp")


def default_out() -> Path:
    if os.environ.get("EVIE_PACK"):
        return Path(os.environ["EVIE_PACK"])
    return (Path("/kaggle/working") if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else Path(".")) / PACK_NAME


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-data", action="store_true", help="skip the small finance/language caches")
    a = ap.parse_args(argv)
    res = results_dir(a.results)
    setup(res, "pack_results")
    out = Path(a.out) if a.out else default_out()
    info = write_pack(res, out, data_dir(), include_data=not a.no_data)
    log.info("PACK written: %s (%d result files, %.1f MB). Download it, attach it to the next session, rerun the same command.",
             info["zip"], info["n_result_files"], info["bytes"] / 2 ** 20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
