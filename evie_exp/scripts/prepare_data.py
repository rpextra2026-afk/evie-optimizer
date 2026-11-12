"""Download + cache data (needs internet; run once, then save $EVIE_DATA as a Kaggle Dataset)."""
from __future__ import annotations

import argparse
import json

from ..domains import finance, language, vision
from ..domains.base import data_dir
from ..logsetup import results_dir, setup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["vision", "finance", "language", "all"])
    ap.add_argument("--tickers-from", default=None, help="old finance checkpoint JSON (its 'tickers' = eval set)")
    a = ap.parse_args()
    log = setup(results_dir(None), "prepare_data")
    log.info("EVIE_DATA = %s", data_dir())
    from ..protocol import resume
    st = resume.restore_from_packs(results_dir(None), [], data_dir())      # data caches from an earlier session's pack
    if st["data_files"]:
        log.info("RESUME: restored %d cached data files from %d pack(s) (no re-download needed)", st["data_files"], len(st["packs"]))
    for d in ([a.domain] if a.domain != "all" else ["vision", "finance", "language"]):
        if d == "vision":
            vision.prepare()
        elif d == "language":
            language.prepare()
        else:
            tj = data_dir() / "finance" / "tickers.json"
            if tj.exists():
                log.info("finance: tickers.json already present (%s): keeping that exact list", tj)
                out = json.load(open(tj)); out.setdefault("from_old_checkpoint", None)
            else:
                out = finance.prepare(a.tickers_from)
            log.info("finance: eval=%d tune=%d skipped=%s from_old=%s", len(out["eval"]), len(out["tune"]),
                     out["skipped"], out["from_old_checkpoint"])
        log.info("%s data ready", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
