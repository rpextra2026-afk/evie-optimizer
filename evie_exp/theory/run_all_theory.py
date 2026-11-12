"""Run R1a-R1g sequentially. Prints ALL CLAIMS PASS or the list of FALSIFIED ids (Gate G0: any FALSIFIED claim stops everything).
    python -m evie_exp.theory.run_all_theory [--smoke] [--out results/theory] [--only R1a,R1e] [--seed 0]"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

from .common import main_for

MODULES = {"R1a": "r1a_riccati", "R1b": "r1b_control_sims", "R1c": "r1c_sgd_quadratic", "R1d": "r1d_heterogeneity",
           "R1e": "r1e_rescaling", "R1f": "r1f_c3_bound", "R1g": "r1g_gate_tables"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/theory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    ids = [i for i in MODULES if not a.only or i in a.only.split(",")]
    failed, t0 = [], time.time()
    for rid in ids:
        mod = importlib.import_module(f"evie_exp.theory.{MODULES[rid]}")
        print(f"\n===== {rid} ({MODULES[rid]}) =====", flush=True)
        main_for(rid, mod.run, ["--out", a.out, "--seed", str(a.seed)] + (["--smoke"] if a.smoke else []))
        s = json.loads((Path(a.out) / rid / "summary.json").read_text())
        failed += [f"{rid}:{c['claim_id']}" for c in s["claims"] if not c["pass"]]
    print(f"\nTotal {time.time() - t0:.0f} s")
    print("ALL CLAIMS PASS" if not failed else "FALSIFIED: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
