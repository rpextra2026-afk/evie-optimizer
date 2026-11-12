"""claim_matrix.md|tex from results/theory/*/summary.json (status computed mechanically from stored pass flags; wording from templates).
  python -m evie_exp.analysis.claim_matrix --theory results/theory --out results/tables
Status: SUPPORTED (all linked claims pass), FALSIFIED (any fails), NOT RUN (a linked claim has no summary)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# (derivation, prediction, run, [(summary id, claim id)...], supported-if, falsified-if)
ROWS = [
    ("A3", "Riccati solution, offset and escape time equal their closed forms", "R1a", [("R1a", "A3")], "max rel err < 1e-8, bound holds", "any case violates"),
    ("A4", "cost rate minimiser k*, lambda* = qRk*/2, lambda'' = delta q H/(2D^3)", "R1b", [("R1b", "A4")], "rel err < 1e-10 / 1e-8", "minimiser not at k*"),
    ("A5", "stationary variance increases with kappa; matches exact Euler value", "R1b", [("R1b", "A5")], "within 4 SE, increasing", "outside 4 SE or not increasing"),
    ("A2", "Riccati gain is optimal among perturbed gains", "R1b", [("R1b", "A2-disc"), ("R1b", "A2"), ("R1b", "A2-MC")], "J(u_eps) > J(u*) for all eps", "any perturbed gain is cheaper"),
    ("A1", "J_gamma monotone in gamma and ordered around E C", "R1b", [("R1b", "A1")], "monotone and ordered", "ordering violated"),
    ("B5", "exact Euler recursion: linear convergence, k_d = k* - eta H/(2R) + O(eta^2)", "R1b", [("R1b", "B5"), ("R1b", "B5-MC")], "ratios and residual within bands", "minimiser not within O(eta)"),
    ("B1", "shipped gate is the exact risk-seeking gain; gains multiply to H", "R1g", [("R1g", "B1")], "identity to 1e-12 (scaled)", "identity violated"),
    ("B2", "m_A(0)=1/sqrt(w), range, crossover, clamps inactive", "R1g", [("R1g", "B2")], "closed forms to 1e-12", "any closed form off"),
    ("B3", "loss-rescaling invariance inside the clip window; broken by warm-up and fixed gamma0", "R1e", [("R1e", "B3"), ("R1e", "B3-H3"), ("R1e", "B3-H4")], "deviations as predicted", "calibrated gate-off deviates > 1e-8"),
    ("B4", "calibration lemma closed forms and clip window", "R1g", [("R1g", "B4")], "closed forms to 1e-12", "any closed form off"),
    ("C0", "opposite noise-floor trends in the control and SGD models", "R1b+R1c", [("R1c", "C0")], "strictly opposite for every coordinate", "same direction"),
    ("C1", "gated quadratic recursions match simulation; stability edge at theta=2", "R1c", [("R1c", "C1"), ("R1c", "C1-asym")], "max z < 5", "z >= 5 anywhere"),
    ("C2", "shapes vs uniform vs oracle; converse; power gates; two-sided bound", "R1d",
     [("R1d", c) for c in ("C2-ordering", "C2-homogeneity", "C2-regime-a", "C2-converse", "C2-misaligned", "C2-regime-c", "C2-bound", "C2-powergates")], "all sign patterns hold", "ordering/homogeneity/sign failure"),
    ("C3", "preconditioned SGD stationarity bound", "R1f", [("R1f", "C3")], "mean within bound in every setting", "exceeds bound by > 4 SE"),
    ("D1", "fixed-gamma gate vs Evie gate gap <= (7/8) y^2", "E2 (FixedGamma, FixedGammaOwn)", [], "see ablation D1 analysis", "see ablation D1 analysis"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", default="results/theory")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args(argv)
    data = {}
    for p in Path(a.theory).glob("*/summary.json"):
        s = json.loads(p.read_text())
        data[s["id"]] = {c["claim_id"]: c for c in s["claims"]}
    md = ["| Derivation | Prediction | Run | Supported if | Falsified if | Observed | Status |", "|---|---|---|---|---|---|---|"]
    tex = ["\\begin{tabular}{llllp{3cm}l}", "Derivation & Prediction & Run & Supported if & Falsified if & Status\\\\ \\hline"]
    statuses = {}
    for der, pred, run, links, sup, fal in ROWS:
        if not links:
            st, obs = "NOT RUN", "GPU ablation (E2)"
        else:
            cl = [data.get(i, {}).get(c) for i, c in links]
            if any(c is None for c in cl):
                st, obs = "NOT RUN", "missing: " + ", ".join(f"{i}/{c}" for (i, c), x in zip(links, cl) if x is None)
            else:
                st = "SUPPORTED" if all(c["pass"] for c in cl) else "FALSIFIED"
                obs = "; ".join(f"{c['claim_id']}={'pass' if c['pass'] else 'FAIL'}" for c in cl)
        statuses[der] = st
        md.append(f"| {der} | {pred} | {run} | {sup} | {fal} | {obs} | {st} |")
        tex.append(f"{der} & {pred} & {run} & {sup} & {fal} & {st}\\\\")
    tex.append("\\end{tabular}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "claim_matrix.md").write_text("\n".join(md) + "\n")
    (out / "claim_matrix.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))
    print("\nFALSIFIED:", [k for k, v in statuses.items() if v == "FALSIFIED"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
