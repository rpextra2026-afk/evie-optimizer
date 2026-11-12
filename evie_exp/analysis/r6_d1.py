"""D1 analysis (7.10): FixedGamma/FixedGammaOwn vs Evie, floor occupancy vs paired difference, analytic bound."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .. import cells as C
from ..optim.gates import s_E, s_F
from ..protocol.runlog import load_valid, run_path
from ..protocol.stats import holm, paired_wilcoxon_exact


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(C.CELLS))
    ap.add_argument("--profile", default="ea")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--fake-data", action="store_true", help="units of a --smoke run")
    a = ap.parse_args(argv)
    cell, prof, res = C.CELLS[a.cell], C.load_profile(a.profile), Path(a.results)
    units, seeds = C.eval_units(cell, None, a.fake_data), C.eval_seeds(cell, prof)
    if a.fake_data:
        seeds = seeds[:1]
    sel = json.load(open(res / cell.name / "lr_selected.json"))
    lr = {"Evie": sel["Evie"]["lr"], "FixedGamma": sel["FixedGamma"]["lr"], "AdamW": sel["AdamW"]["lr"], "FixedGammaOwn": sel["Evie"]["lr"]}
    def get(arm, u, s):
        return load_valid(run_path(res, cell.name, "eval", arm, u, s, lr[arm]))
    lines = [f"# D1 analysis {cell.name}", ""]
    diffs, occ, expo, pvals = {}, [], [], {}
    for arm in ("FixedGamma", "FixedGammaOwn"):
        d = []
        for u in units:
            ev = [get("Evie", u, s) for s in seeds]; fx = [get(arm, u, s) for s in seeds]
            if all(r and r["status"] == "ok" for r in ev + fx):
                d.append((u, np.mean([r["metrics"]["primary"] for r in fx]) - np.mean([r["metrics"]["primary"] for r in ev]),
                          np.mean([r["diagnostics"]["timeseries"]["floor_frac"][-1] for r in fx]),
                          np.mean([r["diagnostics"]["timeseries"]["expo_gt1_frac"][-1] for r in ev])))
        if len(d) >= 3:
            x = np.array([t[1] for t in d]); t = paired_wilcoxon_exact(x, np.zeros_like(x))
            pvals[f"{arm} vs Evie"] = t["p"]
            lines.append(f"- {arm} - Evie: mean diff {x.mean():+.5g}, n_pairs {len(d)}/{len(units)}, exact p {t['p']:.4g}")
            if arm == "FixedGamma":
                fo = np.array([t[2] for t in d]); ex = np.array([t[3] for t in d])
                if np.ptp(fo) > 0:
                    lines.append(f"- Spearman(floor_frac of FixedGamma run, paired diff) = {spearmanr(fo, x).statistic:+.3f}")
                lines.append(f"- Spearman(expo_gt1_frac of Evie run, paired diff) = {spearmanr(ex, x).statistic:+.3f}" if np.ptp(ex) > 0 else "- expo_gt1_frac constant across units")
        else:
            lines.append(f"- {arm}: fewer than 3 complete pairs ({len(d)}); nothing computed")
    hp = holm(pvals)
    lines += ["", "Holm-adjusted: " + ", ".join(f"{k}: {v:.4g}" for k, v in hp.items())]
    # analytic bound on the logged exposure distribution y = (1/w - 1) r
    qs = []
    for u in units:
        r = get("Evie", u, seeds[0])
        if r and r["status"] == "ok" and r["diagnostics"]["r_quantiles_post_calibration"]["q50"] is not None:
            qs.append(r["diagnostics"]["r_quantiles_post_calibration"])
    if qs:
        k = ["q01", "q10", "q50", "q90", "q99", "q999"]
        rq = np.array([[q[x] for x in k] for q in qs]).mean(0)
        y = (1 / 0.85 - 1) * rq
        lines += ["", "Exposure y=(1/w-1)r at r-quantiles (mean over units): " + ", ".join((f"{n}: y={v:.3g}, gap<= {7 / 8 * v * v:.3g}" if v < 1 else f"{n}: y={v:.3g} (>=1: at the floor, bound n/a)") for n, v in zip(k, y))]
    (Path(a.out)).mkdir(parents=True, exist_ok=True)
    Path(a.out, f"{cell.name}_d1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
