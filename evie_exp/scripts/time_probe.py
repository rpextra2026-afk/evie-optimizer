"""Measure it/s on THIS machine and project hours for the lean profile (7.6 step 2).
Prints the run-count formula.  Writes results/time_probe_<cell>.json."""
from __future__ import annotations

import argparse
import json
import logging
import time

import torch

from .. import cells as C
from ..domains import build_task
from ..optim.evie_family import classify_param
from ..optim.factory import build_optimizer
from ..protocol.runlog import atomic_write_json, env_info
from ..train import configure_determinism
from ._common import add_common, make_ctx

log = logging.getLogger("evie_exp")
N_LR = 11  # 7 coarse + 4 refine (+0..3 extension)


def probe_arm(cell, arm, cfg, dev, fake, steps=300):
    configure_determinism()
    task = build_task(cell.domain, "fake0" if fake else _probe_unit(cell), 1, dev, cell.batch_size, fake)
    params = list(task.model.parameters())
    pt = {id(p): classify_param(n, p) for n, p in task.model.named_parameters()}
    fg = 0.1 if arm in ("FixedGamma", "FixedGammaOwn") else None
    opt = build_optimizer(arm, params, 1e-3, cfg, fixed_gamma=fg, rho_bar=0.9, param_types=pt)
    t_ev = time.time()
    task.evaluate("val")
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    ev = time.time() - t_ev
    for w in range(20):  # warm-up
        X, y = task.stream.next()
        opt.zero_grad(); task.loss_fn(task.model(X), y).backward(); opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.time()
    for i in range(steps):
        X, y = task.stream.next()
        opt.zero_grad()
        task.loss_fn(task.model(X), y).backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if hasattr(opt, "pop_diagnostics") and (i + 1) % cell.eval_every == 0:
            opt.pop_diagnostics()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    per_step = (time.time() - t0) / steps
    return per_step, ev


def _probe_unit(cell):
    return C.eval_units(cell)[0]


def main(argv=None) -> int:
    ap = add_common(argparse.ArgumentParser())
    ap.add_argument("--gpus", type=int, default=2, help="parallel shards assumed for the projection")
    a = ap.parse_args(argv)
    cell, profile, ctx = make_ctx(a, "time_probe")
    tuned = list(profile["arms"])
    derived = list(profile.get("derived", {})) + list(profile.get("extra_arms", []))
    arms = tuned + [x for x in derived if x not in tuned]
    res, run_s = {}, {}
    steps = 40 if a.smoke else 300
    for arm in arms:
        ps, ev = probe_arm(cell, arm, ctx.cfg, ctx.device, ctx.fake_data, steps)
        n_ev = cell.steps // cell.eval_every + 1
        run_s[arm] = ps * cell.steps + ev * (n_ev + 1)
        res[arm] = {"s_per_step": ps, "eval_s": ev, "projected_run_s": run_s[arm]}
        log.info("%-14s %.2f ms/step, eval %.2fs -> projected %.1f s/run", arm, ps * 1e3, ev, run_s[arm])
    seeds_e, seeds_t = len(C.eval_seeds(cell, profile)), len(C.tune_seeds(cell, profile))
    n_units_e = 100 if cell.domain == "finance" else 10
    n_units_t = {"vision": 3, "finance": 10, "language": 3}[cell.domain]
    tshr = C.tune_steps(cell, profile) / cell.steps
    n_lr = 11
    tune_s = sum(run_s[x] for x in tuned if x != "FixedGamma") * n_lr * n_units_t * seeds_t * tshr
    eval_s = sum(run_s.values()) * n_units_e * seeds_e
    out = {"cell": cell.name, "device": str(ctx.device), "arms": res, "tune_hours_1gpu": tune_s / 3600,
           "eval_hours_1gpu": eval_s / 3600, "total_hours_1gpu": (tune_s + eval_s) / 3600,
           "total_hours_parallel": (tune_s + eval_s) / 3600 / a.gpus, "gpus_assumed": a.gpus,
           "formula": "tune_runs = n_tuned_arms*n_LR*n_tune_units*n_tune_seeds ; eval_runs = n_arms*n_units*seeds_per_unit",
           "env": env_info(str(ctx.device))}
    atomic_write_json(ctx.results / f"time_probe_{cell.name}.json", out)
    log.info("PROJECTION %s: tuning %.2f h + eval %.2f h = %.2f h on 1 GPU, ~%.2f h on %d GPUs (estimate from %d-step probe)",
             cell.name, tune_s / 3600, eval_s / 3600, (tune_s + eval_s) / 3600, out["total_hours_parallel"], a.gpus, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
