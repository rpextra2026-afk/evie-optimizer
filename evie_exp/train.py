"""One training run (E2 loop).  Faithful port of the legacy loop with the deliberate changes of
spec 9.7: cosine to 0.1*lr stepped after optimizer.step(), clip 5.0 (pre-clip norm recorded, >1 and >5
counted), calibration after step() at loop index calib_step for the Evie-family arms, large-batch eval,
best-val state kept on device, no Hessian probes, divergence guard, TF32 off."""
from __future__ import annotations

import math
import time
from typing import Optional

import torch

from .domains import build_task
from .domains.base import Task
from .optim.evie_family import EvieFamily, classify_param
from .optim.factory import build_optimizer
from .protocol.runlog import SCHEMA_VERSION, env_info

PRIMARY = {"vision": "test_top1", "finance": "test_mse", "language": "test_loss"}


def configure_determinism() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _test_metrics(domain: str, t: dict) -> dict:
    if domain == "vision":
        return {"test_top1": t["top1"], "test_top5": t["top5"], "test_loss": t["loss"]}
    if domain == "finance":
        return {"test_mse": t["loss"], "test_dir_acc": t.get("dir_acc", float("nan"))}
    return {"test_loss": t["loss"], "test_ppl": t["ppl"], "test_top1": t["top1"]}


def train_one(*, cell: str, domain: str, phase: str, arm: str, unit: str, seed: int, lr: float, steps: int,
              batch_size: int, eval_every: int, cfg: dict, device: torch.device, fake_data: bool = False,
              lr_source: str = "tuned", fixed_gamma: Optional[float] = None, rho_bar: Optional[float] = None,
              diag: bool = True, stop_after: Optional[int] = None) -> dict:
    """Returns a run record (schema 1).  ``stop_after`` (calibration prefix) stops after that many steps
    without evaluation and returns only ``diagnostics.calibration``."""
    configure_determinism()
    task: Task = build_task(domain, unit, seed, device, batch_size, fake_data)
    model = task.model
    params = [p for p in model.parameters() if p.requires_grad]
    ptypes = {id(p): classify_param(n, p) for n, p in model.named_parameters()}
    opt = build_optimizer(arm, params, lr, cfg, fixed_gamma=fixed_gamma, rho_bar=rho_bar, param_types=ptypes, diag=diag)
    is_family = isinstance(opt, EvieFamily)
    calib_step = cfg.get("calib_step", 125)
    do_calib = is_family and opt.mode in ("evie", "amplify")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    dev = task.device
    g_sum = torch.zeros((), device=dev, dtype=torch.float64)
    g_max = torch.zeros((), device=dev, dtype=torch.float64)
    g_gt1 = torch.zeros((), device=dev, dtype=torch.float64)
    g_gt5 = torch.zeros((), device=dev, dtype=torch.float64)
    ema = None
    n_int = 0
    curves = {"eval_step": [], "val_loss": [], "val_metric": [], "train_loss_ema": []}
    ts = {k: [] for k in ("eval_step", "gate_mean", "rho_rms", "rho_mean", "floor_frac", "cap_frac",
                          "expo_gt1_frac", "amplify_frac", "hi_cv", "psi_logstd", "psi_gate_spearman",
                          "grad_norm_mean", "grad_norm_max", "clip_gt1_frac", "clip_gt5_frac")}
    by_type: dict = {}
    r_q = None
    best_val, best_step, best_state = float("inf"), -1, None
    init_val = None
    status, err = "ok", None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
    t0 = time.time()
    model.train()
    last_val = float("nan")
    for step in range(steps):
        X, y = task.stream.next()
        opt.zero_grad()
        loss = task.loss_fn(model(X), y)
        loss.backward()
        pre = torch.nn.utils.clip_grad_norm_(params, cfg.get("grad_clip_norm", 5.0)).double()
        g_sum += pre
        g_max = torch.maximum(g_max, pre)
        g_gt1 += (pre > 1.0)
        g_gt5 += (pre > 5.0)
        n_int += 1
        opt.step()
        sched.step()
        if do_calib and step == calib_step:
            opt.calibrate_gamma0()
            if stop_after is not None:
                break
        if stop_after is not None and step + 1 >= stop_after:
            break
        ld = loss.detach()
        ema = ld if ema is None else 0.99 * ema + 0.01 * ld
        if step % eval_every == 0 or step == steps - 1:
            v = task.evaluate("val")
            val = v["loss"]
            last_val = val
            if init_val is None:
                init_val = val
            if not math.isfinite(val) or val > 10.0 * max(init_val, 1e-12) or not math.isfinite(float(ema)):
                status, err = "diverged", f"non-finite or exploding val loss at step {step}: {val}"
                break
            if val < best_val:
                best_val, best_step = val, step
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            curves["eval_step"].append(step)
            curves["val_loss"].append(val)
            curves["val_metric"].append(v["metric"])
            curves["train_loss_ema"].append(float(ema))
            ts["eval_step"].append(step)
            ts["grad_norm_mean"].append(float(g_sum) / max(n_int, 1))
            ts["grad_norm_max"].append(float(g_max))
            ts["clip_gt1_frac"].append(float(g_gt1) / max(n_int, 1))
            ts["clip_gt5_frac"].append(float(g_gt5) / max(n_int, 1))
            g_sum.zero_(); g_max.zero_(); g_gt1.zero_(); g_gt5.zero_(); n_int = 0
            if is_family and opt.diag:
                d = opt.pop_diagnostics()
                gl, het = d["global"] or {}, d.get("het", {})
                for k in ("gate_mean", "rho_rms", "rho_mean", "floor_frac", "cap_frac", "expo_gt1_frac", "amplify_frac"):
                    ts[k].append(float(gl.get(k, float("nan"))))
                for k in ("hi_cv", "psi_logstd", "psi_gate_spearman"):
                    ts[k].append(float(het.get(k, float("nan"))))
                for typ, s in d["by_type"].items():
                    bt = by_type.setdefault(typ, {"gate_mean": [], "rho_rms": []})
                    bt["gate_mean"].append(float(s["gate_mean"]))
                    bt["rho_rms"].append(float(s["rho_rms"]))
                if het.get("r"):
                    r_q = het["r"]
            else:
                for k in ("gate_mean", "rho_rms", "rho_mean", "floor_frac", "cap_frac", "expo_gt1_frac",
                          "amplify_frac", "hi_cv", "psi_logstd", "psi_gate_spearman"):
                    ts[k].append(float("nan"))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else float("nan")

    if is_family and opt.mode == "fixedgamma":
        calib = {"sigma_bar_f2": None, "gamma0_raw": None, "gamma0": fixed_gamma, "clip_status": "na", "step": None}
    elif is_family:
        calib = dict(opt.calib)
    else:
        calib = {"sigma_bar_f2": None, "gamma0_raw": None, "gamma0": None, "clip_status": "na", "step": None}
    rec = {
        "schema_version": SCHEMA_VERSION, "cell": cell, "domain": domain, "phase": phase, "arm": arm,
        "unit": unit, "seed": seed, "lr": lr, "lr_source": lr_source,
        "multiplier": (opt.const_mult if is_family and opt.mode == "const" else None),
        "fixed_gamma": fixed_gamma, "batch_size": batch_size, "steps": steps, "eval_every_steps": eval_every,
        "config": cfg, "code": env_info(str(dev)), "status": status, "error": err, "attempts": 1,
        "curves": curves,
        "diagnostics": {"calibration": calib, "timeseries": ts, "by_tensor_type": by_type,
                        "r_quantiles_post_calibration": r_q or {k: None for k in ("q01", "q10", "q50", "q90", "q99", "q999")}},
    }
    if stop_after is not None:
        rec.update({"metrics": {"primary_name": "none", "primary": float("nan"), "wall_s": wall}})
        return rec
    if status == "diverged":
        rec["metrics"] = {"primary_name": PRIMARY[domain], "primary": float("nan"), "val_loss_best": best_val,
                          "val_loss_final": last_val, "best_step": best_step, "wall_s": wall, "peak_mem_mb": peak}
        return rec
    if best_state is not None:
        model.load_state_dict(best_state)
    test = _test_metrics(domain, task.evaluate("test"))
    primary = test[PRIMARY[domain]]
    if not math.isfinite(primary):
        rec["status"], rec["error"] = "diverged", "non-finite test metric"
    rec["metrics"] = {"primary_name": PRIMARY[domain], "primary": primary, **test, "val_loss_best": best_val,
                      "val_loss_final": last_val, "best_step": best_step, "wall_s": wall, "peak_mem_mb": peak}
    return rec
