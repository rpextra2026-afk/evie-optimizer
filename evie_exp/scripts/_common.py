"""Shared CLI plumbing: results dir, resume-from-packs, deadline, graceful stop, device, context."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from .. import cells as C
from ..domains.base import data_dir
from ..engine import Ctx, parse_shard
from ..logsetup import results_dir, setup
from ..protocol import resume
from ..protocol.budget import DEFAULT_BUDGET_MIN, Deadline, GracefulStop


def add_common(ap: argparse.ArgumentParser, need_cell: bool = True) -> argparse.ArgumentParser:
    if need_cell:
        ap.add_argument("--cell", required=True, choices=list(C.CELLS))
    ap.add_argument("--profile", default="ea")
    ap.add_argument("--results", default=None, help="writable results dir (default $EVIE_RESULTS or ./results)")
    ap.add_argument("--results-in", action="append", default=[], help="extra dir/zip from an earlier session (repeatable)")
    ap.add_argument("--device", default=None, help="cuda:0 | cuda:1 | cpu (default: cuda:0 if available)")
    ap.add_argument("--shard", default="0/1", help="i/n: run the i-th of n contiguous chunks (one per GPU)")
    ap.add_argument("--time-budget-min", type=float, default=DEFAULT_BUDGET_MIN,
                    help="stop starting runs when the next one would not finish inside this budget (default 595 = 9h55)")
    ap.add_argument("--elapsed-offset-min", type=float, default=0.0, help="minutes of the session already used")
    ap.add_argument("--deadline-epoch", type=float, default=None, help="absolute deadline (used by run_all)")
    ap.add_argument("--fake-data", action="store_true", help="synthetic stand-in data (smoke/pytest)")
    ap.add_argument("--max-units", type=int, default=None, help="evaluate only the first N units (first pass)")
    ap.add_argument("--smoke", action="store_true", help="tiny steps, fake data, CPU")
    return ap


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_ctx(args, name: str):
    res = results_dir(args.results)
    log = setup(res, f"{name}_{args.shard.replace('/', 'of')}")
    if args.smoke:
        args.fake_data = True
    stats = resume.restore_from_packs(res, [Path(p) for p in args.results_in], data_dir())
    resume.log_resume(stats, res)
    cell = C.CELLS[args.cell]
    profile = C.load_profile(args.profile)
    profile['_fake'] = bool(args.fake_data)
    profile['_smoke'] = bool(args.smoke)
    cfg = C.cfg_for(cell, args.profile, profile)
    if args.smoke:
        cfg["smoke"] = True
    dev = pick_device(args.device)
    dl = Deadline(args.time_budget_min, args.deadline_epoch, args.elapsed_offset_min)
    stop = GracefulStop().install()
    ctx = Ctx(results=res, read_dirs=[Path(p) for p in args.results_in if Path(p).is_dir()], cfg=cfg, device=dev,
              deadline=dl, stop=stop, shard=parse_shard(args.shard), fake_data=args.fake_data,
              est_run_s=cell.est_run_s)
    log.info("SESSION: %s cell=%s profile=%s device=%s shard=%s budget=%.0f min results=%s fake=%s",
             name, cell.name, args.profile, dev, args.shard, args.time_budget_min, res, args.fake_data)
    return cell, profile, ctx
