"""Finance: yfinance OHLCV -> next-day log-return, chronological 80/10/10, FinanceMLP(64,32).
Copied from finance/benchmark/run1_partA.py (TICKER_POOL l.94, try_load_ticker l.124, FinanceMLP l.212).
Each ticker's arrays are cached to $EVIE_DATA/finance/{ticker}.npz so reruns do not depend on yfinance
revisions; the realised lists go to $EVIE_DATA/finance/tickers.json."""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..protocol.seeding import make_generators
from .base import BatchStream, Task, data_dir, eval_in_batches

log = logging.getLogger("evie_exp")
EVAL_BS = 4096
_CFG = Path(__file__).resolve().parents[1] / "configs" / "finance_tickers.json"


class FinanceMLP(nn.Module):  # copy of run1_partA.py:212-225
    def __init__(self, hidden=(64, 32)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(nn.Linear(5, h1), nn.GELU(), nn.Linear(h1, h2), nn.GELU(), nn.Linear(h2, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def _fdir() -> Path:
    return data_dir() / "finance"


def _download(ticker: str, retries: int = 3):
    import yfinance as yf
    last = None
    for k in range(retries):
        try:
            df = yf.download(ticker, start="2010-01-01", end="2025-01-01", progress=False, auto_adjust=True,
                             threads=False)
            if df is None or df.empty or len(df) < 500:
                raise ValueError(f"insufficient data ({0 if df is None else len(df)} rows)")
            return df
        except Exception as e:
            last = e
            time.sleep(2 * (k + 1))
    raise ValueError(str(last))


def _arrays(df):
    ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float64)
    ratio = ohlcv[1:, 3] / ohlcv[:-1, 3]
    if not np.isfinite(ratio).any():
        raise ValueError("no finite returns")
    y = np.log(np.maximum(ratio, 1e-8))
    X = ohlcv[:-1]
    n = len(X)
    if n < 300:
        raise ValueError("too few usable rows after alignment")
    n_tr, n_va = int(n * 0.8), int(n * 0.1)
    parts = (X[:n_tr], y[:n_tr], X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va], X[n_tr + n_va:], y[n_tr + n_va:])
    if min(len(parts[0]), len(parts[2]), len(parts[4])) < 20:
        raise ValueError("split too small")
    mu, std = parts[0].mean(0, keepdims=True), parts[0].std(0, keepdims=True) + 1e-8
    return ((parts[0] - mu) / std, parts[1], (parts[2] - mu) / std, parts[3], (parts[4] - mu) / std, parts[5])


def try_cache_ticker(ticker: str) -> bool:
    p = _fdir() / f"{ticker}.npz"
    if p.exists():
        return True
    try:
        a = _arrays(_download(ticker))
    except Exception as e:
        log.warning("[SKIP] %s: %s", ticker, e)
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p.with_suffix(".tmp.npz"), *a)
    p.with_suffix(".tmp.npz").replace(p)
    return True


def prepare(tickers_from: str | None = None, n_eval: int = 100, n_tune: int = 10) -> dict:
    """Cache eval + tune tickers and write tickers.json.  ``tickers_from`` = the old finance
    checkpoint JSON (its "tickers" list becomes ``eval``)."""
    cfg = json.load(open(_CFG))
    order = cfg["shuffled_order_seed1234"]
    skipped = []
    if tickers_from:
        eval_t = list(json.load(open(tickers_from))["tickers"])
        if len(eval_t) != n_eval:
            raise ValueError(f"{tickers_from} has {len(eval_t)} tickers, expected {n_eval}")
        for t in eval_t:
            if not try_cache_ticker(t):
                raise RuntimeError(f"ticker {t} from the old run no longer downloads; refusing to substitute silently")
        spare = [t for t in order if t not in eval_t]
    else:
        eval_t, spare = [], []
        for t in order:
            if len(eval_t) < n_eval:
                if try_cache_ticker(t):
                    eval_t.append(t)
                else:
                    skipped.append(t)
            else:
                spare.append(t)
        if len(eval_t) < n_eval:
            log.warning("only %d/%d valid eval tickers", len(eval_t), n_eval)
    tune_t = []
    for t in spare:
        if len(tune_t) >= n_tune:
            break
        if try_cache_ticker(t):
            tune_t.append(t)
        else:
            skipped.append(t)
    out = {"eval": eval_t, "tune": tune_t, "order_seed": cfg["order_seed"], "skipped": skipped,
           "download_date": time.strftime("%Y-%m-%d"), "from_old_checkpoint": bool(tickers_from)}
    (_fdir() / "tickers.json").write_text(json.dumps(out, indent=1))
    return out


def units_for(role: str, n: int | None = None) -> list:
    p = _fdir() / "tickers.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run prepare_data --domain finance")
    lst = json.load(open(p))[role]
    return lst if n is None else lst[:n]


_CACHE: dict = {}


def build(unit: str, seed: int, device: torch.device, batch_size: int = 32) -> Task:
    key = (unit, str(device))
    if key not in _CACHE:
        p = _fdir() / f"{unit}.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run prepare_data --domain finance")
        d = np.load(p)
        a = [d[f"arr_{i}"] for i in range(6)]
        T = lambda x, sq=False: torch.tensor(x, dtype=torch.float32).to(device)
        _CACHE.clear()
        _CACHE[key] = (T(a[0]), T(a[1]).unsqueeze(1), T(a[2]), T(a[3]).unsqueeze(1), T(a[4]), T(a[5]).unsqueeze(1))
    Xtr, ytr, Xva, yva, Xte, yte = _CACHE[key]
    gens = make_generators(seed)
    torch.manual_seed(seed)
    model = FinanceMLP().to(device)
    loss_fn = nn.MSELoss()
    splits = {"val": (Xva, yva), "test": (Xte, yte)}

    def pb(o, y):
        m = y.abs() > 1e-9
        return {"dir": ((o * y) > 0)[m].sum(), "dir_n": m.sum()}

    def evaluate(split: str):
        X, y = splits[split]
        r = eval_in_batches(model, X, y, loss_fn, EVAL_BS, pb)
        r["dir_acc"] = r["dir"] / r["dir_n"] if r.get("dir_n", 0) > 0 else float("nan")
        r["metric"] = r["loss"]
        return r
    return Task(model, loss_fn, BatchStream(Xtr, ytr, batch_size, gens.loader), evaluate, "test_mse", False, device)
