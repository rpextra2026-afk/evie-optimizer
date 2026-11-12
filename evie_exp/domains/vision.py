"""CIFAR-100 + LightCNN.  Dataset logic copied from vision/benchmark/run1_partA.py lines 99-217
(_download_cifar100, build_splits, LightCNN); fine labels only (100 classes).  Each unit is a fresh
random 80/10/10 re-split (np.random.RandomState(split_seed).permutation(60000)) with its own
train mean/std, as in the legacy code."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..protocol.seeding import make_generators
from .base import BatchStream, Task, data_dir, eval_in_batches

log = logging.getLogger("evie_exp")
EVAL_BS = 1000
_CACHE: dict = {}


class LightCNN(nn.Module):  # copy of run1_partA.py:198-217
    def __init__(self, num_classes: int = 100, width=(32, 64)) -> None:
        super().__init__()
        c1, c2 = width
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(c2 * 8 * 8, num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(F.gelu(self.conv1(x)))
        x = self.pool(F.gelu(self.conv2(x)))
        return self.fc(x.flatten(1))


def raw_path() -> Path:
    return data_dir() / "cifar100_uint8.npz"


def prepare(retries: int = 3) -> None:
    """Download once (needs internet) and cache raw uint8 images + fine labels."""
    import time

    import torchvision
    p = raw_path()
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for k in range(retries):
        try:
            root = p.parent / "cifar_tv"
            tr = torchvision.datasets.CIFAR100(root=str(root), train=True, download=True)
            te = torchvision.datasets.CIFAR100(root=str(root), train=False, download=True)
            X = np.concatenate([tr.data, te.data], 0)  # (60000,32,32,3) uint8
            y = np.array(list(tr.targets) + list(te.targets), dtype=np.int64)
            np.savez(p.with_suffix(".tmp.npz"), X=X, y=y)
            p.with_suffix(".tmp.npz").replace(p)
            return
        except Exception as e:
            last = e
            log.warning("CIFAR-100 download failed (%s); retry %d/%d", e, k + 1, retries)
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"CIFAR-100 download failed after {retries} tries: {last!r}")


def units_for(role: str, n: int) -> list:
    base = {"eval": 2024, "tune": 3001}[role]
    return [f"split{base + i}" for i in range(n)]


def _splits(split_seed: int, device: torch.device):
    key = (split_seed, str(device))
    if key in _CACHE:
        return _CACHE[key]
    if not raw_path().exists():
        raise FileNotFoundError(f"{raw_path()} missing: run `python -m evie_exp.scripts.prepare_data --domain vision`")
    d = np.load(raw_path())
    X = np.transpose(d["X"].astype(np.float32), (0, 3, 1, 2))
    y = d["y"]
    n = len(X)
    idx = np.random.RandomState(split_seed).permutation(n)
    n_tr, n_va = int(n * 0.8), int(n * 0.1)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    mu = X[tr].mean(axis=(0, 2, 3), keepdims=True)
    std = X[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-8
    def mk(ix):
        return (torch.tensor((X[ix] - mu) / std, dtype=torch.float32).to(device),
                torch.tensor(y[ix], dtype=torch.long).to(device))
    _CACHE.clear()  # keep one split resident (runs are ordered by unit)
    _CACHE[key] = (mk(tr), mk(va), mk(te))
    return _CACHE[key]


def build(unit: str, seed: int, device: torch.device, batch_size: int = 32) -> Task:
    split_seed = int(unit.replace("split", ""))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _splits(split_seed, device)
    gens = make_generators(seed)
    torch.manual_seed(seed)  # model init is a pure function of seed
    model = LightCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    splits = {"val": (Xva, yva), "test": (Xte, yte)}

    def pb(o, yb):
        return {"top1": (o.argmax(1) == yb).sum(), "top5": (o.topk(5, 1).indices == yb[:, None]).any(1).sum()}

    def evaluate(split: str):
        X, y = splits[split]
        r = eval_in_batches(model, X, y, loss_fn, EVAL_BS, pb)
        r["top1"], r["top5"] = 100.0 * r["top1"] / r["_n"], 100.0 * r["top5"] / r["_n"]
        r["metric"] = r["top1"]
        return r
    return Task(model, loss_fn, BatchStream(Xtr, ytr, batch_size, gens.loader), evaluate, "test_top1", True, device)
