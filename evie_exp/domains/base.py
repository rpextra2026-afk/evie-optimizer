"""Domain/Task protocol.  Every domain pre-loads its splits on the device and draws batches from a
private ``torch.Generator`` (no Hessian probes, no hidden RNG use), so all arms of a (unit, seed)
see identical init and identical batch sequences (T0.11)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, Tuple

import torch
import torch.nn as nn

from ..protocol.seeding import Generators


def data_dir() -> Path:
    d = os.environ.get("EVIE_DATA")
    if d:
        return Path(d)
    return Path("/kaggle/working/data") if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else Path("./data")


class BatchStream:
    """Infinite shuffled batches (drop_last=False, like itertools.cycle(DataLoader(shuffle=True)))."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, gen: torch.Generator) -> None:
        self.X, self.y, self.bs, self.gen = X, y, batch_size, gen
        self.n = X.shape[0]
        self._perm: torch.Tensor | None = None
        self._pos = 0
        self.index_log: list[torch.Tensor] | None = None  # filled by tests

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._perm is None or self._pos >= self.n:
            self._perm = torch.randperm(self.n, generator=self.gen).to(self.X.device)
            self._pos = 0
        idx = self._perm[self._pos:self._pos + self.bs]
        self._pos += self.bs
        if self.index_log is not None:
            self.index_log.append(idx.cpu())
        return self.X[idx], self.y[idx]


@dataclass
class Task:
    model: nn.Module
    loss_fn: Callable
    stream: BatchStream
    evaluate: Callable[[str], Dict[str, float]]   # split in {"val","test"} -> metrics incl. "loss"
    primary_name: str
    primary_higher_is_better: bool
    device: torch.device
    extra: dict = field(default_factory=dict)


@torch.no_grad()
def eval_in_batches(model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss_fn: Callable, bs: int,
                    per_batch: Callable) -> Dict[str, float]:
    """Large-batch eval (D-4). ``per_batch(logits, y)`` returns a dict of additive sums."""
    model.eval()
    tot, n = 0.0, X.shape[0]
    sums: Dict[str, float] = {}
    for i in range(0, n, bs):
        xb, yb = X[i:i + bs], y[i:i + bs]
        out = model(xb)
        tot += float(loss_fn(out, yb)) * xb.shape[0]
        for k, v in per_batch(out, yb).items():
            sums[k] = sums.get(k, 0.0) + float(v)
    model.train()
    return {"loss": tot / max(n, 1), "_n": n, **sums}
