"""Synthetic stand-ins with the real shapes (pytest / --smoke).  Never downloads anything."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..protocol.seeding import make_generators
from .base import BatchStream, Task, eval_in_batches


def build_fake(domain: str, unit: str, seed: int, device: torch.device, batch_size: int) -> Task:
    gens = make_generators(seed)
    ug = torch.Generator().manual_seed(abs(hash(unit)) % (2 ** 31))  # data depends on the unit only
    torch.manual_seed(seed)
    if domain == "vision":
        mk = lambda n: (torch.randn(n, 3, 32, 32, generator=ug), torch.randint(0, 100, (n,), generator=ug))
        from .vision import LightCNN
        model, loss_fn = LightCNN(), nn.CrossEntropyLoss()
        def pb(o, y):
            return {"top1": (o.argmax(1) == y).sum(), "top5": (o.topk(5, 1).indices == y[:, None]).any(1).sum()}
        name, hib = "test_top1", True
        sizes = (512, 128, 128)
    elif domain == "finance":
        mk = lambda n: (torch.randn(n, 5, generator=ug), (0.01 * torch.randn(n, generator=ug)).unsqueeze(1))
        from .finance import FinanceMLP
        model, loss_fn = FinanceMLP(), nn.MSELoss()
        pb = lambda o, y: {"dir": ((o * y) > 0).sum()}
        name, hib = "test_mse", False
        sizes = (400, 60, 60)
    else:
        mk = lambda n: (torch.randint(0, 5000, (n, 4), generator=ug), torch.randint(0, 5000, (n,), generator=ug))
        from .language import NPLM_MLP
        model, loss_fn = NPLM_MLP(), nn.CrossEntropyLoss()
        pb = lambda o, y: {"top1": (o.argmax(-1) == y).sum()}
        name, hib = "test_loss", False
        sizes = (600, 128, 128)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = mk(sizes[0]), mk(sizes[1]), mk(sizes[2])
    model = model.to(device)
    Xtr, ytr, Xva, yva, Xte, yte = [t.to(device) for t in (Xtr, ytr, Xva, yva, Xte, yte)]
    splits = {"val": (Xva, yva), "test": (Xte, yte)}
    def evaluate(split: str):
        X, y = splits[split]
        r = eval_in_batches(model, X, y, loss_fn, 64, pb)
        r["metric"] = r["loss"]
        if domain == "vision":
            r["top1"], r["top5"] = 100.0 * r["top1"] / r["_n"], 100.0 * r["top5"] / r["_n"]
        elif domain == "language":
            r["top1"] = r["top1"] / r["_n"]
            r["ppl"] = float(torch.exp(torch.tensor(min(r["loss"], 50.0))))
        else:
            r["dir_acc"] = r["dir"] / max(r["_n"], 1)
        return r
    return Task(model, loss_fn, BatchStream(Xtr, ytr, batch_size, gens.loader), evaluate, name, hib, device)
