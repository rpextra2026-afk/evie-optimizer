"""WikiText-2 -> NPLM-MLP next-word prediction.  load_corpus/_windows/NPLM_MLP copied from
language/benchmark/language_run1_parta_5kvocab_py.py (l.142, l.197, l.269).  Vocab 5000 (top 4999 + <unk>=0),
10 article-boundary shards, shared val/test windows, k=4.  Tokenised shards cached as .npz."""
from __future__ import annotations

import logging
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..protocol.seeding import make_generators
from .base import BatchStream, Task, data_dir, eval_in_batches

log = logging.getLogger("evie_exp")
VOCAB_SIZE, EMBED_DIM, CONTEXT_K, N_SHARDS, UNK_ID = 5000, 32, 4, 10, 0
EVAL_BS = 8192
_ART = re.compile(r"^\s=\s[^=].*[^=]\s=\s$")


class NPLM_MLP(nn.Module):  # copy of language_run1_parta_5kvocab_py.py:269-286
    def __init__(self, hidden=(64, 32), vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, k=CONTEXT_K) -> None:
        super().__init__()
        h1, h2 = hidden
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(nn.Linear(embed_dim * k, h1), nn.GELU(), nn.Linear(h1, h2), nn.GELU(),
                                 nn.Linear(h2, vocab_size))
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(self.embed(x).flatten(1))


def corpus_path() -> Path:
    return data_dir() / "language" / "corpus.npz"


def _split_into_articles(lines):
    arts, cur = [], []
    for ln in lines:
        if _ART.match(ln) and cur:
            arts.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        arts.append(cur)
    return arts


def prepare(retries: int = 3) -> None:
    p = corpus_path()
    if p.exists():
        return
    import datasets as hfd
    last = None
    for k in range(retries):
        try:
            try:
                ds = hfd.load_dataset("wikitext", "wikitext-2-raw-v1")
            except Exception:       # recent `datasets` releases require the namespaced id (same data)
                ds = hfd.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
            break
        except Exception as e:
            last = e
            log.warning("WikiText-2 download failed (%s); retry %d/%d", e, k + 1, retries)
            time.sleep(5 * (k + 1))
    else:
        raise RuntimeError(f"WikiText-2 download failed after {retries} tries: {last!r}")
    arts = _split_into_articles(list(ds["train"]["text"]))
    toks = [("".join(a)).split() for a in arts]
    cnt = Counter()
    for t in toks:
        cnt.update(t)
    vocab = ["<unk>"] + [w for w, _ in cnt.most_common(VOCAB_SIZE - 1)]
    stoi = {w: i for i, w in enumerate(vocab)}
    enc = lambda ts: np.array([stoi.get(w, 0) for w in ts], dtype=np.int32)
    order = sorted(range(len(toks)), key=lambda i: -len(toks[i]))
    buckets, lens = [[] for _ in range(N_SHARDS)], [0] * N_SHARDS
    for i in order:
        j = min(range(N_SHARDS), key=lambda k: lens[k])
        buckets[j].append(i)
        lens[j] += len(toks[i])
    out = {}
    for sid in range(N_SHARDS):
        ts = []
        for i in sorted(buckets[sid]):
            ts.extend(toks[i])
        out[f"shard_{sid:02d}"] = enc(ts)
    out["val"] = enc(("".join(ds["validation"]["text"])).split())
    out["test"] = enc(("".join(ds["test"]["text"])).split())
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p.with_suffix(".tmp.npz"), **out)
    p.with_suffix(".tmp.npz").replace(p)


def units_for(role: str, n: int | None = None) -> list:
    if role == "eval":
        return [f"shard_{i:02d}" for i in range(N_SHARDS)]
    return ["shard_00", "shard_04", "shard_09"]  # tuning: same shards, disjoint seeds (stated limitation)


def _windows(ids: np.ndarray, k: int = CONTEXT_K):
    arr = ids.astype(np.int64)
    n = len(arr) - k
    X = np.lib.stride_tricks.sliding_window_view(arr, k)[:n]
    return torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(arr[k:k + n].copy())


_CACHE: dict = {}


def build(unit: str, seed: int, device: torch.device, batch_size: int = 32) -> Task:
    key = (unit, str(device))
    if key not in _CACHE:
        if not corpus_path().exists():
            raise FileNotFoundError(f"{corpus_path()} missing: run prepare_data --domain language")
        d = np.load(corpus_path())
        shared = _CACHE.get(("shared", str(device)))
        if shared is None:
            shared = tuple(t.to(device) for t in (*_windows(d["val"]), *_windows(d["test"])))
            _CACHE[("shared", str(device))] = shared
        Xtr, ytr = [t.to(device) for t in _windows(d[unit])]
        for k in [k for k in _CACHE if k[0] != "shared"]:
            del _CACHE[k]
        _CACHE[key] = (Xtr, ytr) + shared
    Xtr, ytr, Xva, yva, Xte, yte = _CACHE[key]
    gens = make_generators(seed)
    torch.manual_seed(seed)
    model = NPLM_MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    splits = {"val": (Xva, yva), "test": (Xte, yte)}

    def pb(o, y):
        m = y != UNK_ID
        return {"top1": (o.argmax(-1) == y)[m].sum(), "top1_n": m.sum()}

    def evaluate(split: str):
        X, y = splits[split]
        r = eval_in_batches(model, X, y, loss_fn, EVAL_BS, pb)
        r["top1"] = r["top1"] / r["top1_n"] if r.get("top1_n", 0) > 0 else float("nan")
        r["ppl"] = float(np.exp(min(r["loss"], 50.0)))
        r["metric"] = r["loss"]
        return r
    return Task(model, loss_fn, BatchStream(Xtr, ytr, batch_size, gens.loader), evaluate, "test_loss", False, device)
