import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import subprocess, sys
def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
try:
    import datasets as hf_datasets
except ImportError:
    _pip("datasets")
    import datasets as hf_datasets

import torch
import torch.nn as nn
import numpy as np
import warnings, time, math, csv, json, itertools, copy, random, re
from datetime import datetime
from collections import defaultdict, Counter
from scipy.stats import wilcoxon
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# ==============================================================================
#  GLOBAL CONFIG  (per spec)
# ==============================================================================
SEEDS               = [42, 43, 44]
MAX_STEPS           = 5000
EVAL_EVERY          = 250
BATCH_SIZE_DEFAULT  = 32
MODEL_SIZE_DEFAULT  = (64, 32)
HESS_FREQ           = 5
N_HUTCH             = 4
GRAD_CLIP_NORM      = 5.0
WEIGHT_DECAY        = 1e-4
TARGET_EPS_FRAC     = 0.1   # NOTE (spec item 21): the vocab reduction to 5000
                            # lowers init loss, making this threshold MORE likely
                            # to trigger before MAX_STEPS, not less -- so the
                            # original 20k-vocab saturation risk is reduced,
                            # not worsened. Kept as spec'd -- watch the logs.
SOPHIA_BETA_H       = 0.99

LR_GRID             = [1e-4, 3e-4, 1e-3, 3e-3]
LR_SEARCH_STEPS     = 100
LR_SEARCH_SEED      = 42
N_TUNING_SHARDS     = 10   # subset of the 10 pseudo-domain shards used for LR search

CLIP_METRIC_THRESHOLD = 1.0
EPS_SCE               = 1e-8   # kept for spec fidelity (SCE denominator floor,
                                # spec item 23); SCE itself is not one of the
                                # LM metrics and is not reported here.

TARGET_WP             = 0.85
WELL_POS_MAX          = 8.0
CALIBRATE_STEP        = 125          # in [100,150] per spec
FAST_EMA_BETA         = 0.9          # separate fast EMA used ONLY for gamma0 calibration

N_SHARDS_TARGET       = 10     # pseudo-domain shards (article-boundary split of WikiText-2 train)

ACTIVATION_FLOP_FACTOR = 1.0

VOCAB_SIZE  = 5000      # reduced from 20000 for cross-domain (fin/lang/vision) runtime parity
                        # top-4999 by frequency + <unk>
EMBED_DIM   = 32        # fixed across both model sizes -> isolates the sweep to MLP hidden capacity
CONTEXT_K   = 4         # preceding tokens used to predict the next word
UNK_TOKEN   = "<unk>"
UNK_ID      = 0          # by construction, <unk> is always vocab index 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR    = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH  = os.path.join(OUT_DIR, "lm_run1b_checkpoint.json")
CSV_PATH   = os.path.join(OUT_DIR, "lm_run1b_results_raw.csv")
JSON_PATH  = os.path.join(OUT_DIR, "lm_run1b_results_agg.json")
LOG_PATH   = os.path.join(OUT_DIR, "lm_run1b_log.txt")

_log_fh = open(LOG_PATH, "a", buffering=1)
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _log_fh.write(line + "\n")
def log_sep(c="=", w=140):
    log(c * w)

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================================================================
#  DATA  (canonical WikiText-2 train/valid/test split -- spec item 9.  Vocab
#  is built from pooled TRAIN-ONLY text and shared across all shards -> no
#  leakage (spec item 8).  The canonical TRAIN split is partitioned into
#  N_SHARDS_TARGET=10 pseudo-domain shards along article boundaries (level-1
#  "= Title =" headings, spec item 26); each shard plays the role a "ticker"
#  played in the finance harness -- its own training stream, its own LR
#  search sample, its own row in the per-shard aggregation table.  The
#  canonical validation/test splits are corpus-level (not per-article), so
#  they are shared, unmodified, across every shard's evaluation.)
# ==============================================================================
_ARTICLE_HEAD_RE = re.compile(r"^\s=\s[^=].*[^=]\s=\s$")

def _split_into_articles(lines):
    articles, cur = [], []
    for ln in lines:
        if _ARTICLE_HEAD_RE.match(ln) and cur:
            articles.append(cur); cur = [ln]
        else:
            cur.append(ln)
    if cur:
        articles.append(cur)
    return articles

def _tokenize(text):
    return text.split()

_RAW = None
_SHARD_TOKENS = {}         # shard_id -> list[int] token ids (that shard's train stream)
_VAL_TOKENS = None
_TEST_TOKENS = None

def load_corpus():
    """Downloads canonical WikiText-2 (raw) once, builds the shared vocab from
    pooled TRAIN text only, splits TRAIN into N_SHARDS_TARGET pseudo-domain
    shards along article boundaries, and tokenizes the canonical valid/test
    splits (shared across all shards)."""
    global _RAW, _VAL_TOKENS, _TEST_TOKENS
    if _RAW is not None:
        return
    log("[DATA] Downloading canonical WikiText-2 (train/valid/test splits)...")
    ds = hf_datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    train_lines = list(ds["train"]["text"])
    valid_lines = list(ds["validation"]["text"])
    test_lines  = list(ds["test"]["text"])

    train_articles = _split_into_articles(train_lines)
    log(f"[DATA] train split has {len(train_articles)} articles.")

    # ---- vocab: pooled TRAIN-ONLY text, top-(VOCAB_SIZE-1) by frequency + <unk> ----
    counter = Counter()
    art_tok_lists = [_tokenize("".join(a)) for a in train_articles]
    for toks in art_tok_lists:
        counter.update(toks)
    most_common = [w for w, _ in counter.most_common(VOCAB_SIZE - 1)]
    vocab = [UNK_TOKEN] + most_common
    stoi = {w: i for i, w in enumerate(vocab)}
    unk_id = stoi[UNK_TOKEN]
    assert unk_id == UNK_ID

    def encode(tokens):
        return [stoi.get(w, unk_id) for w in tokens]

    # ---- split TRAIN articles into N_SHARDS_TARGET shards by article
    # boundary, balanced by token count (greedy: always add the next-largest
    # remaining article to the currently-smallest shard) so shards stay close
    # in length without ever splitting an article across shards. ----
    order = sorted(range(len(art_tok_lists)), key=lambda i: -len(art_tok_lists[i]))
    shard_buckets = [[] for _ in range(N_SHARDS_TARGET)]
    shard_lens = [0] * N_SHARDS_TARGET
    for idx in order:
        j = min(range(N_SHARDS_TARGET), key=lambda k: shard_lens[k])
        shard_buckets[j].append(idx)
        shard_lens[j] += len(art_tok_lists[idx])

    for sid in range(N_SHARDS_TARGET):
        toks = []
        for idx in sorted(shard_buckets[sid]):
            toks.extend(art_tok_lists[idx])
        _SHARD_TOKENS[f"shard_{sid:02d}"] = encode(toks)

    _VAL_TOKENS = encode(_tokenize("".join(valid_lines)))
    _TEST_TOKENS = encode(_tokenize("".join(test_lines)))
    _RAW = True
    log(f"[DATA] vocab size={len(vocab)} (top-{VOCAB_SIZE-1}+<unk>), "
        f"{N_SHARDS_TARGET} shards, val_tokens={len(_VAL_TOKENS)}, test_tokens={len(_TEST_TOKENS)}")

def _windows(token_ids, k=CONTEXT_K):
    """(context, target) pairs: k preceding tokens -> next token."""
    if len(token_ids) <= k:
        return np.zeros((0, k), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    arr = np.asarray(token_ids, dtype=np.int64)
    n = len(arr) - k
    X = np.stack([arr[i:i + k] for i in range(n)])
    y = arr[k:k + n]
    return X, y

_SHARD_WINDOWS = {}
_VAL_WINDOWS = None
_TEST_WINDOWS = None

def try_load_shard(shard_id):
    """Analogous to try_load_ticker in the finance harness: builds this
    shard's windowed train tensors (+ the shared val/test windows, computed
    once) on first access. Returns True if the shard has enough windows to
    train on, False if it should be skipped."""
    load_corpus()
    global _VAL_WINDOWS, _TEST_WINDOWS
    if _VAL_WINDOWS is None:
        _VAL_WINDOWS = _windows(_VAL_TOKENS)
        _TEST_WINDOWS = _windows(_TEST_TOKENS)
    if shard_id in _SHARD_WINDOWS:
        return True
    if shard_id not in _SHARD_TOKENS:
        return False
    Xtr, ytr = _windows(_SHARD_TOKENS[shard_id])
    if len(Xtr) < 50:
        log(f"    [SKIP] {shard_id}: insufficient windows ({len(Xtr)})")
        return False
    _SHARD_WINDOWS[shard_id] = (Xtr, ytr)
    return True

def get_loaders(shard_id, batch_size):
    Xtr, ytr = _SHARD_WINDOWS[shard_id]
    Xva, yva = _VAL_WINDOWS
    Xte, yte = _TEST_WINDOWS
    def ds(Xd, yd):
        return TensorDataset(torch.tensor(Xd, dtype=torch.long),
                              torch.tensor(yd, dtype=torch.long))
    return (DataLoader(ds(Xtr, ytr), batch_size=batch_size, shuffle=True),
            DataLoader(ds(Xva, yva), batch_size=batch_size),
            DataLoader(ds(Xte, yte), batch_size=batch_size))

def select_shards(n_target=N_SHARDS_TARGET, seed=0):
    """All N_SHARDS_TARGET pseudo-domain shards. Unlike the finance ticker
    draw this is a deterministic article-boundary partition, not a random
    sample -- every shard is used. `seed` is kept only for call-signature
    parity with the finance harness's select_tickers()."""
    load_corpus()
    shard_ids = sorted(_SHARD_TOKENS.keys())
    chosen = [s for s in shard_ids if try_load_shard(s)]
    if len(chosen) < n_target:
        log(f"[WARN] Only {len(chosen)}/{n_target} shards had enough windows.")
    return chosen[:n_target]

def stratified_tuning_subset(shards, n=N_TUNING_SHARDS):
    """Evenly-spaced sample of shards for LR search (no sector structure in
    the LM setting, unlike the finance ticker pool, so we just spread the
    sample evenly across the shard list)."""
    if len(shards) <= n:
        return list(shards)
    step = len(shards) / n
    idxs = sorted(set(int(i * step) for i in range(n)))
    return [shards[i] for i in idxs][:n]

# ==============================================================================
#  MODEL  (NPLM-style feedforward MLP: embed last k=4 tokens, concat, 2-layer
#  MLP, project to vocab -> softmax/CE next-word prediction)
# ==============================================================================
class NPLM_MLP(nn.Module):
    def __init__(self, hidden=(64, 32), vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, k=CONTEXT_K):
        super().__init__()
        h1, h2 = hidden
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * k, h1), nn.GELU(),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, vocab_size))
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):          # x: (B, k) long token ids
        e = self.embed(x).flatten(1)     # (B, k*embed_dim)
        return self.net(e)               # (B, vocab_size) logits

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_fwd_flops(model, batch_size):
    """Forward-FLOP estimator rewritten for the LM model (spec item 24):
    embedding lookup is a gather (no FLOPs), so the cost is the two MLP
    hidden layers plus the final vocab-projection Linear(h2, vocab_size) --
    which dominates since vocab_size (20k) >> h1, h2."""
    flops = 0
    for m in model.net.modules():
        if isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features * batch_size
    return int(flops * ACTIVATION_FLOP_FACTOR)

# ==============================================================================
#  OPTIMIZERS
# ==============================================================================
def _spatial_avg(h):
    if h.dim() == 4:
        return h.mean(dim=[2,3], keepdim=True).expand_as(h).clone()
    elif h.dim() == 2:
        return h.mean(dim=1, keepdim=True).expand_as(h).clone()
    elif h.dim() == 1:
        return h.mean().expand_as(h).clone()
    return h.clone()


class RSCOOptimizer(torch.optim.Optimizer):
    """Base RSCO. gamma0 is auto-calibrated ONCE, at a step in [100,150],
    using a SEPARATE fast-EMA variance estimate (beta=FAST_EMA_BETA) that is
    tracked independently of the optimizer's own (slow, beta2=0.999) m1/m2
    state used in the actual update rule. Also tracks the fraction of
    coordinates whose well-positioning factor hits its clamp floor each step
    (`last_clip_coord_frac`) -- the LM harness's `clipcoord%` metric."""
    def __init__(self, params, lr=1e-3, gamma0=1.0, target_wp=TARGET_WP,
                 beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                 fast_beta=FAST_EMA_BETA):
        defaults = dict(lr=lr, gamma0=gamma0, target_wp=target_wp,
                         beta1=beta1, beta2=beta2, eps=eps,
                         weight_decay=weight_decay, calibrated=False,
                         fast_beta=fast_beta)
        super().__init__(params, defaults)
        self.last_clip_coord_frac = float('nan')

    def calibrate_gamma0(self):
        for group in self.param_groups:
            if group['calibrated']:
                continue
            target = group['target_wp']
            sigma2_list = []
            for p in group['params']:
                s = self.state.get(p, {})
                if 'fast_m1' not in s:
                    continue
                sigma2 = (s['fast_m2'] - s['fast_m1'].pow(2)).clamp_(min=0.0)
                sigma2_list.append(sigma2.mean().item())
            if not sigma2_list:
                continue
            mean_s2 = max(float(np.mean(sigma2_list)), 1e-12)
            new_g0 = float(np.clip((1.0 / target - 1.0) / mean_s2, 10.0, 5000.0))
            group['gamma0'] = new_g0
            group['calibrated'] = True
            log(f"    [RSCO] gamma0={new_g0:.1f}  fast_sigma2={mean_s2:.2e}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        clipped_el, total_el = 0, 0
        for group in self.param_groups:
            lr, g0, b1, b2, eps, wd = (group['lr'], group['gamma0'], group['beta1'],
                                        group['beta2'], group['eps'], group['weight_decay'])
            fb = group['fast_beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                grad = p.grad.data
                s = self.state[p]
                if 'step' not in s:
                    s['step'] = 0
                    s['m1'] = torch.zeros_like(p.data)
                    s['m2'] = torch.zeros_like(p.data)
                    s['fast_m1'] = torch.zeros_like(p.data)
                    s['fast_m2'] = torch.zeros_like(p.data)
                s['step'] += 1
                t = s['step']
                # main (slow) EMA used for the actual update
                s['m1'].mul_(b1).add_(grad, alpha=1.0 - b1)
                s['m2'].mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
                # SEPARATE fast EMA used ONLY for gamma0 calibration
                s['fast_m1'].mul_(fb).add_(grad, alpha=1.0 - fb)
                s['fast_m2'].mul_(fb).addcmul_(grad, grad, value=1.0 - fb)

                m1c = s['m1'] / (1.0 - b1 ** t)
                m2c = s['m2'] / (1.0 - b2 ** t)
                sigma2 = (m2c - m1c.pow(2)).clamp_(min=0.0)
                gamma_t = g0 / (1.0 + g0 * sigma2)
                raw_wp = (1.0 - gamma_t * sigma2)
                well_pos = raw_wp.clamp(min=1e-6)
                clipped_el += int((raw_wp < 1e-6).sum().item())
                total_el += raw_wp.numel()
                denom = m2c.sqrt().clamp_(min=eps)
                p.data.addcdiv_(m1c.mul(well_pos.sqrt()), denom, value=-lr)
        self.last_clip_coord_frac = (clipped_el / total_el) if total_el > 0 else float('nan')
        return loss


class RSCOAmplifyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, gamma0=1.0, target_wp=TARGET_WP,
                 beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                 well_pos_max=WELL_POS_MAX, fast_beta=FAST_EMA_BETA):
        defaults = dict(lr=lr, gamma0=gamma0, target_wp=target_wp,
                         beta1=beta1, beta2=beta2, eps=eps,
                         weight_decay=weight_decay, calibrated=False,
                         well_pos_max=well_pos_max, fast_beta=fast_beta)
        super().__init__(params, defaults)
        self.last_clip_coord_frac = float('nan')

    def calibrate_gamma0(self):
        for group in self.param_groups:
            if group['calibrated']:
                continue
            target = group['target_wp']
            sigma2_list = []
            for p in group['params']:
                s = self.state.get(p, {})
                if 'fast_m1' not in s:
                    continue
                sigma2 = (s['fast_m2'] - s['fast_m1'].pow(2)).clamp_(min=0.0)
                sigma2_list.append(sigma2.mean().item())
            if not sigma2_list:
                continue
            mean_s2 = max(float(np.mean(sigma2_list)), 1e-12)
            new_g0 = float(np.clip((1.0 / target - 1.0) / mean_s2, 10.0, 5000.0))
            group['gamma0'] = new_g0
            group['calibrated'] = True
            log(f"    [RSCO-Amplify] gamma0={new_g0:.1f}  fast_sigma2={mean_s2:.2e}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        clipped_el, total_el = 0, 0
        for group in self.param_groups:
            lr, g0, b1, b2 = group['lr'], group['gamma0'], group['beta1'], group['beta2']
            eps, wd, target = group['eps'], group['weight_decay'], group['target_wp']
            wp_max = group['well_pos_max']
            fb = group['fast_beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                grad = p.grad.data
                s = self.state[p]
                if 'step' not in s:
                    s['step'] = 0
                    s['m1'] = torch.zeros_like(p.data)
                    s['m2'] = torch.zeros_like(p.data)
                    s['fast_m1'] = torch.zeros_like(p.data)
                    s['fast_m2'] = torch.zeros_like(p.data)
                s['step'] += 1
                t = s['step']
                s['m1'].mul_(b1).add_(grad, alpha=1.0 - b1)
                s['m2'].mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
                s['fast_m1'].mul_(fb).add_(grad, alpha=1.0 - fb)
                s['fast_m2'].mul_(fb).addcmul_(grad, grad, value=1.0 - fb)

                m1c = s['m1'] / (1.0 - b1 ** t)
                m2c = s['m2'] / (1.0 - b2 ** t)
                sigma2 = (m2c - m1c.pow(2)).clamp_(min=0.0)
                gamma_t = g0 / (1.0 + g0 * sigma2)
                raw_wp = (1.0 / (1.0 + gamma_t * sigma2)) / target
                well_pos = raw_wp.clamp(min=1e-6, max=wp_max)
                clipped_el += int(((raw_wp < 1e-6) | (raw_wp > wp_max)).sum().item())
                total_el += raw_wp.numel()
                denom = m2c.sqrt().clamp_(min=eps)
                p.data.addcdiv_(m1c.mul(well_pos.sqrt()), denom, value=-lr)
        self.last_clip_coord_frac = (clipped_el / total_el) if total_el > 0 else float('nan')
        return loss


class SophiaOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], rho=rho,
                         eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.last_h_norm = float('nan')

    def set_hessian(self, param_h_pairs, beta_h=SOPHIA_BETA_H):
        sq_sum = 0.0
        for p, h in param_h_pairs:
            s = self.state[p]
            h_new = h.detach().abs().clamp_(min=1e-8)
            if 'h_hat' not in s:
                s['h_hat'] = h_new.clone()
            else:
                s['h_hat'].mul_(beta_h).add_(h_new, alpha=1.0 - beta_h)
            sq_sum += h_new.pow(2).sum().item()
        self.last_h_norm = math.sqrt(sq_sum)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, b1, b2, rho, eps, wd = (group['lr'], group['beta1'], group['beta2'],
                                         group['rho'], group['eps'], group['weight_decay'])
            for p in group['params']:
                if p.grad is None:
                    continue
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                grad = p.grad.data
                s = self.state[p]
                if 'step' not in s:
                    s['step'] = 0
                    s['m'] = torch.zeros_like(p.data)
                    s['h_hat'] = torch.ones_like(p.data)
                s['step'] += 1
                t = s['step']
                s['m'].mul_(b1).add_(grad, alpha=1.0 - b1)
                m = s['m'] / (1.0 - b1 ** t)
                h = s['h_hat'].clamp_(min=eps)
                update = (m / (rho * h)).clamp_(-1.0, 1.0)
                p.data.add_(update, alpha=-lr)
        return loss


class AdaHessianOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta_h=0.99,
                 eps=1e-4, weight_decay=0.0, hessian_power=1.0):
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], beta_h=beta_h,
                         eps=eps, weight_decay=weight_decay, hessian_power=hessian_power)
        super().__init__(params, defaults)
        self.last_h_norm = float('nan')

    def set_hessian(self, param_h_pairs, beta_h=0.99):
        sq_sum = 0.0
        for p, h in param_h_pairs:
            if p not in self.state:
                self.state[p] = {}
            s = self.state[p]
            h_smooth = _spatial_avg(h.detach().abs()).clamp(min=1e-8)
            if 'h_hat' not in s:
                s['h_hat'] = h_smooth.clone()
            else:
                s['h_hat'].mul_(beta_h).add_(h_smooth, alpha=1.0 - beta_h)
            sq_sum += h_smooth.pow(2).sum().item()
        self.last_h_norm = math.sqrt(sq_sum)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, b1, b2, eps, wd, k = (group['lr'], group['beta1'], group['beta2'],
                                       group['eps'], group['weight_decay'], group['hessian_power'])
            for p in group['params']:
                if p.grad is None:
                    continue
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                grad = p.grad.data
                s = self.state[p]
                if 'step' not in s:
                    s['step'] = 0
                    s['m'] = torch.zeros_like(p.data)
                    s['v'] = torch.zeros_like(p.data)
                    s['h_hat'] = torch.ones_like(p.data)
                s['step'] += 1
                t = s['step']
                s['m'].mul_(b1).add_(grad, alpha=1.0 - b1)
                m_hat = s['m'] / (1.0 - b1 ** t)
                d = s['h_hat']
                s['v'].mul_(b2).addcmul_(d, d, value=1.0 - b2)
                v_hat = s['v'] / (1.0 - b2 ** t)
                denom = (v_hat.sqrt().pow(k)).clamp_(min=eps)
                p.data.addcdiv_(m_hat, denom, value=-lr)
        return loss


def build_optimizer(opt_name, model, lr):
    if opt_name == "RSCO":
        return RSCOOptimizer(model.parameters(), lr=lr, gamma0=1.0,
                              target_wp=TARGET_WP, weight_decay=WEIGHT_DECAY)
    elif opt_name == "RSCO-Amplify":
        return RSCOAmplifyOptimizer(model.parameters(), lr=lr, gamma0=1.0,
                                     target_wp=TARGET_WP, weight_decay=WEIGHT_DECAY,
                                     well_pos_max=WELL_POS_MAX)
    elif opt_name == "Sophia":
        return SophiaOptimizer(model.parameters(), lr=lr, betas=(0.965, 0.99),
                                rho=0.04, weight_decay=WEIGHT_DECAY)
    elif opt_name == "AdaHessian":
        return AdaHessianOptimizer(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                    eps=1e-4, weight_decay=WEIGHT_DECAY, hessian_power=1.0)
    else:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

OPT_NAMES = ["AdamW", "RSCO-Amplify"]  # PART B of 3 (parallel split)
HESSIAN_CONSUMERS = {"Sophia", "AdaHessian"}

# ==============================================================================
#  HESSIAN ESTIMATORS  (Hutchinson kept for Sophia & AdaHessian; run PASSIVELY
#  for all 5 optimizers every HESS_FREQ steps so hess_norm/hess_cv are
#  computed for every optimizer, per spec)
# ==============================================================================
def hutchinson_hessian(model, params, X, y, criterion, n_hutch=N_HUTCH,
                        optimizer=None, feed_optimizer=False, apply_spatial_avg=False):
    with torch.enable_grad():
        loss_h = criterion(model(X), y)
    grads = torch.autograd.grad(loss_h, params, create_graph=True,
                                 retain_graph=True, allow_unused=True)
    h_acc = [torch.zeros_like(p.data) for p in params]
    for k in range(n_hutch):
        vs = [(torch.randint(0, 2, p.shape, device=p.device, dtype=torch.float32) * 2 - 1).to(p.dtype)
              for p in params]
        gv = sum((g * v).sum() for g, v in zip(grads, vs) if g is not None)
        retain = (k < n_hutch - 1)
        hvps = torch.autograd.grad(gv, params, retain_graph=retain, allow_unused=True)
        for i, (v, hv) in enumerate(zip(vs, hvps)):
            if hv is not None:
                h_acc[i].add_(v * hv.detach())
        del vs, gv, hvps
    del grads
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for h in h_acc:
        h.div_(n_hutch); h.abs_()
    if apply_spatial_avg:
        h_acc = [_spatial_avg(h) for h in h_acc]
    if feed_optimizer and optimizer is not None:
        optimizer.set_hessian(list(zip(params, h_acc)), beta_h=SOPHIA_BETA_H)
    return h_acc

# NOTE: GNB estimator removed. Sophia now uses the SAME Hutchinson diagonal
# estimator as AdaHessian (and as the passive probe run for AdamW/RSCO/
# RSCO-Amplify) -- one Hessian estimator (Hutchinson) for all 5 optimizers.

# ==============================================================================
#  MEMORY / EVAL
# ==============================================================================
def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    try:
        proc = __import__('psutil').Process(os.getpid())
        return proc.memory_info().rss / (1024 ** 2)
    except Exception:
        return float('nan')

@torch.no_grad()
def evaluate(model, loader, criterion):
    """Returns (avg_ce_loss, top1_next_token_accuracy). Accuracy masks out
    targets equal to <unk> (spec item 20 -- the language analog of the
    |y|<=1e-9 mask used for dir_acc in the finance harness); there is no pad
    token here since every window has a full k=4 context."""
    model.eval()
    tot_loss, tot_n = 0.0, 0
    correct, correct_n = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        tot_loss += loss.item() * X.size(0)
        tot_n += X.size(0)
        mask = (y != UNK_ID)
        if mask.any():
            pred = logits.argmax(dim=-1)
            correct += int((pred[mask] == y[mask]).sum().item())
            correct_n += int(mask.sum().item())
    avg_loss = tot_loss / max(tot_n, 1)
    top1acc = (correct / correct_n) if correct_n > 0 else float('nan')
    return avg_loss, top1acc

# ==============================================================================
#  LR GRID SEARCH  -- evenly-spread tuning subset, ALL 5 optimizers, frozen after
# ==============================================================================
def _lr_grid_search_unsafe(opt_name, tuning_shards, batch_size, hidden):
    val_by_lr = defaultdict(list)
    for lr in LR_GRID:
        for shard in tuning_shards:
            set_seed(LR_SEARCH_SEED)
            train_loader, val_loader, _ = get_loaders(shard, batch_size)
            model = NPLM_MLP(hidden).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = build_optimizer(opt_name, model, lr)
            params = [p for p in model.parameters() if p.requires_grad]
            loader_iter = itertools.cycle(train_loader)
            for step in range(LR_SEARCH_STEPS):
                model.train()
                X_batch, y_batch = next(loader_iter)
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                if step % HESS_FREQ == 0 and opt_name in HESSIAN_CONSUMERS:
                    # Sophia and AdaHessian both consume the SAME Hutchinson
                    # diagonal estimator (no GNB -- Hutchinson only, per spec).
                    hutchinson_hessian(model, params, X_batch, y_batch, criterion,
                                        optimizer=optimizer, feed_optimizer=True)
                optimizer.step()
            val_loss, _ = evaluate(model, val_loader, criterion)
            if not math.isnan(val_loss):
                val_by_lr[lr].append(val_loss)
    median_by_lr = {lr: (float(np.median(v)) if v else float('inf'))
                     for lr, v in val_by_lr.items()}
    best_lr = min(median_by_lr, key=median_by_lr.get)
    log(f"  [LR-SEARCH] {opt_name}: medians={ {f'{k:.0e}': round(v,6) for k,v in median_by_lr.items()} } "
        f"-> selected lr={best_lr:.0e}")
    return best_lr

# ==============================================================================
#  SINGLE EXPERIMENT (one optimizer, one shard, one seed)
# ==============================================================================
def _run_experiment_unsafe(opt_name, lr, shard, seed, batch_size=BATCH_SIZE_DEFAULT,
                    hidden=MODEL_SIZE_DEFAULT, max_steps=MAX_STEPS):
    set_seed(seed)
    reset_peak_memory()

    train_loader, val_loader, test_loader = get_loaders(shard, batch_size)
    model = NPLM_MLP(hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    n_params = count_params(model)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = build_optimizer(opt_name, model, lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=lr * 0.1)

    X0, y0 = next(iter(train_loader))
    with torch.no_grad():
        init_loss = criterion(model(X0.to(device)), y0.to(device)).item()

    step_sizes = []
    steps_to_half = steps_to_eps = None
    clip_trigger_count = 0
    clip_coord_fracs = []
    hess_norms = []
    cum_regret = 0.0
    final_train_loss = init_loss
    cumulative_flops = 0.0
    fwd_flops_per_batch = estimate_fwd_flops(model, batch_size)
    is_hessian_consumer = opt_name in HESSIAN_CONSUMERS
    is_rsco_family = opt_name in ("RSCO", "RSCO-Amplify")

    loader_iter = itertools.cycle(train_loader)
    best_val = float('inf'); best_state = None
    best_val_so_far = float('inf')
    calibrated_once = False
    global_step = 0

    t0 = time.time()
    model.train()

    for step in range(max_steps):
        global_step = step + 1
        X_batch, y_batch = next(loader_iter)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        with torch.no_grad():
            params_before = [p.data.clone() for p in model.parameters()]

        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()

        pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        if float(pre_clip_norm) > CLIP_METRIC_THRESHOLD:
            clip_trigger_count += 1

        ran_probe = (step % HESS_FREQ == 0)
        h_list = None
        if ran_probe:
            # Single Hessian estimator (Hutchinson diagonal) for ALL 5
            # optimizers. Sophia and AdaHessian CONSUME it into their update
            # rule (feed_optimizer=True); AdamW/RSCO/RSCO-Amplify only get it
            # PASSIVELY (feed_optimizer=False) so hess_norm/hess_cv are still
            # computed for every optimizer, per spec.
            h_list = hutchinson_hessian(model, params, X_batch, y_batch, criterion,
                                         N_HUTCH, optimizer=optimizer,
                                         feed_optimizer=is_hessian_consumer,
                                         apply_spatial_avg=False)
            if is_hessian_consumer and not math.isnan(optimizer.last_h_norm):
                hess_norms.append(optimizer.last_h_norm)
            else:
                hess_norms.append(math.sqrt(sum(h.pow(2).sum().item() for h in h_list)))

        step_flops = fwd_flops_per_batch * 3
        if ran_probe:
            step_flops += N_HUTCH * fwd_flops_per_batch * 3
        cumulative_flops += step_flops

        optimizer.step()
        scheduler.step()

        cc = getattr(optimizer, 'last_clip_coord_frac', float('nan'))
        if not (isinstance(cc, float) and math.isnan(cc)):
            clip_coord_fracs.append(cc)

        if is_rsco_family and not calibrated_once and step == CALIBRATE_STEP:
            optimizer.calibrate_gamma0()
            calibrated_once = True

        lv = loss.item()

        with torch.no_grad():
            delta = torch.cat([(p.data - pb).flatten()
                                for p, pb in zip(model.parameters(), params_before)])
        del params_before

        delta_sq = delta.pow(2).sum().item()
        step_sizes.append(math.sqrt(delta_sq / max(n_params, 1)))   # RMS step size

        if steps_to_half is None and lv <= 0.5 * init_loss:
            steps_to_half = step
        if steps_to_eps is None and lv <= TARGET_EPS_FRAC * init_loss:
            steps_to_eps = step
        final_train_loss = lv

        if step % EVAL_EVERY == 0:
            val_loss, _ = evaluate(model, val_loader, criterion)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
            best_val_so_far = min(best_val_so_far, val_loss)
            cum_regret += max(val_loss - best_val_so_far, 0.0)
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    # `valloss`/`ppl`/`top1acc` (spec item 20) are the FINAL evaluation of the
    # best-checkpointed model on the held-out TEST split (consistent with the
    # finance and vision harnesses which both evaluate on the test loader here).
    test_loss_final, top1acc_final = evaluate(model, test_loader, criterion)

    wall = time.time() - t0
    mem = peak_memory_mb()
    mean_step_size = float(np.mean(step_sizes)) if step_sizes else float('nan')
    gflops = cumulative_flops / max(global_step, 1) / 1e9   # per-step GFLOPs
    exaflops = cumulative_flops / 1e18
    clip_trigger_freq = clip_trigger_count / max(max_steps, 1)
    clip_coord_freq = float(np.mean(clip_coord_fracs)) if clip_coord_fracs else float('nan')

    if hess_norms:
        hess_diag_norm = float(np.mean(hess_norms))
        hn_std = float(np.std(hess_norms))
        hess_diag_cv = hn_std / hess_diag_norm if hess_diag_norm > 0 else float('nan')
    else:
        hess_diag_norm = hess_diag_cv = float('nan')

    # ppl is computed over the reduced VOCAB_SIZE=5000 output space; valid for
    # within-run optimizer comparisons (e.g. RSCO vs AdamW, same vocab throughout)
    # but NOT numerically comparable to published WikiText-2 perplexities, which
    # conventionally use ~20k-33k vocab.
    ppl = math.exp(min(test_loss_final, 50.0)) if not math.isnan(test_loss_final) else float('nan')

    return {
        'valloss': test_loss_final, 'ppl': ppl, 'top1acc': top1acc_final,
        'steps50%': steps_to_half if steps_to_half is not None else max_steps,
        'stepseps': steps_to_eps if steps_to_eps is not None else max_steps,
        'wall(s)': wall, 'peakmemmb': mem,
        'gflops': gflops, 'exaflops': exaflops,
        'stepsize': mean_step_size,
        'clipfreq%': clip_trigger_freq,
        'clipcoord%': clip_coord_freq,
        'hessnorm': hess_diag_norm, 'hesscv': hess_diag_cv,
        'regret': cum_regret,
        'n_params': n_params, 'lr_used': lr,
    }

METRIC_KEYS = ['valloss','ppl','top1acc','steps50%','stepseps','wall(s)',
               'peakmemmb','gflops','exaflops','stepsize','clipfreq%',
               'clipcoord%','hessnorm','hesscv','regret']
# direction: True = lower is better
LOWER_BETTER = {
    'valloss': True, 'ppl': True, 'top1acc': False, 'steps50%': True,
    'stepseps': True, 'wall(s)': True, 'peakmemmb': True, 'gflops': True,
    'exaflops': True, 'stepsize': None, 'clipfreq%': True,
    'clipcoord%': None, 'hessnorm': None, 'hesscv': True, 'regret': True,
}

# ==============================================================================
#  CHECKPOINTING
# ==============================================================================
def load_checkpoint():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {"shards": None, "lr_used": {}, "completed": {}, "raw": {}}

def save_checkpoint(ckpt):
    tmp = CKPT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    os.replace(tmp, CKPT_PATH)

# ==============================================================================
#  AGGREGATION / STATS
# ==============================================================================
def aggregate_shard(seed_results):
    agg = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in seed_results if not (isinstance(r[k], float) and math.isnan(r[k]))]
        agg[k] = float(np.median(vals)) if vals else float('nan')
    return agg

def holm_bonferroni(pvals_dict):
    items = [(k, p) for k, p in pvals_dict.items() if not math.isnan(p)]
    items_sorted = sorted(items, key=lambda x: x[1])
    m = len(items_sorted)
    corrected = {k: float('nan') for k in pvals_dict}
    for rank0, (k, p) in enumerate(items_sorted):
        corrected[k] = min(p * (m - rank0), 1.0)
    return corrected

def cross_shard_score(agg_by_shard, opt_names, shards):
    """'genscore%' column = cross-shard generalization score:
    (1 - mean_valloss / worst_optimizer_mean_valloss) * 100"""
    means = {}
    for opt in opt_names:
        vals = [agg_by_shard[opt][s]['valloss'] for s in shards
                 if not math.isnan(agg_by_shard[opt][s]['valloss'])]
        means[opt] = float(np.mean(vals)) if vals else float('nan')
    worst = max((v for v in means.values() if not math.isnan(v)), default=1.0)
    return {opt: ((1.0 - m / worst) * 100.0 if (not math.isnan(m) and worst > 0) else float('nan'))
            for opt, m in means.items()}

# ==============================================================================
#  FINAL SINGLE TABLE  (per spec: rows=optimizer, all metrics, mean+-std,
#  winner column, RSCO-Amplify vs AdamW %delta + p_holm)
# ==============================================================================
def build_final_table(agg_by_shard, opt_names, shards):
    challenger = "RSCO-Amplify"
    genscore = cross_shard_score(agg_by_shard, opt_names, shards)

    # per-optimizer mean+-std across shards (of the per-shard seed-medians)
    stats = {}
    for opt in opt_names:
        stats[opt] = {}
        for k in METRIC_KEYS:
            vals = [agg_by_shard[opt][s][k] for s in shards
                     if not math.isnan(agg_by_shard[opt][s][k])]
            stats[opt][k] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float('nan'), float('nan'))
        stats[opt]['genscore%'] = (genscore[opt], 0.0)

    all_keys = METRIC_KEYS + ['genscore%']
    dir_map = dict(LOWER_BETTER); dir_map['genscore%'] = False

    # winner per metric (direction-aware; None-direction metrics -> no winner claim)
    winners = {}
    for k in all_keys:
        lb = dir_map.get(k)
        candidates = {opt: stats[opt][k][0] for opt in opt_names if not math.isnan(stats[opt][k][0])}
        if not candidates or lb is None:
            winners[k] = "n/a"
            continue
        winners[k] = min(candidates, key=candidates.get) if lb else max(candidates, key=candidates.get)

    # RSCO-Amplify vs AdamW: per-metric Wilcoxon on paired per-shard medians + Holm-Bonferroni
    # Only computable when this run's opt_names includes both sides of the comparison
    # (a run split across parallel scripts may only contain one of the two).
    raw_p = {}
    delta_pct = {}
    can_compare = (challenger in opt_names) and ("AdamW" in opt_names)
    for k in METRIC_KEYS:
        if not can_compare:
            raw_p[k] = float('nan')
            delta_pct[k] = float('nan')
            continue
        c_vals, b_vals = [], []
        for s in shards:
            c = agg_by_shard[challenger][s][k]
            b = agg_by_shard["AdamW"][s][k]
            if not (math.isnan(c) or math.isnan(b)):
                c_vals.append(c); b_vals.append(b)
        if len(c_vals) >= 10:
            try:
                _, p = wilcoxon(c_vals, b_vals)
            except ValueError:
                p = float('nan')
        else:
            p = float('nan')
        raw_p[k] = p
        lb = dir_map.get(k)
        if b_vals and c_vals:
            b_med, c_med = float(np.median(b_vals)), float(np.median(c_vals))
            if b_med != 0:
                rel = (b_med - c_med) / abs(b_med) * 100.0
                delta_pct[k] = rel if lb else -rel
            else:
                delta_pct[k] = float('nan')
        else:
            delta_pct[k] = float('nan')
    corrected_p = holm_bonferroni(raw_p)

    return stats, winners, delta_pct, corrected_p, all_keys

def print_final_table(stats, winners, delta_pct, corrected_p, all_keys, opt_names):
    # Transposed layout: rows = metrics, cols = optimizers (whichever ones
    # are present in this run), with a WINNER column at the end.
    metric_col_w = 18
    opt_col_w = max(18, max((len(o) for o in opt_names), default=18) + 4)
    log_sep("=", 200)
    log("  RUN 1 RESULTS  (mean +/- std across shards of per-shard seed-medians; N=10 pseudo-domain shards, 3 seeds)")
    log_sep("-", 200)
    header = f"  {'Metric':<18}"
    for opt in opt_names:
        header += f"{opt:>{opt_col_w}}"
    header += f"{'WINNER':>{opt_col_w}}"
    log(header)
    log("  " + "-" * (18 + opt_col_w * (len(opt_names) + 1)))
    for k in all_keys:
        row = f"  {k:<18}"
        for opt in opt_names:
            m, sd = stats[opt][k]
            cell = f"{m:.4g}+/-{sd:.2g}" if not math.isnan(m) else "nan"
            row += f"{cell:>{opt_col_w}}"
        row += f"{winners[k]:>{opt_col_w}}"
        log(row)
    log_sep("-", 200)
    log("  RSCO-Amplify vs AdamW  (median %delta, positive = RSCO-Amplify better; direction-aware; Holm-Bonferroni corrected p)")
    if "RSCO-Amplify" in opt_names and "AdamW" in opt_names:
        for k in METRIC_KEYS:
            d = delta_pct.get(k, float('nan'))
            p = corrected_p.get(k, float('nan'))
            log(f"    {k:<18}  %delta={d:+7.2f}%   p_holm={p:.3e}")
    else:
        log("    (skipped: this run does not include both RSCO-Amplify and AdamW)")
    log_sep("=", 200)

# ==============================================================================
#  CSV
# ==============================================================================
def write_csv(raw_by_opt_shard, opt_names, shards, seeds, lr_used):
    fieldnames = ['opt', 'shard', 'seed', 'lr_used'] + METRIC_KEYS
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for opt in opt_names:
            for shard in shards:
                runs = raw_by_opt_shard.get(opt, {}).get(shard, [])
                for i, seed in enumerate(seeds):
                    if i < len(runs):
                        row = {'opt': opt, 'shard': shard, 'seed': seed,
                               'lr_used': lr_used.get(opt, '')}
                        row.update({k: runs[i].get(k, float('nan')) for k in METRIC_KEYS})
                        writer.writerow(row)
    log(f"[CSV] Saved -> {CSV_PATH}")

# ==============================================================================
#  MAIN
# ==============================================================================

def lr_grid_search(*args, **kwargs):
    try:
        return _lr_grid_search_unsafe(*args, **kwargs)
    except Exception as e:
        log(f"[ERROR] lr_grid_search failed: {e}")
        import traceback
        traceback.print_exc()
        return 1e-3

def run_experiment(*args, **kwargs):
    try:
        return _run_experiment_unsafe(*args, **kwargs)
    except Exception as e:
        log(f"[ERROR] run_experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'valloss': float('nan'), 'ppl': float('nan'), 'top1acc': float('nan'),
            'steps50%': 5000, 'stepseps': 5000, 'wall(s)': 0.0,
            'peakmemmb': 0.0, 'gflops': 0.0, 'exaflops': 0.0,
            'stepsize': float('nan'), 'clipfreq%': float('nan'),
            'clipcoord%': float('nan'), 'hessnorm': float('nan'),
            'hesscv': float('nan'), 'regret': float('nan'),
            'n_params': 0, 'lr_used': 1e-3
        }

def main():
    log_sep("=")
    log(f"  RUN 1 PART B (AdamW, RSCO-Amplify)  |  device={device}  torch={torch.__version__}")
    log_sep("=")

    ckpt = load_checkpoint()

    # ---- 1. shard selection (resume if checkpointed) ----
    if ckpt["shards"] is None:
        shards = select_shards(N_SHARDS_TARGET, seed=1234)
        ckpt["shards"] = shards
        save_checkpoint(ckpt)
    else:
        shards = ckpt["shards"]
        for s in shards:            # re-warm data cache
            try_load_shard(s)
    log(f"[SHARDS] Using {len(shards)} pseudo-domain shards.")

    # ---- 2. LR search (resume if checkpointed) ----
    if not ckpt["lr_used"]:
        tuning_subset = stratified_tuning_subset(shards, N_TUNING_SHARDS)
        log(f"[LR-SEARCH] Tuning subset ({len(tuning_subset)}): {tuning_subset}")
        lr_used = {}
        for opt_name in OPT_NAMES:
            lr_used[opt_name] = lr_grid_search(opt_name, tuning_subset, BATCH_SIZE_DEFAULT, MODEL_SIZE_DEFAULT)
        ckpt["lr_used"] = lr_used
        save_checkpoint(ckpt)
    else:
        lr_used = ckpt["lr_used"]
    log(f"[LR-SEARCH] Frozen LRs: {lr_used}")

    # ---- 3. full sweep, with checkpointing per (opt,shard) ----
    raw = {opt: {s: [] for s in shards} for opt in OPT_NAMES}
    for opt in OPT_NAMES:
        for s in shards:
            key = f"{opt}|{s}"
            if key in ckpt["raw"]:
                raw[opt][s] = ckpt["raw"][key]

    total = len(shards) * len(OPT_NAMES) * len(SEEDS)
    done = sum(len(v) for opt in raw for v in raw[opt].values())
    log(f"[SWEEP] {done}/{total} runs already complete (resumed).")

    for s_idx, shard in enumerate(shards):
        for opt in OPT_NAMES:
            key = f"{opt}|{shard}"
            if key in ckpt["completed"]:
                continue
            lr = lr_used[opt]
            results = []
            for seed in SEEDS:
                r = run_experiment(opt, lr, shard, seed,
                                    batch_size=BATCH_SIZE_DEFAULT, hidden=MODEL_SIZE_DEFAULT,
                                    max_steps=MAX_STEPS)
                results.append(r)
            raw[opt][shard] = results
            ckpt["raw"][key] = results
            ckpt["completed"][key] = True
            save_checkpoint(ckpt)
        log(f"[PROGRESS {s_idx+1}/{len(shards)}] {shard} done (all optimizers).")

    # ---- 4. aggregate + report ----
    agg_by_shard = {opt: {s: aggregate_shard(raw[opt][s]) for s in shards} for opt in OPT_NAMES}

    write_csv(raw, OPT_NAMES, shards, SEEDS, lr_used)
    with open(JSON_PATH, 'w') as jf:
        json.dump({
            'metadata': {'shards': shards, 'seeds': SEEDS, 'max_steps': MAX_STEPS,
                         'batch_size': BATCH_SIZE_DEFAULT, 'model_hidden': MODEL_SIZE_DEFAULT,
                         'vocab_size': VOCAB_SIZE, 'embed_dim': EMBED_DIM, 'context_k': CONTEXT_K,
                         'vocab_note': 'VOCAB_SIZE reduced from 20000 to 5000 for cross-domain runtime '
                                       'parity; perplexities here are valid for within-run optimizer '
                                       'comparisons only and are NOT comparable to published WikiText-2 '
                                       'perplexities (~20k-33k vocab).',
                         'lr_used': lr_used, 'timestamp': datetime.now().isoformat()},
            'agg_by_shard': agg_by_shard,
        }, jf, indent=2)
    log(f"[JSON] Saved -> {JSON_PATH}")

    stats, winners, delta_pct, corrected_p, all_keys = build_final_table(agg_by_shard, OPT_NAMES, shards)
    print_final_table(stats, winners, delta_pct, corrected_p, all_keys, OPT_NAMES)

    log_sep("=")
    log(f"  ALL DONE  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_sep("=")
    _log_fh.close()

if __name__ == '__main__':
    main()