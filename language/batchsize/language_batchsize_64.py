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
import warnings, time, math, csv, json, itertools, copy, random, re, signal, shutil, traceback
from datetime import datetime
from collections import defaultdict, Counter
from scipy.stats import wilcoxon
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# ==============================================================================
#  GLOBAL CONFIG  (per spec -- mirrors finance Run 2's batch-size sweep axis)
# ==============================================================================
SEEDS               = [42, 43]
EVAL_EVERY          = 250
MODEL_SIZE_DEFAULT  = (64, 32)

# ---- Run-2 batch-size sweep axis ----
BATCH_SIZES         = [64]                   # batch size run: batch 64
FULL_STEPS_BY_BATCH = {2: 80000, 4: 40000, 8: 20000, 16: 10000, 64: 2500}
LR_SEARCH_SAMPLES_SEEN = 3200                # fixed budget: 100 * 32 / batch steps
HESS_FREQ           = 5
N_HUTCH             = 4
GRAD_CLIP_NORM      = 5.0
WEIGHT_DECAY        = 1e-4
TARGET_EPS_FRAC     = 0.1
SOPHIA_BETA_H       = 0.99

LR_GRID             = [1e-4, 3e-4, 1e-3, 3e-3]
LR_SEARCH_SEED      = 42
N_TUNING_SHARDS     = 10   # subset of the 10 pseudo-domain shards used for LR search

CLIP_METRIC_THRESHOLD = 1.0
EPS_SCE              = 1e-8

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

# ---- Fix 1: fixed, deterministic subsample of the validation split used for
# in-loop (every EVAL_EVERY steps, and LR-search) scoring only. The final
# post-training test-set evaluation NEVER uses this -- it always uses the
# full test split. ----
VAL_FAST_SIZE = 2000
VAL_FAST_SEED = 777

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR    = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH  = os.path.join(OUT_DIR, "lm_run2_batch64_checkpoint.json")
CKPT_BAK_PATH = CKPT_PATH + ".bak"
CSV_PATH   = os.path.join(OUT_DIR, "lm_run2_batch64_results_raw.csv")
JSON_PATH  = os.path.join(OUT_DIR, "lm_run2_batch64_results_agg.json")
LOG_PATH   = os.path.join(OUT_DIR, "lm_run2_batch64_log.txt")

_log_fh = open(LOG_PATH, "a", buffering=1)
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        _log_fh.write(line + "\n")
    except Exception:
        pass
def log_sep(c="=", w=140):
    log(c * w)

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================================================================
#  DATA  (canonical WikiText-2 train/valid/test split. Vocab is built from
#  pooled TRAIN-ONLY text and shared across all shards -> no leakage.  The
#  canonical TRAIN split is partitioned into N_SHARDS_TARGET=10 pseudo-domain
#  shards along article boundaries; each shard plays the role a "ticker"
#  played in the finance harness -- its own training stream, its own LR
#  search sample, its own row in the per-shard aggregation table.  The
#  canonical validation/test splits are corpus-level (not per-article), so
#  they are shared, unmodified, across every shard's evaluation. Identical
#  partition to Run 1 so results are directly comparable across batch sizes.)
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

def load_corpus(max_retries=5, base_delay=2.0):
    """Downloads canonical WikiText-2 (raw) once, builds the shared vocab from
    pooled TRAIN text only, splits TRAIN into N_SHARDS_TARGET pseudo-domain
    shards along article boundaries, and tokenizes the canonical valid/test
    splits (shared across all shards). Fix 2: retries the download with
    exponential backoff before giving up, instead of failing on the first
    transient network error."""
    global _RAW, _VAL_TOKENS, _TEST_TOKENS
    if _RAW is not None:
        return

    ds = None
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            log(f"[DATA] Downloading canonical WikiText-2 (train/valid/test splits)... "
                f"(attempt {attempt}/{max_retries})")
            ds = hf_datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
            break
        except Exception as e:
            last_err = e
            delay = base_delay * (2 ** (attempt - 1))
            log(f"[DATA][WARN] download failed ({e}); retrying in {delay:.1f}s...")
            time.sleep(delay)
    if ds is None:
        raise RuntimeError(f"[DATA] Failed to download WikiText-2 after {max_retries} attempts: {last_err}")

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

def _fixed_subsample(X, y, n, seed):
    """Deterministic fixed-seed subsample of (X, y), capped at n rows. Used
    only for the fast in-loop validation proxy -- never for the reported
    test-set metrics."""
    if len(X) <= n:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    idx.sort()
    return X[idx], y[idx]

_SHARD_WINDOWS = {}
_VAL_WINDOWS = None
_VAL_WINDOWS_FAST = None
_TEST_WINDOWS = None

def try_load_shard(shard_id):
    """Analogous to try_load_ticker in the finance harness: builds this
    shard's windowed train tensors (+ the shared val/test windows, computed
    once) on first access. Returns True if the shard has enough windows to
    train on, False if it should be skipped."""
    load_corpus()
    global _VAL_WINDOWS, _VAL_WINDOWS_FAST, _TEST_WINDOWS
    if _VAL_WINDOWS is None:
        _VAL_WINDOWS = _windows(_VAL_TOKENS)
        _TEST_WINDOWS = _windows(_TEST_TOKENS)
        Xva, yva = _VAL_WINDOWS
        _VAL_WINDOWS_FAST = _fixed_subsample(Xva, yva, VAL_FAST_SIZE, VAL_FAST_SEED)
        log(f"[DATA] val windows: full={len(Xva)}  fast-subsample={len(_VAL_WINDOWS_FAST[0])} "
            f"(VAL_FAST_SIZE={VAL_FAST_SIZE}, seed={VAL_FAST_SEED}) -- fast subsample used ONLY "
            f"for in-loop/LR-search scoring; final reported metrics always use the full TEST split.")
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
    """Returns (train_loader, val_loader_full, val_loader_fast, test_loader).
    val_loader_fast is the subsampled loader used for in-loop checkpoint
    selection / regret (Fix 1). The final reported metrics come from
    test_loader, untouched by this fix."""
    Xtr, ytr = _SHARD_WINDOWS[shard_id]
    Xva, yva = _VAL_WINDOWS
    Xvf, yvf = _VAL_WINDOWS_FAST
    Xte, yte = _TEST_WINDOWS
    def ds(Xd, yd):
        return TensorDataset(torch.tensor(Xd, dtype=torch.long),
                              torch.tensor(yd, dtype=torch.long))
    return (DataLoader(ds(Xtr, ytr), batch_size=batch_size, shuffle=True),
            DataLoader(ds(Xva, yva), batch_size=batch_size),
            DataLoader(ds(Xvf, yvf), batch_size=batch_size),
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
    sample evenly across the shard list). Fixed once and reused across every
    batch size, same as the finance harness's fixed tuning-ticker subset."""
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
    """Forward-FLOP estimator for the LM model: embedding lookup is a gather
    (no FLOPs), so the cost is the two MLP hidden layers plus the final
    vocab-projection Linear(h2, vocab_size) -- which dominates since
    vocab_size >> h1, h2."""
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


class EvieOptimizer(torch.optim.Optimizer):
    """Base Evie. gamma0 is auto-calibrated ONCE, at a step in [100,150],
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
            log(f"    [Evie] gamma0={new_g0:.1f}  fast_sigma2={mean_s2:.2e}")

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


class EvieAmplifyOptimizer(torch.optim.Optimizer):
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
            log(f"    [Evie-Amplify] gamma0={new_g0:.1f}  fast_sigma2={mean_s2:.2e}")

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
                # FIX 3: h_hat must only be defaulted the first time this
                # param's state is touched, independent of whether 'step'
                # was just initialized above. Previously this line lived
                # inside the `if 'step' not in s:` block, which meant it
                # was NOT re-run on later calls -- but set_hessian() runs
                # BEFORE step() on probe steps (including step 0), writing
                # the real Hutchinson h_hat into a state dict that had no
                # 'step' key yet. On step 0, step() would then hit
                # `'step' not in s` == True and stomp the just-written
                # h_hat back to torch.ones_like(p.data), discarding the
                # first Hessian probe. Gating solely on 'h_hat' not in s
                # preserves whatever set_hessian() already wrote.
                if 'h_hat' not in s:
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
                # FIX 3: same h_hat-clobber fix as SophiaOptimizer.step()
                # above -- see the comment there for the full explanation.
                if 'h_hat' not in s:
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
    if opt_name == "Evie":
        return EvieOptimizer(model.parameters(), lr=lr, gamma0=1.0,
                              target_wp=TARGET_WP, weight_decay=WEIGHT_DECAY)
    elif opt_name == "Evie-Amplify":
        return EvieAmplifyOptimizer(model.parameters(), lr=lr, gamma0=1.0,
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

OPT_NAMES = ["AdamW", "Evie", "Evie-Amplify", "Sophia", "AdaHessian"]
HESSIAN_CONSUMERS = {"Sophia", "AdaHessian"}

# ==============================================================================
#  HESSIAN ESTIMATORS  (Hutchinson diagonal, consumed only by Sophia &
#  AdaHessian every HESS_FREQ steps -- same restriction as the finance Run-2
#  harness: no passive probe for AdamW/Evie/Evie-Amplify, so hessnorm/hesscv
#  are nan for those three in this batch-sweep run.)
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
    targets equal to <unk>; there is no pad token here since every window
    has a full k=4 context."""
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
#  LR GRID SEARCH  -- fixed tuning subset, ALL 5 optimizers, frozen after.
#  n_search_steps varies with batch size so samples-seen (3,200) stays fixed
#  across the sweep -- identical mechanism to the finance Run-2 harness.
#  (Fix 1: scoring now uses the fast validation subsample, not the full
#  validation split -- LR search runs a val eval once per (lr, shard)
#  combination, so this was paying the full-val cost LR_GRID x tuning_shards
#  times.)
# ==============================================================================
def _lr_grid_search_unsafe(opt_name, tuning_shards, batch_size, hidden, n_search_steps):
    val_by_lr = defaultdict(list)
    for lr in LR_GRID:
        for shard in tuning_shards:
            set_seed(LR_SEARCH_SEED)
            train_loader, val_loader, val_loader_fast, _ = get_loaders(shard, batch_size)
            model = NPLM_MLP(hidden).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = build_optimizer(opt_name, model, lr)
            params = [p for p in model.parameters() if p.requires_grad]
            loader_iter = itertools.cycle(train_loader)
            for step in range(n_search_steps):
                model.train()
                X_batch, y_batch = next(loader_iter)
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                if step % HESS_FREQ == 0 and opt_name in HESSIAN_CONSUMERS:
                    hutchinson_hessian(model, params, X_batch, y_batch, criterion,
                                        optimizer=optimizer, feed_optimizer=True)
                optimizer.step()
            val_loss, _ = evaluate(model, val_loader_fast, criterion)
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
#  (Fix 1: the in-loop EVAL_EVERY-step eval used for best_state checkpoint
#  selection and the `regret` metric now uses val_loader_fast, the fixed
#  subsample. The final, one-time post-training evaluation below still uses
#  test_loader -- the FULL test split, completely untouched by this fix.)
# ==============================================================================
def _run_experiment_unsafe(opt_name, lr, shard, seed, batch_size,
                    hidden=MODEL_SIZE_DEFAULT, max_steps=None):
    set_seed(seed)
    reset_peak_memory()

    train_loader, val_loader, val_loader_fast, test_loader = get_loaders(shard, batch_size)
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
    is_evie_family = opt_name in ("Evie", "Evie-Amplify")

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

        ran_probe = (step % HESS_FREQ == 0) and is_hessian_consumer
        h_list = None
        if ran_probe:
            # Hessian estimator (Hutchinson diagonal) runs ONLY for the two
            # optimizers that actually consume it (Sophia, AdaHessian) --
            # same restriction as the finance Run-2 harness.
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

        if is_evie_family and not calibrated_once and step == CALIBRATE_STEP:
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
            # Fix 1: fast subsample, not the full validation split.
            val_loss, _ = evaluate(model, val_loader_fast, criterion)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
            best_val_so_far = min(best_val_so_far, val_loss)
            cum_regret += max(val_loss - best_val_so_far, 0.0)
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    # Untouched by Fix 1: final one-time evaluation on the FULL test split --
    # this is the source of the actually-reported valloss/ppl/top1acc.
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

    # ppl is computed over the reduced VOCAB_SIZE=5000 output space; it's valid
    # for within-run optimizer comparisons (e.g. Evie vs AdamW, same vocab
    # throughout) but is NOT numerically comparable to published WikiText-2
    # perplexities, which conventionally use ~20k-33k vocab.
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
#  (Fix 2: atomic + crash-safe save with .bak rotation; resilient load with
#  .bak fallback and a migration/setdefault pass for old-format checkpoints.)
# ==============================================================================
def lr_grid_search(*args, **kwargs):
    try:
        return _lr_grid_search_unsafe(*args, **kwargs)
    except Exception as e:
        log(f"[ERROR] lr_grid_search failed: {e}")
        traceback.print_exc()
        return 1e-3

def run_experiment(*args, **kwargs):
    try:
        return _run_experiment_unsafe(*args, **kwargs)
    except Exception as e:
        log(f"[ERROR] run_experiment failed: {e}")
        traceback.print_exc()
        return {
            "valloss": float("nan"), "ppl": float("nan"), "top1acc": float("nan"),
            "steps50%": kwargs.get("max_steps", 0),
            "stepseps": kwargs.get("max_steps", 0),
            "wall(s)": 0.0, "peakmemmb": 0.0,
            "gflops": 0.0, "exaflops": 0.0,
            "stepsize": float("nan"), "clipfreq%": float("nan"),
            "clipcoord%": float("nan"),
            "hessnorm": float("nan"), "hesscv": float("nan"),
            "regret": float("nan"),
            "n_params": 0, "lr_used": args[1] if len(args) > 1 else 1e-3,
        }

def _default_ckpt():
    return {"shards": None, "tuning_subset": None, "by_batch": {}}

def _migrate_checkpoint(ckpt):
    """Ensure checkpoints written by the pre-fix version of this script
    (which tracked completion per (opt, shard) job rather than per seed)
    still load without KeyErrors."""
    if not isinstance(ckpt, dict):
        return _default_ckpt()
    ckpt.setdefault("shards", None)
    ckpt.setdefault("tuning_subset", None)
    ckpt.setdefault("by_batch", {})
    for bkey, bckpt in ckpt["by_batch"].items():
        bckpt.setdefault("lr_used", {})
        bckpt.setdefault("completed", {})
        bckpt.setdefault("raw", {})
        # old-format "raw" entries are already a list of per-seed result
        # dicts (one per completed seed) -- that shape is unchanged, so no
        # further migration is needed there. "completed" entries from the
        # old per-job flag are left as-is; the new resume logic checks
        # len(raw[key]) directly and does not depend on this flag except to
        # avoid re-deriving it once a job is fully done.
        for key, results in list(bckpt["raw"].items()):
            if not isinstance(results, list):
                bckpt["raw"][key] = []
    return ckpt

def load_checkpoint():
    """Fix 2: resilient load -- fall back to .bak if the main file is
    corrupt/unreadable; start fresh if both are unreadable."""
    for path in (CKPT_PATH, CKPT_BAK_PATH):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                ckpt = json.load(f)
            if path == CKPT_BAK_PATH:
                log(f"[CKPT][WARN] main checkpoint unreadable; loaded from .bak fallback.")
            return _migrate_checkpoint(ckpt)
        except Exception as e:
            log(f"[CKPT][WARN] failed to load checkpoint from {path}: {e}")
    log("[CKPT] No valid checkpoint found -- starting fresh.")
    return _default_ckpt()

def save_checkpoint(ckpt):
    """Atomic write (tmp file + fsync + os.replace), with rotation of the
    previous good checkpoint to .bak first. Never raises -- a failed save
    logs a warning and the run continues (Fix 2)."""
    try:
        tmp = CKPT_PATH + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(ckpt, f)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(CKPT_PATH):
            try:
                shutil.copyfile(CKPT_PATH, CKPT_BAK_PATH)
            except Exception as e:
                log(f"[CKPT][WARN] failed to rotate backup checkpoint: {e}")
        os.replace(tmp, CKPT_PATH)
    except Exception as e:
        log(f"[CKPT][WARN] checkpoint save failed (continuing run): {e}")

# ---- Fix 2: best-effort signal handling -- flush the checkpoint before exit
# on SIGTERM/SIGINT. Won't help against a hard kill with no grace period, but
# helps for anything that sends a signal first (e.g. platform timeouts that
# SIGTERM before SIGKILL). ----
_CKPT_REF = {"ckpt": None}

def _signal_flush_handler(signum, frame):
    log(f"[SIGNAL] received signal {signum} -- flushing checkpoint before exit.")
    if _CKPT_REF["ckpt"] is not None:
        save_checkpoint(_CKPT_REF["ckpt"])
    raise SystemExit(128 + signum)

def _install_signal_handlers():
    try:
        signal.signal(signal.SIGTERM, _signal_flush_handler)
        signal.signal(signal.SIGINT, _signal_flush_handler)
    except Exception as e:
        log(f"[SIGNAL][WARN] could not install signal handlers: {e}")

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

# ==============================================================================
#  BATCH-SIZE TREND TABLES
#  (a) valloss mean+-std, rows=batch size, cols=optimizer
#  (b) Evie-Amplify vs AdamW %delta (+ p_holm, corrected jointly across the
#      batch-size rows) on valloss, rows=batch size
# ==============================================================================
def build_batch_trend_tables(agg_by_batch_shard, opt_names, shards, batch_sizes):
    challenger = "Evie-Amplify"

    # (a) mean+-std valloss per (batch, opt), across shards of seed-medians
    loss_stats = {}
    for b in batch_sizes:
        loss_stats[b] = {}
        for opt in opt_names:
            vals = [agg_by_batch_shard[b][opt][s]['valloss'] for s in shards
                    if not math.isnan(agg_by_batch_shard[b][opt][s]['valloss'])]
            loss_stats[b][opt] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float('nan'), float('nan'))

    # (b) Evie-Amplify vs AdamW %delta on valloss (lower is better) + Wilcoxon,
    # Holm-Bonferroni corrected jointly across all batch-size rows
    raw_p = {}
    delta_pct = {}
    for b in batch_sizes:
        c_vals, b_vals = [], []
        for s in shards:
            c = agg_by_batch_shard[b][challenger][s]['valloss']
            a = agg_by_batch_shard[b]["AdamW"][s]['valloss']
            if not (math.isnan(c) or math.isnan(a)):
                c_vals.append(c); b_vals.append(a)
        if len(c_vals) >= 10:
            try:
                _, p = wilcoxon(c_vals, b_vals)
            except ValueError:
                p = float('nan')
        else:
            p = float('nan')
        raw_p[b] = p
        if b_vals and c_vals:
            a_med, c_med = float(np.median(b_vals)), float(np.median(c_vals))
            delta_pct[b] = (a_med - c_med) / abs(a_med) * 100.0 if a_med != 0 else float('nan')
        else:
            delta_pct[b] = float('nan')
    corrected_p = holm_bonferroni(raw_p)

    return loss_stats, delta_pct, corrected_p

def print_batch_trend_tables(loss_stats, delta_pct, corrected_p, opt_names, batch_sizes):
    log_sep("=", 160)
    log("  RUN 2 -- BATCH-SIZE SWEEP: valloss (mean +/- std across shards of per-shard seed-medians)")
    log_sep("-", 160)
    header = f"  {'Batch':<8}"
    for opt in opt_names:
        header += f"{opt:>22}"
    log(header)
    log("  " + "-" * (8 + 22 * len(opt_names)))
    for b in batch_sizes:
        row = f"  {b:<8}"
        for opt in opt_names:
            m, sd = loss_stats[b][opt]
            cell = f"{m:.4g}+/-{sd:.2g}" if not math.isnan(m) else "nan"
            row += f"{cell:>22}"
        log(row)
    log_sep("-", 160)
    log("  Evie-Amplify vs AdamW on valloss  (%delta, positive = Evie-Amplify better; Holm-Bonferroni p across batch sizes)")
    log(f"  {'Batch':<8}{'%delta':>12}{'p_holm':>14}")
    for b in batch_sizes:
        d = delta_pct.get(b, float('nan'))
        p = corrected_p.get(b, float('nan'))
        log(f"  {b:<8}{d:>+11.2f}%{p:>14.3e}")
    log_sep("=", 160)

def print_full_metrics_table(agg_by_batch_shard, opt_names, shards, batch_sizes):
    """Full per-metric summary table across ALL METRIC_KEYS (not just
    valloss), mean +/- std across shards of per-shard seed-medians. Brings
    this script's output to parity with the LANGUAGE_BATCHSIZE_* siblings,
    which already print this."""
    log_sep("=", 200)
    log("  FULL METRICS TABLE  (mean +/- std across shards of per-shard seed-medians)")
    log_sep("-", 200)
    for b in batch_sizes:
        log(f"  --- batch_size={b} ---")
        header = f"  {'metric':<14}"
        for opt in opt_names:
            header += f"{opt:>22}"
        log(header)
        log("  " + "-" * (14 + 22 * len(opt_names)))
        for metric in METRIC_KEYS:
            row = f"  {metric:<14}"
            for opt in opt_names:
                vals = [agg_by_batch_shard[b][opt][s][metric] for s in shards
                        if not (isinstance(agg_by_batch_shard[b][opt][s][metric], float)
                                and math.isnan(agg_by_batch_shard[b][opt][s][metric]))]
                if vals:
                    m, sd = float(np.mean(vals)), float(np.std(vals))
                    cell = f"{m:.4g}+/-{sd:.2g}"
                else:
                    cell = "nan"
                row += f"{cell:>22}"
            log(row)
    log_sep("=", 200)

# ==============================================================================
#  CSV
# ==============================================================================
def write_csv(raw_by_batch_opt_shard, opt_names, shards, seeds, lr_used_by_batch, batch_sizes):
    fieldnames = ['batch_size', 'opt', 'shard', 'seed', 'lr_used'] + METRIC_KEYS
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b in batch_sizes:
            for opt in opt_names:
                for shard in shards:
                    runs = raw_by_batch_opt_shard.get(b, {}).get(opt, {}).get(shard, [])
                    for i, seed in enumerate(seeds):
                        if i < len(runs):
                            row = {'batch_size': b, 'opt': opt, 'shard': shard, 'seed': seed,
                                   'lr_used': lr_used_by_batch.get(b, {}).get(opt, '')}
                            row.update({k: runs[i].get(k, float('nan')) for k in METRIC_KEYS})
                            writer.writerow(row)
    log(f"[CSV] Saved -> {CSV_PATH}")

# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    log_sep("=")
    log(f"  RUN 2 -- BATCH-SIZE SWEEP (LM)  |  device={device}  torch={torch.__version__}")
    log(f"  batch sizes: {BATCH_SIZES}  |  full-run steps: {FULL_STEPS_BY_BATCH}")
    log(f"  vocab_size={VOCAB_SIZE}  n_shards_target={N_SHARDS_TARGET}")
    log(f"  VAL_FAST_SIZE={VAL_FAST_SIZE}  VAL_FAST_SEED={VAL_FAST_SEED} "
        f"(in-loop/LR-search eval proxy only; final metrics use full test split)")
    log_sep("=")

    ckpt = load_checkpoint()
    _CKPT_REF["ckpt"] = ckpt

    # ---- 1. shard selection (same 10 shards as Run 1, resume if checkpointed) ----
    if ckpt["shards"] is None:
        shards = select_shards(N_SHARDS_TARGET, seed=1234)
        ckpt["shards"] = shards
        save_checkpoint(ckpt)
    else:
        shards = ckpt["shards"]
        for s in shards:            # re-warm data cache
            try_load_shard(s)
    log(f"[SHARDS] Using {len(shards)} pseudo-domain shards.")

    # ---- 2. fixed tuning subset, same across every batch size ----
    if ckpt["tuning_subset"] is None:
        tuning_subset = stratified_tuning_subset(shards, N_TUNING_SHARDS)
        ckpt["tuning_subset"] = tuning_subset
        save_checkpoint(ckpt)
    else:
        tuning_subset = ckpt["tuning_subset"]
    log(f"[TUNING SUBSET] ({len(tuning_subset)}, fixed across all batch sizes): {tuning_subset}")

    raw_by_batch = {}
    lr_used_by_batch = {}
    agg_by_batch_shard = {}

    for batch_size in BATCH_SIZES:
        bkey = str(batch_size)
        full_steps = FULL_STEPS_BY_BATCH[batch_size]
        n_search_steps = round(LR_SEARCH_SAMPLES_SEEN / batch_size)
        log_sep("=")
        log(f"  BATCH SIZE {batch_size}  |  lr-search steps={n_search_steps} (samples seen={n_search_steps*batch_size})"
            f"  |  full-run steps={full_steps} (samples seen={full_steps*batch_size})")
        log_sep("=")

        bckpt = ckpt["by_batch"].setdefault(bkey, {"lr_used": {}, "completed": {}, "raw": {}})

        # ---- 2a. LR search at this batch size (resume if checkpointed) ----
        if not bckpt["lr_used"]:
            lr_used = {}
            for opt_name in OPT_NAMES:
                lr_used[opt_name] = lr_grid_search(opt_name, tuning_subset, batch_size,
                                                          MODEL_SIZE_DEFAULT, n_search_steps)
            bckpt["lr_used"] = lr_used
            save_checkpoint(ckpt)
        else:
            lr_used = bckpt["lr_used"]
        log(f"[LR-SEARCH batch={batch_size}] Frozen LRs: {lr_used}")
        lr_used_by_batch[batch_size] = lr_used

        # ---- 2b. full sweep at this batch size, checkpointed PER SEED (Fix 2:
        # a (opt, shard) job that's half-done resumes from the missing seed
        # only, instead of restarting both; checkpoint saved after EACH seed,
        # not after the whole job) ----
        raw = {opt: {s: [] for s in shards} for opt in OPT_NAMES}
        for opt in OPT_NAMES:
            for s in shards:
                key = f"{opt}|{s}"
                if key in bckpt["raw"]:
                    raw[opt][s] = bckpt["raw"][key]

        total_seeds = len(shards) * len(OPT_NAMES) * len(SEEDS)
        done_seeds = sum(len(v) for opt in raw for v in raw[opt].values())
        log(f"[SWEEP batch={batch_size}] {done_seeds}/{total_seeds} seed-runs already complete (resumed).")

        for s_idx, shard in enumerate(shards):
            for opt in OPT_NAMES:
                key = f"{opt}|{shard}"
                lr = lr_used[opt]
                existing = bckpt["raw"].get(key, [])
                results = list(existing)  # keep already-completed seeds

                for seed_idx, seed in enumerate(SEEDS):
                    if seed_idx < len(results):
                        continue  # this seed already has a saved result
                    try:
                        r = run_experiment(opt, lr, shard, seed,
                                            batch_size=batch_size, hidden=MODEL_SIZE_DEFAULT,
                                            max_steps=full_steps)
                    except Exception as e:
                        # run_experiment already catches internally, but this
                        # is an extra outer guard so one bad seed can never
                        # take down the whole sweep.
                        log(f"[ERROR] unhandled exception for {key} seed={seed}: {e}")
                        traceback.print_exc()
                        r = {
                            'valloss': float('nan'), 'ppl': float('nan'), 'top1acc': float('nan'),
                            'steps50%': full_steps, 'stepseps': full_steps,
                            'wall(s)': 0.0, 'peakmemmb': 0.0, 'gflops': 0.0, 'exaflops': 0.0,
                            'stepsize': float('nan'), 'clipfreq%': float('nan'),
                            'clipcoord%': float('nan'), 'hessnorm': float('nan'),
                            'hesscv': float('nan'), 'regret': float('nan'),
                            'n_params': 0, 'lr_used': lr,
                        }
                    results.append(r)
                    bckpt["raw"][key] = results
                    save_checkpoint(ckpt)
                    log(f"[PROGRESS batch={batch_size}] {key}|seed={seed} done "
                        f"({len(results)}/{len(SEEDS)} seeds).")

                raw[opt][shard] = results
                if len(results) >= len(SEEDS):
                    bckpt["completed"][key] = True
            log(f"[PROGRESS batch={batch_size} {s_idx+1}/{len(shards)}] {shard} done (all optimizers).")

        raw_by_batch[batch_size] = raw
        agg_by_batch_shard[batch_size] = {opt: {s: aggregate_shard(raw[opt][s]) for s in shards}
                                           for opt in OPT_NAMES}

    # ---- 3. aggregate + report across all batch sizes. Fix 2: every output
    # stage below is independently wrapped -- all of this is derived from
    # data already safely on disk in the checkpoint, so a failure in any one
    # stage logs and continues rather than crashing the run or losing raw
    # results. ----
    try:
        write_csv(raw_by_batch, OPT_NAMES, shards, SEEDS, lr_used_by_batch, BATCH_SIZES)
    except Exception as e:
        log(f"[ERROR] CSV write failed: {e}")
        traceback.print_exc()

    try:
        with open(JSON_PATH, 'w') as jf:
            json.dump({
                'metadata': {'shards': shards, 'seeds': SEEDS, 'batch_sizes': BATCH_SIZES,
                             'full_steps_by_batch': FULL_STEPS_BY_BATCH,
                             'lr_search_samples_seen': LR_SEARCH_SAMPLES_SEEN,
                             'model_hidden': MODEL_SIZE_DEFAULT, 'tuning_subset': tuning_subset,
                             'vocab_size': VOCAB_SIZE, 'embed_dim': EMBED_DIM, 'context_k': CONTEXT_K,
                             'val_fast_size': VAL_FAST_SIZE, 'val_fast_seed': VAL_FAST_SEED,
                             'vocab_note': 'VOCAB_SIZE reduced to 5000 (from 20000) for cross-domain '
                                           '(fin/lang/vision) runtime parity; ppl figures are valid for '
                                           'within-run optimizer comparisons only and are NOT numerically '
                                           'comparable to published WikiText-2 perplexities (~20k-33k vocab).',
                             'metric_keys': METRIC_KEYS,
                             'lr_used_by_batch': lr_used_by_batch, 'timestamp': datetime.now().isoformat()},
                'agg_by_batch_shard': agg_by_batch_shard,
            }, jf, indent=2)
        log(f"[JSON] Saved -> {JSON_PATH}")
    except Exception as e:
        log(f"[ERROR] JSON write failed: {e}")
        traceback.print_exc()

    try:
        loss_stats, delta_pct, corrected_p = build_batch_trend_tables(agg_by_batch_shard, OPT_NAMES, shards, BATCH_SIZES)
        print_batch_trend_tables(loss_stats, delta_pct, corrected_p, OPT_NAMES, BATCH_SIZES)
    except Exception as e:
        log(f"[ERROR] batch trend table failed: {e}")
        traceback.print_exc()

    try:
        print_full_metrics_table(agg_by_batch_shard, OPT_NAMES, shards, BATCH_SIZES)
    except Exception as e:
        log(f"[ERROR] full metrics table failed: {e}")
        traceback.print_exc()

    log_sep("=")
    log(f"  ALL DONE  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_sep("=")

if __name__ == '__main__':
    _install_signal_handlers()
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        log(f"[FATAL] unhandled exception in main(): {e}")
        traceback.print_exc()
        if _CKPT_REF["ckpt"] is not None:
            save_checkpoint(_CKPT_REF["ckpt"])
        raise
    finally:
        try:
            _log_fh.close()
        except Exception:
            pass