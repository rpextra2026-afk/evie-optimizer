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
import signal, threading, traceback
from datetime import datetime
from collections import defaultdict, Counter
from scipy.stats import wilcoxon
from torch.utils.data import TensorDataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# ==============================================================================
#  GLOBAL CONFIG  (per spec)
# ==============================================================================
SEEDS               = [42, 43]
EVAL_EVERY          = 250
MODEL_SIZE_DEFAULT  = (64, 32)

# ---- LANGUAGE_BATCHSIZE batch-size sweep axis (mirrors finance Run 2) ----
BATCH_SIZES         = [2]                    # batch size run: LANGUAGE_BATCHSIZE 2
FULL_STEPS_BY_BATCH = {2: 80000, 4: 40000, 8: 20000, 16: 10000, 64: 2500}
LR_SEARCH_SAMPLES_SEEN = 3200                 # fixed budget: 100 * 32 / batch steps
HESS_FREQ           = 5
N_HUTCH              = 4
GRAD_CLIP_NORM       = 5.0
WEIGHT_DECAY         = 1e-4
TARGET_EPS_FRAC     = 0.1   # NOTE (flagged risk, spec item 21): a smaller vocab lowers
                            # init loss, which makes this 10%-of-init threshold MORE
                            # likely to trigger before `MAX_STEPS` (not less) -- so the
                            # original risk of `stepseps` saturating to max_steps is
                            # reduced, not worsened, at smaller vocab scales.
SOPHIA_BETA_H       = 0.99

LR_GRID             = [1e-4, 3e-4, 1e-3, 3e-3]
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

VOCAB_SIZE  = 1000     # top-(VOCAB_SIZE-1) by frequency + <unk>
EMBED_DIM   = 32        # fixed across both model sizes -> isolates the sweep to MLP hidden capacity
CONTEXT_K   = 4         # preceding tokens used to predict the next word
UNK_TOKEN   = "<unk>"
UNK_ID      = 0          # by construction, <unk> is always vocab index 0

# ---- Fix 1: fast (subsampled) in-loop validation eval ----
VAL_FAST_SIZE = 2000     # cap on # windows used for in-loop val eval / LR-search val scoring
VAL_FAST_SEED = 777      # fixed seed for the deterministic subsample

# ---- Fix 3: eval loaders use their own (larger) batch size, decoupled from
# whatever training batch_size the sweep point uses. This only changes how
# fast eval runs -- loss/acc/ppl are computed identically (full-batch
# averaged loss and top1acc are batch-size-invariant; only the number of
# forward-pass kernel launches changes). ----
EVAL_BATCH_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- multi-GPU setup: shard-level jobs are pinned round-robin across every
# visible CUDA device using THREADS (not processes) -- see the note above
# run_shard_sweep_multi_gpu() below for why. `device` remains the default
# device used by any code path that doesn't explicitly pick a shard-specific
# device (e.g. LR search, which still runs on a single device). ----
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEVICES = [torch.device(f"cuda:{i}") for i in range(N_GPUS)] if N_GPUS > 0 else [device]

OUT_DIR    = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH  = os.path.join(OUT_DIR, "language_batchsize2_checkpoint.json")
CKPT_BAK_PATH = CKPT_PATH + ".bak"
CSV_PATH   = os.path.join(OUT_DIR, "language_batchsize2_results_raw.csv")
JSON_PATH  = os.path.join(OUT_DIR, "language_batchsize2_results_agg.json")
LOG_PATH   = os.path.join(OUT_DIR, "language_batchsize2_log.txt")

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
#  DATA
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
_SHARD_TOKENS = {}
_VAL_TOKENS = None
_TEST_TOKENS = None

def load_corpus(max_retries=5, base_delay=2.0):
    """Corpus/dataset download with retry + exponential backoff (Fix 2.7)."""
    global _RAW, _VAL_TOKENS, _TEST_TOKENS
    if _RAW is not None:
        return
    log("[DATA] Downloading canonical WikiText-2 (train/valid/test splits)...")

    ds = None
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            ds = hf_datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
            break
        except Exception as e:
            last_exc = e
            delay = base_delay * (2 ** (attempt - 1))
            log(f"[DATA][WARN] load_dataset attempt {attempt}/{max_retries} failed: {e} "
                f"-- retrying in {delay:.1f}s")
            time.sleep(delay)
    if ds is None:
        log(f"[DATA][FATAL] could not download WikiText-2 after {max_retries} attempts.")
        raise RuntimeError(f"WikiText-2 download failed after {max_retries} attempts") from last_exc

    train_lines = list(ds["train"]["text"])
    valid_lines = list(ds["validation"]["text"])
    test_lines  = list(ds["test"]["text"])

    train_articles = _split_into_articles(train_lines)
    log(f"[DATA] train split has {len(train_articles)} articles.")

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
_VAL_WINDOWS_FAST = None   # Fix 1: fixed subsample of _VAL_WINDOWS

def _make_fast_val_windows(val_windows, cap=VAL_FAST_SIZE, seed=VAL_FAST_SEED):
    Xva, yva = val_windows
    n = len(Xva)
    if n <= cap:
        log(f"[DATA] fast-val subsample: full val size ({n}) <= cap ({cap}); using full set.")
        return Xva, yva
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=cap, replace=False)
    idx.sort()
    log(f"[DATA] fast-val subsample: {cap} windows sampled (seed={seed}) from full val "
        f"set of {n} windows. Full test-set eval at the end of each run is unaffected.")
    return Xva[idx], yva[idx]

def try_load_shard(shard_id):
    load_corpus()
    global _VAL_WINDOWS, _TEST_WINDOWS, _VAL_WINDOWS_FAST
    if _VAL_WINDOWS is None:
        _VAL_WINDOWS = _windows(_VAL_TOKENS)
        _TEST_WINDOWS = _windows(_TEST_TOKENS)
        _VAL_WINDOWS_FAST = _make_fast_val_windows(_VAL_WINDOWS)
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
    """Returns (train_loader, val_loader_fast, val_loader_full, test_loader).
    val_loader_fast is the subsampled loader used for in-loop checkpoint
    selection / regret (Fix 1). val_loader_full is kept available for any
    caller that explicitly wants the full validation split; the final
    reported metrics come from test_loader, untouched by this fix.

    Fix 3: the train_loader uses the sweep's `batch_size` (this is the
    thing being swept), but every eval loader (val_loader_fast,
    val_loader_full, test_loader) uses EVAL_BATCH_SIZE instead. Eval passes
    are done under torch.no_grad() with no optimizer state coupling to
    batch size, so this is purely a speed change -- the averaged loss and
    top1acc computed in evaluate() are identical either way, just computed
    in far fewer, far larger microbatches when the sweep's train batch_size
    is tiny (e.g. batch_size=2)."""
    Xtr, ytr = _SHARD_WINDOWS[shard_id]
    Xva_fast, yva_fast = _VAL_WINDOWS_FAST
    Xva, yva = _VAL_WINDOWS
    Xte, yte = _TEST_WINDOWS
    def ds(Xd, yd):
        return TensorDataset(torch.tensor(Xd, dtype=torch.long),
                              torch.tensor(yd, dtype=torch.long))
    pin = torch.cuda.is_available()
    return (DataLoader(ds(Xtr, ytr), batch_size=batch_size, shuffle=True, pin_memory=pin),
            DataLoader(ds(Xva_fast, yva_fast), batch_size=EVAL_BATCH_SIZE, pin_memory=pin),
            DataLoader(ds(Xva, yva), batch_size=EVAL_BATCH_SIZE, pin_memory=pin),
            DataLoader(ds(Xte, yte), batch_size=EVAL_BATCH_SIZE, pin_memory=pin))

def select_shards(n_target=N_SHARDS_TARGET, seed=0):
    load_corpus()
    shard_ids = sorted(_SHARD_TOKENS.keys())
    chosen = [s for s in shard_ids if try_load_shard(s)]
    if len(chosen) < n_target:
        log(f"[WARN] Only {len(chosen)}/{n_target} shards had enough windows.")
    return chosen[:n_target]

def stratified_tuning_subset(shards, n=N_TUNING_SHARDS):
    if len(shards) <= n:
        return list(shards)
    step = len(shards) / n
    idxs = sorted(set(int(i * step) for i in range(n)))
    return [shards[i] for i in idxs][:n]

# ==============================================================================
#  MODEL
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

    def forward(self, x):
        e = self.embed(x).flatten(1)
        return self.net(e)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_fwd_flops(model, batch_size):
    """Forward-FLOP estimator for the LM model: embedding lookup is a gather
    (no FLOPs), so the cost is the two MLP hidden layers plus the final
    vocab-projection Linear(h2, vocab_size) -- which dominates whenever
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


class RSCOOptimizer(torch.optim.Optimizer):
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
        clipped_el = None
        total_el = 0
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
                s['m1'].mul_(b1).add_(grad, alpha=1.0 - b1)
                s['m2'].mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
                s['fast_m1'].mul_(fb).add_(grad, alpha=1.0 - fb)
                s['fast_m2'].mul_(fb).addcmul_(grad, grad, value=1.0 - fb)

                m1c = s['m1'] / (1.0 - b1 ** t)
                m2c = s['m2'] / (1.0 - b2 ** t)
                sigma2 = (m2c - m1c.pow(2)).clamp_(min=0.0)
                gamma_t = g0 / (1.0 + g0 * sigma2)
                raw_wp = (1.0 - gamma_t * sigma2)
                well_pos = raw_wp.clamp(min=1e-6)
                clip_hit = (raw_wp < 1e-6).sum()
                clipped_el = clip_hit if clipped_el is None else clipped_el + clip_hit
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
        clipped_el = None
        total_el = 0
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
                clip_hit = ((raw_wp < 1e-6) | (raw_wp > wp_max)).sum()
                clipped_el = clip_hit if clipped_el is None else clipped_el + clip_hit
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
        sq_sum = None
        for p, h in param_h_pairs:
            s = self.state[p]
            h_new = h.detach().abs().clamp_(min=1e-8)
            if 'h_hat' not in s:
                s['h_hat'] = h_new.clone()
            else:
                s['h_hat'].mul_(beta_h).add_(h_new, alpha=1.0 - beta_h)
            term = h_new.pow(2).sum()
            sq_sum = term if sq_sum is None else sq_sum + term
        self.last_h_norm = sq_sum.sqrt() if sq_sum is not None else float('nan')

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
        sq_sum = None
        for p, h in param_h_pairs:
            if p not in self.state:
                self.state[p] = {}
            s = self.state[p]
            h_smooth = _spatial_avg(h.detach().abs()).clamp(min=1e-8)
            if 'h_hat' not in s:
                s['h_hat'] = h_smooth.clone()
            else:
                s['h_hat'].mul_(beta_h).add_(h_smooth, alpha=1.0 - beta_h)
            term = h_smooth.pow(2).sum()
            sq_sum = term if sq_sum is None else sq_sum + term
        self.last_h_norm = sq_sum.sqrt() if sq_sum is not None else float('nan')

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

OPT_NAMES = ["AdamW", "RSCO", "RSCO-Amplify", "Sophia", "AdaHessian"]
HESSIAN_CONSUMERS = {"Sophia", "AdaHessian"}

# ==============================================================================
#  HESSIAN ESTIMATORS
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
def reset_peak_memory(dev=None):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)

def peak_memory_mb(dev=None):
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
    try:
        proc = __import__('psutil').Process(os.getpid())
        return proc.memory_info().rss / (1024 ** 2)
    except Exception:
        return float('nan')

@torch.no_grad()
def evaluate(model, loader, criterion, dev=None):
    dev = dev or device
    model.eval()
    tot_loss, tot_n = 0.0, 0
    correct, correct_n = 0, 0
    for X, y in loader:
        X, y = X.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
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
#  LR GRID SEARCH
#
#  Fix 5: checkpointed per (opt, lr, shard) cell. `lr_search_ckpt` (a dict
#  living under bckpt["lr_search"]) accumulates {(opt, lr, shard): val_loss}
#  as each cell finishes, and save_checkpoint() is called after every cell.
#  On resume, any cell already present is skipped and its cached val_loss
#  reused -- so a timeout mid-search loses at most the one in-flight cell,
#  not the whole 4-LR x 10-shard x 5-opt search.
# ==============================================================================
def _lr_cell_key(opt_name, lr, shard):
    return f"{opt_name}|{lr:.0e}|{shard}"

def _lr_grid_search_unsafe(opt_name, tuning_shards, batch_size, hidden, n_search_steps,
                            dev=None, lr_search_ckpt=None, ckpt=None):
    dev = dev or device
    lr_search_ckpt = lr_search_ckpt if lr_search_ckpt is not None else {}
    val_by_lr = defaultdict(list)
    n_resumed = 0
    n_computed = 0
    for lr in LR_GRID:
        for shard in tuning_shards:
            cell_key = _lr_cell_key(opt_name, lr, shard)
            if cell_key in lr_search_ckpt:
                cached = lr_search_ckpt[cell_key]
                if cached is not None and not (isinstance(cached, float) and math.isnan(cached)):
                    val_by_lr[lr].append(cached)
                n_resumed += 1
                continue

            set_seed(LR_SEARCH_SEED)
            train_loader, val_loader_fast, _val_loader_full, _ = get_loaders(shard, batch_size)
            model = NPLM_MLP(hidden).to(dev)
            criterion = nn.CrossEntropyLoss()
            optimizer = build_optimizer(opt_name, model, lr)
            params = [p for p in model.parameters() if p.requires_grad]
            loader_iter = itertools.cycle(train_loader)
            for step in range(n_search_steps):
                model.train()
                X_batch, y_batch = next(loader_iter)
                X_batch, y_batch = X_batch.to(dev, non_blocking=True), y_batch.to(dev, non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                if step % HESS_FREQ == 0 and opt_name in HESSIAN_CONSUMERS:
                    hutchinson_hessian(model, params, X_batch, y_batch, criterion,
                                        optimizer=optimizer, feed_optimizer=True)
                optimizer.step()
            # Fix 1: LR-search validation scoring uses the fast (subsampled) val
            # loader -- same overhead problem as the in-loop eval, smaller scale.
            # Fix 3: that loader now also runs at EVAL_BATCH_SIZE.
            val_loss, _ = evaluate(model, val_loader_fast, criterion, dev)
            lr_search_ckpt[cell_key] = val_loss if not math.isnan(val_loss) else None
            n_computed += 1
            if ckpt is not None:
                save_checkpoint(ckpt)
            if not math.isnan(val_loss):
                val_by_lr[lr].append(val_loss)

    if n_resumed:
        log(f"  [LR-SEARCH] {opt_name}: resumed {n_resumed} cached cell(s), computed {n_computed} new cell(s).")
    median_by_lr = {lr: (float(np.median(v)) if v else float('inf'))
                     for lr, v in val_by_lr.items()}
    best_lr = min(median_by_lr, key=median_by_lr.get)
    log(f"  [LR-SEARCH] {opt_name}: medians={ {f'{k:.0e}': round(v,6) for k,v in median_by_lr.items()} } "
        f"-> selected lr={best_lr:.0e}")
    return best_lr

# ==============================================================================
#  SINGLE EXPERIMENT (one optimizer, one shard, one seed) -- takes an
#  explicit `dev` so shard-sweep jobs can be pinned round-robin across
#  every visible GPU.
# ==============================================================================
def _run_experiment_unsafe(opt_name, lr, shard, seed, batch_size,
                    hidden=MODEL_SIZE_DEFAULT, max_steps=None, dev=None):
    dev = dev or device
    set_seed(seed)
    reset_peak_memory(dev)

    train_loader, val_loader_fast, val_loader_full, test_loader = get_loaders(shard, batch_size)
    model = NPLM_MLP(hidden).to(dev)
    criterion = nn.CrossEntropyLoss()
    n_params = count_params(model)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = build_optimizer(opt_name, model, lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=lr * 0.1)

    X0, y0 = next(iter(train_loader))
    with torch.no_grad():
        init_loss = criterion(model(X0.to(dev, non_blocking=True)), y0.to(dev, non_blocking=True)).item()

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
        X_batch, y_batch = X_batch.to(dev, non_blocking=True), y_batch.to(dev, non_blocking=True)

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
            h_list = hutchinson_hessian(model, params, X_batch, y_batch, criterion,
                                         N_HUTCH, optimizer=optimizer,
                                         feed_optimizer=is_hessian_consumer,
                                         apply_spatial_avg=False)
            lhn = optimizer.last_h_norm
            if is_hessian_consumer and not (isinstance(lhn, float) and math.isnan(lhn)):
                hess_norms.append(lhn)
            else:
                hess_norms.append(math.sqrt(sum(h.pow(2).sum().item() for h in h_list)))

        step_flops = fwd_flops_per_batch * 3
        if ran_probe:
            step_flops += N_HUTCH * fwd_flops_per_batch * 3
        cumulative_flops += step_flops

        optimizer.step()
        scheduler.step()

        cc = getattr(optimizer, 'last_clip_coord_frac', float('nan'))
        if isinstance(cc, torch.Tensor) or not (isinstance(cc, float) and math.isnan(cc)):
            clip_coord_fracs.append(cc)

        if is_rsco_family and not calibrated_once and step == CALIBRATE_STEP:
            optimizer.calibrate_gamma0()
            calibrated_once = True

        lv = loss.item()

        with torch.no_grad():
            delta = torch.cat([(p.data - pb).flatten()
                                for p, pb in zip(model.parameters(), params_before)])
        del params_before

        delta_sq = delta.pow(2).sum()
        step_sizes.append(delta_sq)

        if steps_to_half is None and lv <= 0.5 * init_loss:
            steps_to_half = step
        if steps_to_eps is None and lv <= TARGET_EPS_FRAC * init_loss:
            steps_to_eps = step
        final_train_loss = lv

        if step % EVAL_EVERY == 0:
            # Fix 1: in-loop checkpoint-selection / regret eval runs on the
            # fixed fast (subsampled) validation loader, not the full split.
            # Fix 3: that loader now also runs at EVAL_BATCH_SIZE.
            val_loss, _ = evaluate(model, val_loader_fast, criterion, dev)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
            best_val_so_far = min(best_val_so_far, val_loss)
            cum_regret += max(val_loss - best_val_so_far, 0.0)
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    # Final, one-time evaluation on the FULL test set -- untouched by Fix 1,
    # sped up (not changed numerically) by Fix 3's EVAL_BATCH_SIZE.
    # This is the source of the actually-reported valloss/ppl/top1acc.
    test_loss_final, top1acc_final = evaluate(model, test_loader, criterion, dev)

    wall = time.time() - t0
    mem = peak_memory_mb(dev)
    if step_sizes:
        step_sizes_t = torch.sqrt(torch.stack(step_sizes) / max(n_params, 1))
        mean_step_size = float(step_sizes_t.mean().item())
    else:
        mean_step_size = float('nan')
    gflops = cumulative_flops / max(global_step, 1) / 1e9
    exaflops = cumulative_flops / 1e18
    clip_trigger_freq = clip_trigger_count / max(max_steps, 1)
    if clip_coord_fracs:
        if isinstance(clip_coord_fracs[0], torch.Tensor):
            clip_coord_freq = float(torch.stack(clip_coord_fracs).mean().item())
        else:
            clip_coord_freq = float(np.mean(clip_coord_fracs))
    else:
        clip_coord_freq = float('nan')

    if hess_norms:
        if isinstance(hess_norms[0], torch.Tensor):
            hn_t = torch.stack(hess_norms)
            hess_diag_norm = float(hn_t.mean().item())
            hn_std = float(hn_t.std(unbiased=False).item())
        else:
            hess_diag_norm = float(np.mean(hess_norms))
            hn_std = float(np.std(hess_norms))
        hess_diag_cv = hn_std / hess_diag_norm if hess_diag_norm > 0 else float('nan')
    else:
        hess_diag_norm = hess_diag_cv = float('nan')

    # ppl is computed over the (possibly reduced) VOCAB_SIZE output space; it's valid
    # for within-run optimizer comparisons (e.g. RSCO vs AdamW, same vocab throughout)
    # but is NOT numerically comparable across different VOCAB_SIZE settings or to
    # published WikiText-2 perplexities that use a different vocabulary size.
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
LOWER_BETTER = {
    'valloss': True, 'ppl': True, 'top1acc': False, 'steps50%': True,
    'stepseps': True, 'wall(s)': True, 'peakmemmb': True, 'gflops': True,
    'exaflops': True, 'stepsize': None, 'clipfreq%': True,
    'clipcoord%': None, 'hessnorm': None, 'hesscv': True, 'regret': True,
}

# ==============================================================================
#  CHECKPOINTING  (Fix 2: atomic + fsync + .bak rotation + corrupt fallback;
#  Fix 5: LR search cells now live under bckpt["lr_search"] and are saved
#  after every individual cell, same lock/atomic-write path as everything
#  else.)
# ==============================================================================
_ckpt_lock = threading.Lock()

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

def _default_checkpoint():
    return {"shards": None, "tuning_subset": None, "by_batch": {}, "smoke_test_done": False}

def _migrate_checkpoint(ckpt):
    """Fix 2.6: migration/setdefault pass so checkpoints written by the
    pre-fix version of this script still load without KeyErrors."""
    if not isinstance(ckpt, dict):
        return _default_checkpoint()
    ckpt.setdefault("shards", None)
    ckpt.setdefault("tuning_subset", None)
    ckpt.setdefault("by_batch", {})
    ckpt.setdefault("smoke_test_done", False)   # Fix 4
    for bkey, bckpt in ckpt["by_batch"].items():
        bckpt.setdefault("lr_used", {})
        bckpt.setdefault("lr_search", {})    # Fix 5: per-cell LR search cache
        bckpt.setdefault("completed", {})
        bckpt.setdefault("raw", {})
        # migrate old-style raw[key] = [seed0_result, seed1_result] (already
        # a list, so this is a no-op for old files) and ensure "completed"
        # entries are recomputed from how many seed results are present,
        # since the old "completed" semantics (whole job) still work fine
        # as a superset check but we don't want to trust a stale True for a
        # key that's since gained a partial-seed list under a newer run.
        for key, results in list(bckpt["raw"].items()):
            if not isinstance(results, list):
                bckpt["raw"][key] = []
    return ckpt

def load_checkpoint():
    """Fix 2.6: resilient load -- fall back to .bak if the main file is
    corrupt/unreadable; start fresh if both are unreadable."""
    for path in (CKPT_PATH, CKPT_BAK_PATH):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                ckpt = json.load(f)
            if path == CKPT_BAK_PATH:
                log(f"[CHECKPOINT][WARN] main checkpoint unreadable; loaded from .bak fallback.")
            return _migrate_checkpoint(ckpt)
        except Exception as e:
            log(f"[CHECKPOINT][WARN] failed to load {path}: {e}")
    log("[CHECKPOINT] no valid checkpoint found -- starting fresh.")
    return _default_checkpoint()

def save_checkpoint(ckpt):
    """Atomic write (tmp file + fsync + os.replace), with rotation of the
    previous good checkpoint to .bak first. Never raises -- a failed save
    logs a warning and the run continues (Fix 2.5)."""
    with _ckpt_lock:
        try:
            tmp = CKPT_PATH + f".tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(ckpt, f)
                f.flush()
                os.fsync(f.fileno())
            if os.path.exists(CKPT_PATH):
                try:
                    with open(CKPT_PATH) as f:
                        json.load(f)  # only rotate a known-good file to .bak
                    os.replace(CKPT_PATH, CKPT_BAK_PATH)
                except Exception:
                    pass  # current on-disk file already bad; don't propagate that to .bak
            os.replace(tmp, CKPT_PATH)
        except Exception as e:
            log(f"[CHECKPOINT][WARN] save_checkpoint failed: {e}")

# ---- Fix 2.8: SIGTERM/SIGINT flush handler (best-effort) ----
_shutdown_ckpt_ref = {"ckpt": None}

def _install_signal_handlers():
    def _handler(signum, frame):
        log(f"[SIGNAL] received signal {signum} -- flushing checkpoint before exit.")
        if _shutdown_ckpt_ref["ckpt"] is not None:
            save_checkpoint(_shutdown_ckpt_ref["ckpt"])
        sys.exit(1)
    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
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
# ==============================================================================
def build_batch_trend_tables(agg_by_batch_shard, opt_names, shards, batch_sizes):
    challenger = "RSCO-Amplify"

    loss_stats = {}
    for b in batch_sizes:
        loss_stats[b] = {}
        for opt in opt_names:
            vals = [agg_by_batch_shard[b][opt][s]['valloss'] for s in shards
                    if not math.isnan(agg_by_batch_shard[b][opt][s]['valloss'])]
            loss_stats[b][opt] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float('nan'), float('nan'))

    raw_p = {}
    delta_pct = {}
    for b in batch_sizes:
        c_vals, a_vals = [], []
        for s in shards:
            c = agg_by_batch_shard[b][challenger][s]['valloss']
            a = agg_by_batch_shard[b]["AdamW"][s]['valloss']
            if not (math.isnan(c) or math.isnan(a)):
                c_vals.append(c); a_vals.append(a)
        if len(c_vals) >= 10:
            try:
                _, p = wilcoxon(c_vals, a_vals)
            except ValueError:
                p = float('nan')
        else:
            p = float('nan')
        raw_p[b] = p
        if a_vals and c_vals:
            a_med, c_med = float(np.median(a_vals)), float(np.median(c_vals))
            delta_pct[b] = (a_med - c_med) / abs(a_med) * 100.0 if a_med != 0 else float('nan')
        else:
            delta_pct[b] = float('nan')
    corrected_p = holm_bonferroni(raw_p)

    return loss_stats, delta_pct, corrected_p

def print_batch_trend_tables(loss_stats, delta_pct, corrected_p, opt_names, batch_sizes):
    log_sep("=", 160)
    log("  LANGUAGE_BATCHSIZE SWEEP: valloss (mean +/- std across shards of per-shard seed-medians)")
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
    log("  RSCO-Amplify vs AdamW on valloss  (%delta, positive = RSCO-Amplify better; Holm-Bonferroni p across batch sizes)")
    log(f"  {'Batch':<8}{'%delta':>12}{'p_holm':>14}")
    for b in batch_sizes:
        d = delta_pct.get(b, float('nan'))
        p = corrected_p.get(b, float('nan'))
        log(f"  {b:<8}{d:>+11.2f}%{p:>14.3e}")
    log_sep("=", 160)

def print_full_metrics_table(agg_by_batch_shard, opt_names, shards, batch_sizes):
    """Full metric breakdown (mean+-std across shards of per-shard seed-medians),
    per optimizer, per batch size, across every entry in METRIC_KEYS -- not just
    valloss. Complements the trend tables above."""
    log_sep("=", 160)
    log("  FULL METRIC TABLE (mean +/- std across shards of per-shard seed-medians)")
    for b in batch_sizes:
        log_sep("-", 160)
        log(f"  Batch size {b}")
        log_sep("-", 160)
        for metric in METRIC_KEYS:
            header = f"  {metric:<12}"
            for opt in opt_names:
                header += f"{opt:>22}"
            log(header)
            row = f"  {'':<12}"
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
    log_sep("=", 160)

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
#  MULTI-GPU SHARD SWEEP (THREADS, not processes)
#
#  IMPORTANT: this deliberately does NOT use ProcessPoolExecutor/multiprocessing.
#  This code is meant to also run from inside a Jupyter/ipykernel cell, where
#  `__main__` is not a real importable module file. `spawn`-based process
#  pools need to re-import that module in every child process to find
#  top-level worker/initializer functions -- which is impossible from a
#  notebook `__main__`, and fails with
#      AttributeError: Can't get attribute '<...>' on <module '__main__' ...>
#  or, if it gets further, a BrokenProcessPool when a child dies trying.
#  This is not fixable by moving functions around; it's a fundamental
#  incompatibility between spawn-based multiprocessing and notebook execution.
#
#  Threads sidestep this entirely: there's no pickling of functions across a
#  process boundary, so nothing needs to be re-importable. CUDA kernels also
#  release the GIL while executing on-device, so multiple threads issuing
#  work to *different* GPUs (or overlapping small ops on the same GPU) still
#  get real concurrency where it matters, without any of the spawn/fork +
#  CUDA-context headaches process pools bring.
#
#  Checkpoint safety (Fix 2): per-SEED checkpointing now, not per-job. Each
#  worker thread loops over SEEDS itself, skips seeds already present in the
#  checkpoint, and after each individual seed completes: appends the result
#  under `_ckpt_lock`, calls save_checkpoint(ckpt) immediately, and logs
#  progress. A job's "completed" flag is only set once both seeds are
#  present. This means a hard kill mid-job loses at most one seed's worth
#  of work instead of the whole (optimizer, shard) job.
# ==============================================================================

def run_shard_sweep_multi_gpu(shards, lr_used, batch_size, full_steps, bckpt, ckpt):
    """One thread per (optimizer, shard) job, at most len(DEVICES) in flight
    at once, each job pinned to a specific GPU. Checkpoints after EACH SEED
    (Fix 2.1/2.2), so a restart resumes from the next missing seed rather
    than losing an entire in-progress job."""
    jobs = []
    for shard in shards:
        for opt in OPT_NAMES:
            key = f"{opt}|{shard}"
            existing = bckpt["raw"].get(key, [])
            if len(existing) >= len(SEEDS):
                continue
            jobs.append((opt, shard))

    if not jobs:
        return

    log(f"[SWEEP] Dispatching {len(jobs)} (optimizer, shard) jobs across "
        f"{len(DEVICES)} device(s): {[str(d) for d in DEVICES]}")

    def _shard_progress_str(shard):
        n_opt_done = sum(1 for o in OPT_NAMES
                          if len(bckpt["raw"].get(f"{o}|{shard}", [])) >= len(SEEDS))
        return f"{n_opt_done}/{len(OPT_NAMES)} optimizers done"

    def _do_job(job_idx_and_job):
        """Loops over SEEDS explicitly, skipping any seed index already
        present in the checkpoint for this key, and checkpoints after each
        individual seed (Fix 2.2/2.3)."""
        idx, (opt, shard) = job_idx_and_job
        dev = DEVICES[idx % len(DEVICES)]
        lr = lr_used[opt]
        key = f"{opt}|{shard}"

        with _ckpt_lock:
            results = list(bckpt["raw"].get(key, []))

        for seed_idx, seed in enumerate(SEEDS):
            if seed_idx < len(results):
                continue  # already have this seed's result from a prior run
            try:
                r = run_experiment(opt, lr, shard, seed, batch_size=batch_size,
                                    hidden=MODEL_SIZE_DEFAULT, max_steps=full_steps, dev=dev)
            except Exception as e:
                # run_experiment() already has its own try/except, but guard
                # the call site too (Fix 2.7) so nothing here can kill the
                # worker thread outright.
                log(f"[ERROR] {key}|seed={seed} raised unhandled exception: {e}")
                traceback.print_exc()
                r = {
                    "valloss": float("nan"), "ppl": float("nan"), "top1acc": float("nan"),
                    "steps50%": full_steps, "stepseps": full_steps,
                    "wall(s)": 0.0, "peakmemmb": 0.0, "gflops": 0.0, "exaflops": 0.0,
                    "stepsize": float("nan"), "clipfreq%": float("nan"),
                    "clipcoord%": float("nan"), "hessnorm": float("nan"),
                    "hesscv": float("nan"), "regret": float("nan"),
                    "n_params": 0, "lr_used": lr,
                }

            with _ckpt_lock:
                cur = bckpt["raw"].setdefault(key, [])
                if seed_idx < len(cur):
                    cur[seed_idx] = r
                else:
                    cur.append(r)
                results = list(cur)
                if len(cur) >= len(SEEDS):
                    bckpt["completed"][key] = True
            save_checkpoint(ckpt)
            log(f"[CHECKPOINT] {key}|seed={seed} done ({len(results)}/{len(SEEDS)} seeds for this job).")

        return opt, shard

    n_done = 0
    with ThreadPoolExecutor(max_workers=max(len(DEVICES), 1)) as ex:
        futures = {ex.submit(_do_job, (i, job)): job for i, job in enumerate(jobs)}
        for fut in _as_completed(futures):
            job = futures[fut]
            try:
                opt, shard = fut.result()
            except Exception as e:
                # Fix 2.7: one job's exception must not stop collection of
                # the others -- log and continue.
                opt, shard = job
                log(f"[ERROR] job {opt}|{shard} failed in future collection: {e}")
                traceback.print_exc()
            n_done += 1
            key = f"{opt}|{shard}"
            log(f"[CHECKPOINT] {key} job wrapper complete ({n_done}/{len(jobs)} jobs done). "
                f"shard={shard}: {_shard_progress_str(shard)}.")
            if _shard_progress_str(shard).startswith(str(len(OPT_NAMES))):
                log(f"[CHECKPOINT] shard={shard} fully complete (all {len(OPT_NAMES)} "
                    f"optimizers) -- a restart now will resume past this shard.")

def smoke_test(shards, lr_used, batch_size, n_smoke_steps=300, n_workers=None):
    """Required before trusting the timing of the full sweep: run a handful
    of concurrent (thread-based) jobs for a small number of steps and time
    it, so we get a real per-step wall-clock number at actual concurrency
    instead of a FLOP-based estimate. Uses the SAME thread-based execution
    path as the real sweep (run_shard_sweep_multi_gpu), just against a
    throwaway checkpoint dict and a couple of shards/optimizers, so if this
    passes, the real sweep's execution mechanism is validated too -- no
    separate process-pool code path to diverge from what actually runs."""
    n_workers = n_workers or max(len(DEVICES), 1)
    log_sep("=")
    log(f"  SMOKE TEST  |  batch_size={batch_size}  n_workers={n_workers}  "
        f"n_smoke_steps={n_smoke_steps}  devices={[str(d) for d in DEVICES]}")
    log_sep("=")

    test_shards = shards[:max(1, min(2, len(shards)))]
    jobs = []
    for shard in test_shards:
        for opt in OPT_NAMES[:2]:          # AdamW + RSCO is enough to smoke-test the pool
            jobs.append((opt, shard))
    jobs = jobs[:max(n_workers, 1)]

    def _do_smoke_job(job_idx_and_job):
        idx, (opt, shard) = job_idx_and_job
        dev = DEVICES[idx % len(DEVICES)]
        lr = lr_used.get(opt, 1e-3)
        r = run_experiment(opt, lr, shard, SEEDS[0], batch_size=batch_size,
                            hidden=MODEL_SIZE_DEFAULT, max_steps=n_smoke_steps, dev=dev)
        return opt, shard, r

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_do_smoke_job, (i, job)) for i, job in enumerate(jobs)]
        for fut in _as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                log(f"[SMOKE][ERROR] a smoke job failed: {e}")
                traceback.print_exc()
    wall = time.time() - t0
    per_step_wall = wall / max(n_smoke_steps, 1)

    # checkpoint round-trip sanity check (single-writer, so this is really
    # just confirming disk I/O works -- concurrency-safety here comes from
    # the lock serializing save_checkpoint calls, not from this check).
    smoke_ckpt_path = CKPT_PATH + ".smoketest"
    smoke_ckpt = {"shards": test_shards, "tuning_subset": None,
                  "by_batch": {str(batch_size): {"lr_used": lr_used, "completed": {}, "raw": {}}}}
    for opt_name, shard, r in results:
        smoke_ckpt["by_batch"][str(batch_size)]["raw"][f"{opt_name}|{shard}"] = [r]
        smoke_ckpt["by_batch"][str(batch_size)]["completed"][f"{opt_name}|{shard}"] = True
    ok = False
    try:
        tmp = smoke_ckpt_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(smoke_ckpt, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, smoke_ckpt_path)
        with open(smoke_ckpt_path) as f:
            reloaded = json.load(f)
        ok = len(reloaded["by_batch"][str(batch_size)]["raw"]) == len(results)
        os.remove(smoke_ckpt_path)
    except Exception as e:
        log(f"[SMOKE][WARN] checkpoint round-trip check failed: {e}")

    log(f"[SMOKE] {len(jobs)} concurrent jobs, {n_smoke_steps} steps each  ->  "
        f"wall={wall:.1f}s  (~{per_step_wall*1000:.1f} ms/step measured at this concurrency)")
    log(f"[SMOKE] checkpoint round-trip: {'OK' if ok else 'FAILED'}")
    if not ok:
        log("[SMOKE] WARNING: checkpoint round-trip did not contain all expected job keys.")
    log_sep("=")
    return {"wall_s": wall, "ms_per_step": per_step_wall * 1000.0, "n_workers": len(jobs),
            "checkpoint_ok": ok}


# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    log_sep("=")
    log(f"  LANGUAGE_BATCHSIZE 2 RUN  |  devices={[str(d) for d in DEVICES]}  torch={torch.__version__}")
    log(f"  batch sizes: {BATCH_SIZES}  |  full-run steps: {FULL_STEPS_BY_BATCH}")
    log(f"  vocab_size={VOCAB_SIZE}  n_shards_target={N_SHARDS_TARGET}")
    log(f"  VAL_FAST_SIZE={VAL_FAST_SIZE}  VAL_FAST_SEED={VAL_FAST_SEED}  EVAL_BATCH_SIZE={EVAL_BATCH_SIZE}  "
        f"(in-loop val eval + LR-search val scoring only; final test eval is full-set, same batch size)")
    log_sep("=")

    ckpt = load_checkpoint()
    _shutdown_ckpt_ref["ckpt"] = ckpt
    _install_signal_handlers()
    resumed_run = os.path.exists(CKPT_PATH)  # true whenever load_checkpoint found something to resume

    if ckpt["shards"] is None:
        shards = select_shards(N_SHARDS_TARGET, seed=1234)
        ckpt["shards"] = shards
        save_checkpoint(ckpt)
    else:
        shards = ckpt["shards"]
        for s in shards:
            try_load_shard(s)
    log(f"[SHARDS] Using {len(shards)} pseudo-domain shards.")
    log(f"[DATA] full val windows={len(_VAL_WINDOWS[0])}  "
        f"fast val windows={len(_VAL_WINDOWS_FAST[0])}")

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
        log(f"  LANGUAGE_BATCHSIZE {batch_size}  |  lr-search steps={n_search_steps} (samples seen={n_search_steps*batch_size})"
            f"  |  full-run steps={full_steps} (samples seen={full_steps*batch_size})")
        log_sep("=")

        bckpt = ckpt["by_batch"].setdefault(bkey, {"lr_used": {}, "lr_search": {}, "completed": {}, "raw": {}})
        bckpt.setdefault("lr_search", {})   # Fix 5: guard in case an in-memory dict predates the key

        if not bckpt["lr_used"]:
            lr_used = {}
            for opt_name in OPT_NAMES:
                # Fix 5: pass the per-batch lr_search cache + ckpt through so
                # each (opt, lr, shard) cell is checkpointed as it completes,
                # instead of the whole grid search being all-or-nothing.
                lr_used[opt_name] = lr_grid_search(opt_name, tuning_subset, batch_size,
                                                     MODEL_SIZE_DEFAULT, n_search_steps,
                                                     lr_search_ckpt=bckpt["lr_search"], ckpt=ckpt)
                bckpt["lr_used"][opt_name] = lr_used[opt_name]
                save_checkpoint(ckpt)
        else:
            lr_used = bckpt["lr_used"]
        log(f"[LR-SEARCH batch={batch_size}] Frozen LRs: {lr_used}")
        lr_used_by_batch[batch_size] = lr_used

        # ---- smoke test (spec items 11-12): required before scaling up to
        # the full worker count / full step count. Run this once per batch
        # size and eyeball the log before trusting the full sweep below.
        #
        # Fix 4: on a resumed run, skip it by default -- the timing was
        # already validated in an earlier attempt against this same
        # checkpoint, and re-running it just burns wall-clock every restart.
        # SKIP_SMOKE_TEST=1 still forces it off unconditionally (fresh or
        # resumed). Set FORCE_SMOKE_TEST=1 to force it back on even on a
        # resumed run (e.g. if you changed GPU type between restarts). ----
        force_skip = os.environ.get("SKIP_SMOKE_TEST") == "1"
        force_run = os.environ.get("FORCE_SMOKE_TEST") == "1"
        skip_as_resumed = resumed_run and ckpt.get("smoke_test_done") and not force_run
        if force_skip or skip_as_resumed:
            reason = "SKIP_SMOKE_TEST=1" if force_skip else "already run earlier in this checkpoint's lifetime"
            log(f"[SMOKE] Skipping smoke test ({reason}).")
        else:
            smoke_test(shards, lr_used, batch_size, n_smoke_steps=300, n_workers=4)
            ckpt["smoke_test_done"] = True
            save_checkpoint(ckpt)

        raw = {opt: {s: [] for s in shards} for opt in OPT_NAMES}
        for opt in OPT_NAMES:
            for s in shards:
                key = f"{opt}|{s}"
                if key in bckpt["raw"]:
                    raw[opt][s] = bckpt["raw"][key]

        total = len(shards) * len(OPT_NAMES) * len(SEEDS)
        done = sum(len(v) for opt in raw for v in raw[opt].values())
        log(f"[SWEEP batch={batch_size}] {done}/{total} runs already complete (resumed).")

        # ---- full sweep, fanned out across a thread pool (spec item 8) ----
        run_shard_sweep_multi_gpu(shards, lr_used, batch_size, full_steps, bckpt, ckpt)
        for opt in OPT_NAMES:
            for s in shards:
                key = f"{opt}|{s}"
                if key in bckpt["raw"]:
                    raw[opt][s] = bckpt["raw"][key]

        raw_by_batch[batch_size] = raw
        agg_by_batch_shard[batch_size] = {opt: {s: aggregate_shard(raw[opt][s]) for s in shards}
                                           for opt in OPT_NAMES}

    # ---- Fix 2.7: independent try/except around each output stage -- all
    # derived from data already safely on disk in the checkpoint, so a
    # failure in any one must log and continue, not crash / lose raw results.
    try:
        write_csv(raw_by_batch, OPT_NAMES, shards, SEEDS, lr_used_by_batch, BATCH_SIZES)
    except Exception as e:
        log(f"[ERROR] write_csv failed: {e}")
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
                             'eval_batch_size': EVAL_BATCH_SIZE,
                             'vocab_note': f'VOCAB_SIZE={VOCAB_SIZE} for this run; ppl figures use this '
                                           f'vocab and are NOT numerically comparable across runs with a '
                                           f'different VOCAB_SIZE, nor to published WikiText-2 perplexities '
                                           f'that use a different vocabulary size.',
                             'lr_used_by_batch': lr_used_by_batch, 'timestamp': datetime.now().isoformat()},
                'agg_by_batch_shard': agg_by_batch_shard,
            }, jf, indent=2)
        log(f"[JSON] Saved -> {JSON_PATH}")
    except Exception as e:
        log(f"[ERROR] JSON aggregate write failed: {e}")
        traceback.print_exc()

    try:
        loss_stats, delta_pct, corrected_p = build_batch_trend_tables(agg_by_batch_shard, OPT_NAMES, shards, BATCH_SIZES)
        print_batch_trend_tables(loss_stats, delta_pct, corrected_p, OPT_NAMES, BATCH_SIZES)
        print_full_metrics_table(agg_by_batch_shard, OPT_NAMES, shards, BATCH_SIZES)
    except Exception as e:
        log(f"[ERROR] summary table printing failed: {e}")
        traceback.print_exc()

    log_sep("=")
    log(f"  ALL DONE  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_sep("=")

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        log(f"[FATAL] unhandled exception in main(): {e}")
        traceback.print_exc()
        if _shutdown_ckpt_ref["ckpt"] is not None:
            log("[FATAL] flushing checkpoint one more time before re-raising.")
            save_checkpoint(_shutdown_ckpt_ref["ckpt"])
        raise
    finally:
        try:
            _log_fh.close()
        except Exception:
            pass