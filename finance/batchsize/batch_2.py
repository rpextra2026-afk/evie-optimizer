import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import subprocess, sys
def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
try:
    import yfinance as yf
except ImportError:
    _pip("yfinance")
    import yfinance as yf

import torch
import torch.nn as nn
import numpy as np
import warnings, time, math, csv, json, itertools, copy, random
from datetime import datetime
from collections import defaultdict
from scipy.stats import wilcoxon
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# ==============================================================================
#  GLOBAL CONFIG  (per spec)
# ==============================================================================
SEEDS               = [42, 43]
EVAL_EVERY           = 100
MODEL_SIZE_DEFAULT  = (64, 32)

# ---- Run-2 batch-size sweep axis ----
BATCH_SIZES         = [2]                    # batch size run: batch 2
FULL_STEPS_BY_BATCH = {2: 10000, 4: 40000, 8: 20000, 16: 10000, 64: 2500}
LR_SEARCH_SAMPLES_SEEN = 3200                # fixed budget: 100 * 32 / batch steps
HESS_FREQ           = 5
N_HUTCH             = 4
GRAD_CLIP_NORM      = 5.0
WEIGHT_DECAY        = 1e-4
TARGET_EPS_FRAC     = 0.1
SOPHIA_BETA_H       = 0.99

LR_GRID             = [1e-4, 3e-4, 1e-3, 3e-3]
LR_SEARCH_SEED      = 42
N_TUNING_TICKERS    = 10   # ~2 per sector across 5 sectors

CLIP_METRIC_THRESHOLD = 1.0
EPS_SCE              = 1e-8

TARGET_WP            = 0.85
WELL_POS_MAX          = 8.0
CALIBRATE_STEP        = 125          # in [100,150] per spec
FAST_EMA_BETA         = 0.9          # separate fast EMA used ONLY for gamma0 calibration

N_TICKERS_TARGET      = 50

ACTIVATION_FLOP_FACTOR = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR    = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH  = os.path.join(OUT_DIR, "run2_batch2_checkpoint.json")
CSV_PATH   = os.path.join(OUT_DIR, "run2_batch2_results_raw.csv")
JSON_PATH  = os.path.join(OUT_DIR, "run2_batch2_results_agg.json")
LOG_PATH   = os.path.join(OUT_DIR, "run2_batch2_log.txt")

_log_fh = open(LOG_PATH, "a", buffering=1)
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _log_fh.write(line + "\n")
def log_sep(c="=", w=140):
    log(c * w)

# ==============================================================================
#  TICKER POOL  (sector-tagged, over-provisioned so N_TICKERS_TARGET is reachable
#  even if some names are delisted / fail to download). Random draw from a large
#  candidate universe -> "any random 100 stocks" per spec.
# ==============================================================================
TICKER_POOL = {
    "Tech": ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AVGO","AMD","ORCL",
             "CRM","ADBE","CSCO","ACN","INTC","IBM","TXN","QCOM","INTU","NOW",
             "PYPL","UBER","SNOW","PANW","ADI","MU","LRCX","KLAC","APH","CDNS"],
    "Finance": ["JPM","BAC","WFC","GS","MS","BLK","AXP","SPGI","CB","PGR",
                "SCHW","C","MMC","AON","USB","PNC","TFC","AIG","MET","TRV",
                "COF","BK","STT","ICE","CME","MCO","FIS","FITB","HBAN","RF"],
    "Healthcare": ["UNH","JNJ","LLY","ABBV","MRK","PFE","TMO","ABT","DHR","MDT",
                   "BMY","AMGN","GILD","CI","ELV","ISRG","VRTX","REGN","ZTS","BSX",
                   "SYK","BDX","HCA","CVS","MCK","EW","IDXX","IQV","A","DXCM"],
    "Energy": ["XOM","CVX","COP","SLB","EOG","PSX","VLO","MPC","OXY","HAL",
               "WMB","KMI","OKE","BKR","FANG","DVN","HES","TRGP","CTRA","APA"],
    "Consumer_Industrials": ["PG","KO","PEP","WMT","HD","MCD","NKE","BA","CAT","HON",
                              "UPS","RTX","LMT","DE","GE","MMM","UNP","LOW","TGT","SBUX",
                              "COST","CL","KMB","MO","PM","DIS","CMCSA","VZ","T","TMUS"],
}
SECTOR_OF = {t: s for s, lst in TICKER_POOL.items() for t in lst}
ALL_CANDIDATES = [t for lst in TICKER_POOL.values() for t in lst]

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================================================================
#  DATA  (chronological 80/10/10, no leakage: scaler fit on TRAIN split only)
# ==============================================================================
_DATA_CACHE = {}
_FAILED_TICKERS = set()

def try_load_ticker(ticker):
    """Downloads + validates a ticker. Returns True on success, False if it
    should be skipped (delisted / empty / too short / bad data)."""
    if ticker in _DATA_CACHE:
        return True
    if ticker in _FAILED_TICKERS:
        return False
    try:
        df = yf.download(ticker, start="2010-01-01", end="2025-01-01",
                          progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty or len(df) < 500:
            raise ValueError(f"insufficient data ({0 if df is None else len(df)} rows)")
        ohlcv = df[['Open','High','Low','Close','Volume']].values.astype(np.float64)
        close = ohlcv[:, 3]
        ratio = close[1:] / close[:-1]
        if not np.isfinite(ratio).any():
            raise ValueError("no finite returns")
        n_bad = int(np.sum((ratio <= 1e-8) | ~np.isfinite(ratio)))
        log(f"    [WARN] {ticker}: clamp on {n_bad} rows") if n_bad > 0 else None
        log_ret = np.log(np.maximum(ratio, 1e-8))
        X, y = ohlcv[:-1], log_ret
        n = len(X)
        if n < 300:
            raise ValueError("too few usable rows after alignment")
        n_tr, n_va = int(n * 0.8), int(n * 0.1)
        X_tr, y_tr = X[:n_tr], y[:n_tr]
        X_va, y_va = X[n_tr:n_tr+n_va], y[n_tr:n_tr+n_va]
        X_te, y_te = X[n_tr+n_va:], y[n_tr+n_va:]
        if min(len(X_tr), len(X_va), len(X_te)) < 20:
            raise ValueError("split too small")
        mu  = X_tr.mean(0, keepdims=True)          # fit scaler on TRAIN ONLY -> no leakage
        std = X_tr.std(0, keepdims=True) + 1e-8
        X_tr = (X_tr - mu) / std
        X_va = (X_va - mu) / std
        X_te = (X_te - mu) / std
        _DATA_CACHE[ticker] = (X_tr, y_tr, X_va, y_va, X_te, y_te)
        return True
    except Exception as e:
        log(f"    [SKIP] {ticker}: {e}")
        _FAILED_TICKERS.add(ticker)
        return False

def get_loaders(ticker, batch_size):
    X_tr, y_tr, X_va, y_va, X_te, y_te = _DATA_CACHE[ticker]
    def ds(Xd, yd):
        return TensorDataset(torch.tensor(Xd, dtype=torch.float32),
                              torch.tensor(yd, dtype=torch.float32).unsqueeze(1))
    return (DataLoader(ds(X_tr, y_tr), batch_size=batch_size, shuffle=True),
            DataLoader(ds(X_va, y_va), batch_size=batch_size),
            DataLoader(ds(X_te, y_te), batch_size=batch_size))

def select_tickers(n_target=N_TICKERS_TARGET, seed=0):
    """Randomly samples candidates (any random stocks, sector tag is only used
    for LR-search stratification) and downloads until n_target valid tickers
    are collected, skipping delisted / failed names."""
    rng = random.Random(seed)
    pool = ALL_CANDIDATES[:]
    rng.shuffle(pool)
    chosen = []
    for t in pool:
        if len(chosen) >= n_target:
            break
        if try_load_ticker(t):
            chosen.append(t)
    if len(chosen) < n_target:
        log(f"[WARN] Only found {len(chosen)}/{n_target} valid tickers from candidate pool.")
    return chosen

def stratified_tuning_subset(tickers, n=N_TUNING_TICKERS):
    """~2 per sector, inferred from block membership, drawn from the tickers
    actually available in this run."""
    by_sector = defaultdict(list)
    for t in tickers:
        by_sector[SECTOR_OF.get(t, "Unknown")].append(t)
    subset = []
    sectors = list(by_sector.keys())
    per_sector = max(1, n // max(len(sectors), 1))
    for s in sectors:
        subset.extend(by_sector[s][:per_sector])
    # top up / trim to exactly n if possible
    if len(subset) < n:
        remaining = [t for t in tickers if t not in subset]
        subset.extend(remaining[: n - len(subset)])
    return subset[:n]

# ==============================================================================
#  MODEL
# ==============================================================================
class FinanceMLP(nn.Module):
    def __init__(self, hidden=(64, 32)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(5, h1), nn.GELU(),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_fwd_flops(model, batch_size):
    flops = 0
    for m in model.modules():
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
    state used in the actual update rule."""
    def __init__(self, params, lr=1e-3, gamma0=1.0, target_wp=TARGET_WP,
                 beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                 fast_beta=FAST_EMA_BETA):
        defaults = dict(lr=lr, gamma0=gamma0, target_wp=target_wp,
                         beta1=beta1, beta2=beta2, eps=eps,
                         weight_decay=weight_decay, calibrated=False,
                         fast_beta=fast_beta)
        super().__init__(params, defaults)

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
                well_pos = (1.0 - gamma_t * sigma2).clamp_(min=1e-6)
                denom = m2c.sqrt().clamp_(min=eps)
                p.data.addcdiv_(m1c.mul(well_pos.sqrt()), denom, value=-lr)
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
                raw_wp = 1.0 / (1.0 + gamma_t * sigma2)
                well_pos = (raw_wp / target).clamp_(min=1e-6, max=wp_max)
                denom = m2c.sqrt().clamp_(min=eps)
                p.data.addcdiv_(m1c.mul(well_pos.sqrt()), denom, value=-lr)
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

OPT_NAMES = ["AdamW", "RSCO", "RSCO-Amplify", "Sophia", "AdaHessian"]
HESSIAN_CONSUMERS = {"Sophia", "AdaHessian"}

# ==============================================================================
#  HESSIAN ESTIMATORS  (Hutchinson diagonal, consumed only by Sophia &
#  AdaHessian every HESS_FREQ steps. No passive probe for AdamW/RSCO/
#  RSCO-Amplify — hess_diag_norm/hess_diag_cv/sce are nan for those three.)
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
# RSCO-Amplify) — one Hessian estimator (Hutchinson) for all 5 optimizers.

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
    model.eval()
    tot_loss, tot_n = 0.0, 0
    preds_all, tgts_all = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        p = model(X)
        tot_loss += criterion(p, y).item() * X.size(0)
        tot_n += X.size(0)
        preds_all.append(p.cpu()); tgts_all.append(y.cpu())
    mse = tot_loss / max(tot_n, 1)
    pred, tgt = torch.cat(preds_all), torch.cat(tgts_all)
    mask = tgt.abs() > 1e-9
    dacc = ((pred[mask] * tgt[mask]) > 0).float().mean().item() if mask.sum() > 0 else float('nan')
    return mse, dacc

# ==============================================================================
#  LR GRID SEARCH  — stratified tuning subset, ALL 5 optimizers, frozen after
# ==============================================================================
def _lr_grid_search_unsafe(opt_name, tuning_tickers, batch_size, hidden, n_search_steps):
    val_by_lr = defaultdict(list)
    for lr in LR_GRID:
        for ticker in tuning_tickers:
            set_seed(LR_SEARCH_SEED)
            train_loader, val_loader, _ = get_loaders(ticker, batch_size)
            model = FinanceMLP(hidden).to(device)
            criterion = nn.MSELoss()
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
                    # Sophia and AdaHessian both consume the SAME Hutchinson
                    # diagonal estimator (no GNB — Hutchinson only, per spec).
                    hutchinson_hessian(model, params, X_batch, y_batch, criterion,
                                        optimizer=optimizer, feed_optimizer=True)
                optimizer.step()
            val_mse, _ = evaluate(model, val_loader, criterion)
            if not math.isnan(val_mse):
                val_by_lr[lr].append(val_mse)
    median_by_lr = {lr: (float(np.median(v)) if v else float('inf'))
                     for lr, v in val_by_lr.items()}
    best_lr = min(median_by_lr, key=median_by_lr.get)
    log(f"  [LR-SEARCH] {opt_name}: medians={ {f'{k:.0e}': round(v,6) for k,v in median_by_lr.items()} } "
        f"-> selected lr={best_lr:.0e}")
    return best_lr

# ==============================================================================
#  SINGLE EXPERIMENT (one optimizer, one ticker, one seed)
# ==============================================================================
def _run_experiment_unsafe(opt_name, lr, ticker, seed, batch_size,
                    hidden=MODEL_SIZE_DEFAULT, max_steps=None):
    set_seed(seed)
    reset_peak_memory()

    train_loader, val_loader, test_loader = get_loaders(ticker, batch_size)
    model = FinanceMLP(hidden).to(device)
    criterion = nn.MSELoss()
    n_params = count_params(model)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = build_optimizer(opt_name, model, lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=lr * 0.1)

    X0, y0 = next(iter(train_loader))
    with torch.no_grad():
        init_loss = criterion(model(X0.to(device)), y0.to(device)).item()

    tvi_num_sum, tvi_den_sum = 0.0, 0.0
    prev_delta = None
    step_sizes = []
    steps_to_half = steps_to_eps = None
    clip_trigger_count = 0
    hess_norms = []
    cum_regret = 0.0
    final_train_loss = init_loss
    qf_sum, qf_count = 0.0, 0
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

        ran_probe = (step % HESS_FREQ == 0) and is_hessian_consumer
        h_list = None
        if ran_probe:
            # Hessian estimator (Hutchinson diagonal) now runs ONLY for the
            # two optimizers that actually consume it (Sophia, AdaHessian).
            # AdamW/RSCO/RSCO-Amplify no longer get a passive probe, so
            # hess_diag_norm/hess_diag_cv/sce will be nan for those three.
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

        if is_rsco_family and not calibrated_once and step == CALIBRATE_STEP:
            optimizer.calibrate_gamma0()
            calibrated_once = True

        lv = loss.item()

        with torch.no_grad():
            delta = torch.cat([(p.data - pb).flatten()
                                for p, pb in zip(model.parameters(), params_before)])
        del params_before

        delta_sq = delta.pow(2).sum().item()
        tvi_den_sum += delta_sq
        if prev_delta is not None:
            tvi_num_sum += (delta - prev_delta).pow(2).sum().item()
        prev_delta = delta
        step_sizes.append(math.sqrt(delta_sq / max(n_params, 1)))   # RMS step size

        if h_list is not None:
            offset, qf = 0, 0.0
            for h in h_list:
                n_el = h.numel()
                u_slice = delta[offset:offset + n_el]
                qf += (h.reshape(-1) * u_slice.pow(2)).sum().item()
                offset += n_el
            qf_sum += qf; qf_count += 1

        if steps_to_half is None and lv <= 0.5 * init_loss:
            steps_to_half = step
        if steps_to_eps is None and lv <= TARGET_EPS_FRAC * init_loss:
            steps_to_eps = step
        final_train_loss = lv

        if step % EVAL_EVERY == 0:
            val_mse, _ = evaluate(model, val_loader, criterion)
            if val_mse < best_val:
                best_val = val_mse
                best_state = copy.deepcopy(model.state_dict())
            best_val_so_far = min(best_val_so_far, val_mse)
            cum_regret += max(val_mse - best_val_so_far, 0.0)
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mse, test_dacc = evaluate(model, test_loader, criterion)

    wall = time.time() - t0
    mem = peak_memory_mb()
    tvi = tvi_num_sum / max(tvi_den_sum, 1e-12) if global_step >= 2 else float('nan')
    mean_step_size = float(np.mean(step_sizes)) if step_sizes else float('nan')
    gflops = cumulative_flops / max(global_step, 1) / 1e9   # per-step GFLOPs
    exaflops = cumulative_flops / 1e18
    clip_trigger_freq = clip_trigger_count / max(max_steps, 1)

    if hess_norms:
        hess_diag_norm = float(np.mean(hess_norms))
        hn_std = float(np.std(hess_norms))
        hess_diag_cv = hn_std / hess_diag_norm if hess_diag_norm > 0 else float('nan')
    else:
        hess_diag_norm = hess_diag_cv = float('nan')

    if qf_count > 0:
        mean_qf = (qf_sum / qf_count) / max(n_params, 1)
        delta_loss = init_loss - final_train_loss
        sce = delta_loss / max(mean_qf, EPS_SCE)
    else:
        sce = float('nan')

    return {
        'test_mse': test_mse, 'test_rmse': math.sqrt(max(test_mse, 0.0)),
        'dir_acc': test_dacc,
        'steps_half': steps_to_half if steps_to_half is not None else max_steps,
        'steps_eps': steps_to_eps if steps_to_eps is not None else max_steps,
        'wall_s': wall, 'peak_mem_mb': mem,
        'gflops': gflops, 'exaflops': exaflops,
        'mean_step_sz': mean_step_size,
        'clip_trigger_freq': clip_trigger_freq,
        'hess_diag_norm': hess_diag_norm, 'hess_diag_cv': hess_diag_cv,
        'regret': cum_regret, 'tvi': tvi, 'sce': sce,
        'n_params': n_params, 'lr_used': lr,
    }

METRIC_KEYS = ['test_mse','test_rmse','dir_acc','steps_half','steps_eps','wall_s',
               'peak_mem_mb','gflops','exaflops','mean_step_sz','clip_trigger_freq',
               'hess_diag_norm','hess_diag_cv','regret','tvi','sce']
# direction: True = lower is better
LOWER_BETTER = {
    'test_mse': True, 'test_rmse': True, 'dir_acc': False, 'steps_half': True,
    'steps_eps': True, 'wall_s': True, 'peak_mem_mb': True, 'gflops': True,
    'exaflops': True, 'mean_step_sz': None, 'clip_trigger_freq': True,
    'hess_diag_norm': None, 'hess_diag_cv': True, 'regret': True,
    'tvi': True, 'sce': None,
}

# ==============================================================================
#  CHECKPOINTING
# ==============================================================================
def lr_grid_search(*args, **kwargs):
    try:
        return _lr_grid_search_unsafe(*args, **kwargs)
    except Exception as e:
        log(f"[ERROR] lr_grid_search failed: {e}")
        import traceback; traceback.print_exc()
        return 1e-3

def run_experiment(*args, **kwargs):
    try:
        return _run_experiment_unsafe(*args, **kwargs)
    except Exception as e:
        log(f"[ERROR] run_experiment failed: {e}")
        import traceback; traceback.print_exc()
        return {
            'test_mse': float('nan'), 'test_rmse': float('nan'),
            'dir_acc': float('nan'),
            'steps_half': kwargs.get('max_steps', 0),
            'steps_eps': kwargs.get('max_steps', 0),
            'wall_s': 0.0, 'peak_mem_mb': 0.0,
            'gflops': 0.0, 'exaflops': 0.0,
            'mean_step_sz': float('nan'),
            'clip_trigger_freq': float('nan'),
            'hess_diag_norm': float('nan'), 'hess_diag_cv': float('nan'),
            'regret': float('nan'), 'tvi': float('nan'), 'sce': float('nan'),
            'n_params': 0, 'lr_used': args[1] if len(args) > 1 else 1e-3,
        }

def load_checkpoint():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {"tickers": None, "tuning_subset": None, "by_batch": {}}

def save_checkpoint(ckpt):
    tmp = CKPT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    os.replace(tmp, CKPT_PATH)

# ==============================================================================
#  AGGREGATION / STATS
# ==============================================================================
def aggregate_ticker(seed_results):
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

def cross_ticker_score(agg_by_ticker, opt_names, tickers):
    """'few-shot%' column = cross-ticker generalization score:
    (1 - mean_test_mse / worst_optimizer_mean_test_mse) * 100"""
    means = {}
    for opt in opt_names:
        vals = [agg_by_ticker[opt][t]['test_mse'] for t in tickers
                 if not math.isnan(agg_by_ticker[opt][t]['test_mse'])]
        means[opt] = float(np.mean(vals)) if vals else float('nan')
    worst = max((v for v in means.values() if not math.isnan(v)), default=1.0)
    return {opt: ((1.0 - m / worst) * 100.0 if (not math.isnan(m) and worst > 0) else float('nan'))
            for opt, m in means.items()}

# ==============================================================================
#  BATCH-SIZE TREND TABLES
#  (a) test_mse mean+-std, rows=batch size, cols=optimizer
#  (b) RSCO-Amplify vs AdamW %delta (+ p_holm, corrected jointly across the
#      batch-size rows) on test_mse, rows=batch size
# ==============================================================================
def build_batch_trend_tables(agg_by_batch_ticker, opt_names, tickers, batch_sizes):
    challenger = "RSCO-Amplify"

    # (a) mean+-std test_mse per (batch, opt), across tickers of seed-medians
    mse_stats = {}
    for b in batch_sizes:
        mse_stats[b] = {}
        for opt in opt_names:
            vals = [agg_by_batch_ticker[b][opt][t]['test_mse'] for t in tickers
                    if not math.isnan(agg_by_batch_ticker[b][opt][t]['test_mse'])]
            mse_stats[b][opt] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float('nan'), float('nan'))

    # (b) RSCO-Amplify vs AdamW %delta on test_mse (lower is better) + Wilcoxon,
    # Holm-Bonferroni corrected jointly across all batch-size rows
    raw_p = {}
    delta_pct = {}
    for b in batch_sizes:
        c_vals, b_vals = [], []
        for t in tickers:
            c = agg_by_batch_ticker[b][challenger][t]['test_mse']
            a = agg_by_batch_ticker[b]["AdamW"][t]['test_mse']
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

    return mse_stats, delta_pct, corrected_p

def print_batch_trend_tables(mse_stats, delta_pct, corrected_p, opt_names, batch_sizes):
    log_sep("=", 160)
    log("  RUN 2 — BATCH-SIZE SWEEP: test_mse (mean +/- std across tickers of per-ticker seed-medians)")
    log_sep("-", 160)
    header = f"  {'Batch':<8}"
    for opt in opt_names:
        header += f"{opt:>22}"
    log(header)
    log("  " + "-" * (8 + 22 * len(opt_names)))
    for b in batch_sizes:
        row = f"  {b:<8}"
        for opt in opt_names:
            m, sd = mse_stats[b][opt]
            cell = f"{m:.4g}+/-{sd:.2g}" if not math.isnan(m) else "nan"
            row += f"{cell:>22}"
        log(row)
    log_sep("-", 160)
    log("  RSCO-Amplify vs AdamW on test_mse  (%delta, positive = RSCO-Amplify better; Holm-Bonferroni p across batch sizes)")
    log(f"  {'Batch':<8}{'%delta':>12}{'p_holm':>14}")
    for b in batch_sizes:
        d = delta_pct.get(b, float('nan'))
        p = corrected_p.get(b, float('nan'))
        log(f"  {b:<8}{d:>+11.2f}%{p:>14.3e}")
    log_sep("=", 160)

# ==============================================================================
#  CSV
# ==============================================================================
def write_csv(raw_by_batch_opt_ticker, opt_names, tickers, seeds, lr_used_by_batch, batch_sizes):
    fieldnames = ['batch_size', 'opt', 'ticker', 'seed', 'lr_used'] + METRIC_KEYS
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b in batch_sizes:
            for opt in opt_names:
                for ticker in tickers:
                    runs = raw_by_batch_opt_ticker.get(b, {}).get(opt, {}).get(ticker, [])
                    for i, seed in enumerate(seeds):
                        if i < len(runs):
                            row = {'batch_size': b, 'opt': opt, 'ticker': ticker, 'seed': seed,
                                   'lr_used': lr_used_by_batch.get(b, {}).get(opt, '')}
                            row.update({k: runs[i].get(k, float('nan')) for k in METRIC_KEYS})
                            writer.writerow(row)
    log(f"[CSV] Saved -> {CSV_PATH}")

# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    log_sep("=")
    log(f"  RUN 2 — BATCH-SIZE SWEEP  |  device={device}  torch={torch.__version__}")
    log(f"  batch sizes: {BATCH_SIZES}  |  full-run steps: {FULL_STEPS_BY_BATCH}")
    log_sep("=")

    ckpt = load_checkpoint()

    # ---- 1. ticker selection (same 100 tickers as Run 1, resume if checkpointed) ----
    if ckpt["tickers"] is None:
        tickers = select_tickers(N_TICKERS_TARGET, seed=1234)
        ckpt["tickers"] = tickers
        save_checkpoint(ckpt)
    else:
        tickers = ckpt["tickers"]
        for t in tickers:            # re-warm data cache
            try_load_ticker(t)
    log(f"[TICKERS] Using {len(tickers)} tickers.")

    # ---- 2. fixed tuning subset, same across every batch size ----
    if ckpt["tuning_subset"] is None:
        tuning_subset = stratified_tuning_subset(tickers, N_TUNING_TICKERS)
        ckpt["tuning_subset"] = tuning_subset
        save_checkpoint(ckpt)
    else:
        tuning_subset = ckpt["tuning_subset"]
    log(f"[TUNING SUBSET] ({len(tuning_subset)}, fixed across all batch sizes): {tuning_subset}")

    raw_by_batch = {}
    lr_used_by_batch = {}
    agg_by_batch_ticker = {}

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
                try:
                    lr_used[opt_name] = lr_grid_search(opt_name, tuning_subset, batch_size,
                                                         MODEL_SIZE_DEFAULT, n_search_steps)
                except Exception as e:
                    log(f"[ERROR] lr_grid_search({opt_name}) crashed: {e}; using 1e-3")
                    lr_used[opt_name] = 1e-3
            bckpt["lr_used"] = lr_used
            save_checkpoint(ckpt)
        else:
            lr_used = bckpt["lr_used"]
        log(f"[LR-SEARCH batch={batch_size}] Frozen LRs: {lr_used}")
        lr_used_by_batch[batch_size] = lr_used

        # ---- 2b. full sweep at this batch size, checkpointed per (opt,ticker) ----
        raw = {opt: {t: [] for t in tickers} for opt in OPT_NAMES}
        for opt in OPT_NAMES:
            for t in tickers:
                key = f"{opt}|{t}"
                if key in bckpt["raw"]:
                    raw[opt][t] = bckpt["raw"][key]

        total = len(tickers) * len(OPT_NAMES) * len(SEEDS)
        done = sum(len(v) for opt in raw for v in raw[opt].values())
        log(f"[SWEEP batch={batch_size}] {done}/{total} runs already complete (resumed).")

        for t_idx, ticker in enumerate(tickers):
            for opt in OPT_NAMES:
                key = f"{opt}|{ticker}"
                if key in bckpt["completed"]:
                    continue
                lr = lr_used[opt]
                results = []
                for seed in SEEDS:
                    try:
                        r = run_experiment(opt, lr, ticker, seed,
                                            batch_size=batch_size, hidden=MODEL_SIZE_DEFAULT,
                                            max_steps=full_steps)
                    except Exception as e:
                        log(f"[ERROR] run_experiment({opt}, ticker={ticker}, seed={seed}) crashed: {e}")
                        import traceback; traceback.print_exc()
                        r = {
                            'test_mse': float('nan'), 'test_rmse': float('nan'),
                            'dir_acc': float('nan'),
                            'steps_half': full_steps, 'steps_eps': full_steps,
                            'wall_s': 0.0, 'peak_mem_mb': 0.0,
                            'gflops': 0.0, 'exaflops': 0.0,
                            'mean_step_sz': float('nan'),
                            'clip_trigger_freq': float('nan'),
                            'hess_diag_norm': float('nan'), 'hess_diag_cv': float('nan'),
                            'regret': float('nan'), 'tvi': float('nan'), 'sce': float('nan'),
                            'n_params': 0, 'lr_used': lr,
                        }
                    results.append(r)
                raw[opt][ticker] = results
                bckpt["raw"][key] = results
                bckpt["completed"][key] = True
                save_checkpoint(ckpt)
            log(f"[PROGRESS batch={batch_size} {t_idx+1}/{len(tickers)}] {ticker} done (all optimizers).")

        raw_by_batch[batch_size] = raw
        agg_by_batch_ticker[batch_size] = {opt: {t: aggregate_ticker(raw[opt][t]) for t in tickers}
                                            for opt in OPT_NAMES}

    # ---- 3. aggregate + report across all batch sizes ----
    write_csv(raw_by_batch, OPT_NAMES, tickers, SEEDS, lr_used_by_batch, BATCH_SIZES)
    with open(JSON_PATH, 'w') as jf:
        json.dump({
            'metadata': {'tickers': tickers, 'seeds': SEEDS, 'batch_sizes': BATCH_SIZES,
                         'full_steps_by_batch': FULL_STEPS_BY_BATCH,
                         'lr_search_samples_seen': LR_SEARCH_SAMPLES_SEEN,
                         'model_hidden': MODEL_SIZE_DEFAULT, 'tuning_subset': tuning_subset,
                         'lr_used_by_batch': lr_used_by_batch, 'timestamp': datetime.now().isoformat()},
            'agg_by_batch_ticker': agg_by_batch_ticker,
        }, jf, indent=2)
    log(f"[JSON] Saved -> {JSON_PATH}")

    mse_stats, delta_pct, corrected_p = build_batch_trend_tables(agg_by_batch_ticker, OPT_NAMES, tickers, BATCH_SIZES)
    print_batch_trend_tables(mse_stats, delta_pct, corrected_p, OPT_NAMES, BATCH_SIZES)

    log_sep("=")
    log(f"  ALL DONE  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_sep("=")
    _log_fh.close()

if __name__ == '__main__':
    main()