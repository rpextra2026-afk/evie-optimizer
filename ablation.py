import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import subprocess, sys
def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import yfinance as yf
except ImportError:
    _pip("yfinance"); import yfinance as yf
try:
    import datasets as hf_datasets
except ImportError:
    _pip("datasets"); import datasets as hf_datasets
try:
    import torchvision
except ImportError:
    _pip("torchvision"); import torchvision

import torch
import torch.nn as nn
import numpy as np
import warnings, time, math, csv, json, itertools, copy, random, re
from datetime import datetime
from collections import defaultdict
from scipy.stats import wilcoxon
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# ==============================================================================
#  GLOBAL CONFIG (unchanged constants from the original file, except EVAL_SEEDS)
# ==============================================================================
CALIBRATION_SEED     = 0                                   # disjoint from eval seeds
EVAL_SEEDS           = [42, 43, 44]                         # << reduced from 10 to 3, per instruction
LR_SEARCH_SEED       = 1234                                 # disjoint from EVAL_SEEDS
assert LR_SEARCH_SEED not in EVAL_SEEDS, "LR_SEARCH_SEED must be disjoint from EVAL_SEEDS"
assert CALIBRATION_SEED not in EVAL_SEEDS, "CALIBRATION_SEED must be disjoint from EVAL_SEEDS"

MAX_STEPS            = 5000          # finance / vision / language benchmark: identical per spec
BATCH_SIZE           = 32
GRAD_CLIP_NORM       = 5.0
WEIGHT_DECAY         = 1e-4
TARGET_EPS_FRAC      = 0.1

LR_GRID              = [1e-4, 3e-4, 1e-3, 3e-3]
CALIBRATE_STEP       = 125
LR_SEARCH_STEPS      = CALIBRATE_STEP + 1
N_TUNING_TICKERS     = 10
N_TUNING_SHARDS      = 10             # language tuning uses all 10 canonical shards, per spec

# domain-specific eval cadence, per spec ("EVAL_EVERY differs by domain")
EVAL_EVERY_BY_DOMAIN = {"finance": 100, "vision": 100, "language": 250}

TARGET_WP            = 0.85
WELL_POS_MAX         = 8.0
FAST_EMA_BETA        = 0.9
EPS_SCE              = 1e-8

# vision constants (spec: PART V1)
VISION_SPLIT_SEEDS   = [2024, 2025, 2026]     # the 3 "paired task instances" (NOT 3 datasets)
VISION_MODEL_WIDTH   = (32, 64)               # benchmark default width, per spec
VISION_NUM_CLASSES   = 100

# language constants (spec: PART L1)
LANGUAGE_SHARD_IDS   = [0, 1, 2]              # 3 of the 10 canonical deterministic shards
LANGUAGE_N_SHARDS    = 10
LANGUAGE_VOCAB_SIZE  = 5000
LANGUAGE_CONTEXT_K   = 4
LANGUAGE_EMBED_DIM   = 32
LANGUAGE_MODEL_HIDDEN = (64, 32)              # benchmark default hidden dims, per spec

# finance constants (unchanged from original file)
MODEL_HIDDEN         = (64, 32)
N_TICKERS_TASK        = 3                     # << reduced from 100 to 3, per instruction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH        = os.path.join(OUT_DIR, "ablation_checkpoint.json")
RAW_CSV_PATH     = os.path.join(OUT_DIR, "raw_results.csv")
PER_TASK_CSV     = os.path.join(OUT_DIR, "per_task_summary.csv")
ABLATION_CSV     = os.path.join(OUT_DIR, "ablation_summary.csv")
ABLATION_TEX     = os.path.join(OUT_DIR, "ablation_summary.tex")
DIAG_PATH        = os.path.join(OUT_DIR, "mechanism_diagnostics.csv")
FIGURE_PATH      = os.path.join(OUT_DIR, "paired_improvement_figure.png")
APPENDIX_TABLE   = os.path.join(OUT_DIR, "appendix_table.md")
CALIB_PATH       = os.path.join(OUT_DIR, "ablation_calibration.json")
UNITTEST_PATH    = os.path.join(OUT_DIR, "ablation_unit_tests.json")
LOG_PATH         = os.path.join(OUT_DIR, "ablation_log.txt")

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
#  OPTIMIZERS -- copied VERBATIM from the original stella_ablation.py.
#  Do not edit the math below this line under any circumstance.
# ==============================================================================
class RSCOOptimizer(torch.optim.Optimizer):
    """Base Stella. gamma0 auto-calibrated ONCE at step in [100,150] from a
    separate fast-EMA variance estimate. Diagnostics (gate value, sigma^2,
    gamma_t, floor-active flag) are recorded per step for later aggregation;
    this does not affect the update rule."""
    def __init__(self, params, lr=1e-3, gamma0=1.0, target_wp=TARGET_WP,
                 beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                 fast_beta=FAST_EMA_BETA, collect_diagnostics=False):
        defaults = dict(lr=lr, gamma0=gamma0, target_wp=target_wp,
                         beta1=beta1, beta2=beta2, eps=eps,
                         weight_decay=weight_decay, calibrated=False,
                         fast_beta=fast_beta)
        super().__init__(params, defaults)
        self.collect_diagnostics = collect_diagnostics
        self.diag = defaultdict(list)

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
            log(f"    [Stella] gamma0={new_g0:.1f}  fast_sigma2={mean_s2:.2e}")

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
                s['m1'].mul_(b1).add_(grad, alpha=1.0 - b1)
                s['m2'].mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
                s['fast_m1'].mul_(fb).add_(grad, alpha=1.0 - fb)
                s['fast_m2'].mul_(fb).addcmul_(grad, grad, value=1.0 - fb)

                m1c = s['m1'] / (1.0 - b1 ** t)
                m2c = s['m2'] / (1.0 - b2 ** t)
                sigma2 = (m2c - m1c.pow(2)).clamp_(min=0.0)
                gamma_t = g0 / (1.0 + g0 * sigma2)
                raw_gate_val = 1.0 - gamma_t * sigma2
                well_pos = raw_gate_val.clamp_(min=1e-6)
                if self.collect_diagnostics:
                    self.diag['sigma2'].append(sigma2.mean().item())
                    self.diag['gamma_t'].append(gamma_t.mean().item())
                    self.diag['gate'].append(well_pos.mean().item())
                    self.diag['floor_active_frac'].append(
                        (raw_gate_val < 1e-6).float().mean().item())
                denom = m2c.sqrt().clamp_(min=eps)
                p.data.addcdiv_(m1c.mul(well_pos.sqrt()), denom, value=-lr)
        return loss


class StellaFixedGamma(torch.optim.Optimizer):
    """Component-C ablation: identical gate form (B) to base Stella, but
    gamma_t is FIXED instead of recomputed every step. See original file's
    docstring for the well-posedness caveat (no analytical guarantee here)."""
    def __init__(self, params, lr=1e-3, gamma_fixed=1.0,
                 beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                 collect_diagnostics=False):
        defaults = dict(lr=lr, gamma_fixed=gamma_fixed, beta1=beta1, beta2=beta2,
                         eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.collect_diagnostics = collect_diagnostics
        self.diag = defaultdict(list)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, gf, b1, b2, eps, wd = (group['lr'], group['gamma_fixed'], group['beta1'],
                                        group['beta2'], group['eps'], group['weight_decay'])
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
                s['step'] += 1
                t = s['step']
                s['m1'].mul_(b1).add_(grad, alpha=1.0 - b1)
                s['m2'].mul_(b2).addcmul_(grad, grad, value=1.0 - b2)

                m1c = s['m1'] / (1.0 - b1 ** t)
                m2c = s['m2'] / (1.0 - b2 ** t)
                sigma2 = (m2c - m1c.pow(2)).clamp_(min=0.0)
                raw_gate_val = 1.0 - gf * sigma2
                well_pos = raw_gate_val.clamp_(min=1e-6)
                if self.collect_diagnostics:
                    self.diag['sigma2'].append(sigma2.mean().item())
                    self.diag['gamma_t'].append(gf)
                    self.diag['gate'].append(well_pos.mean().item())
                    self.diag['floor_active_frac'].append(
                        (raw_gate_val < 1e-6).float().mean().item())
                denom = m2c.sqrt().clamp_(min=eps)
                p.data.addcdiv_(m1c.mul(well_pos.sqrt()), denom, value=-lr)
        return loss


class RSCOAmplifyOptimizer(torch.optim.Optimizer):
    """Stella-Amplify, copied verbatim (math untouched)."""
    def __init__(self, params, lr=1e-3, gamma0=1.0, target_wp=TARGET_WP,
                 beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                 well_pos_max=WELL_POS_MAX, fast_beta=FAST_EMA_BETA,
                 collect_diagnostics=False):
        defaults = dict(lr=lr, gamma0=gamma0, target_wp=target_wp,
                         beta1=beta1, beta2=beta2, eps=eps,
                         weight_decay=weight_decay, calibrated=False,
                         well_pos_max=well_pos_max, fast_beta=fast_beta)
        super().__init__(params, defaults)
        self.collect_diagnostics = collect_diagnostics
        self.diag = defaultdict(list)

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
            log(f"    [Stella-Amplify] gamma0={new_g0:.1f}  fast_sigma2={mean_s2:.2e}")

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
                well_pos_raw = raw_wp / target
                well_pos = well_pos_raw.clamp(min=1e-6, max=wp_max)
                if self.collect_diagnostics:
                    self.diag['sigma2'].append(sigma2.mean().item())
                    self.diag['gamma_t'].append(gamma_t.mean().item())
                    self.diag['raw_wp'].append(raw_wp.mean().item())
                    self.diag['well_pos'].append(well_pos.mean().item())
                    self.diag['gain'].append(well_pos.sqrt().mean().item())
                    self.diag['frac_gain_gt1'].append((well_pos > 1.0).float().mean().item())
                    self.diag['wpmax_active_frac'].append(
                        (well_pos_raw > wp_max).float().mean().item())
                denom = m2c.sqrt().clamp_(min=eps)
                p.data.addcdiv_(m1c.mul(well_pos.sqrt()), denom, value=-lr)
        return loss


def build_optimizer(arm_name, model, lr, gamma_fixed_value=None, collect_diagnostics=False):
    if arm_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    elif arm_name == "Stella":
        return RSCOOptimizer(model.parameters(), lr=lr, gamma0=1.0,
                              target_wp=TARGET_WP, weight_decay=WEIGHT_DECAY,
                              collect_diagnostics=collect_diagnostics)
    elif arm_name == "Stella-FixedGamma":
        assert gamma_fixed_value is not None, "gamma_fixed_value must be calibrated first"
        return StellaFixedGamma(model.parameters(), lr=lr, gamma_fixed=gamma_fixed_value,
                                 weight_decay=WEIGHT_DECAY,
                                 collect_diagnostics=collect_diagnostics)
    elif arm_name == "Stella-Amplify":
        return RSCOAmplifyOptimizer(model.parameters(), lr=lr, gamma0=1.0,
                                     target_wp=TARGET_WP, weight_decay=WEIGHT_DECAY,
                                     well_pos_max=WELL_POS_MAX,
                                     collect_diagnostics=collect_diagnostics)
    else:
        raise ValueError(f"unknown arm {arm_name}")

ARM_NAMES = ["AdamW", "Stella", "Stella-FixedGamma", "Stella-Amplify"]
IS_STELLA_FAMILY_ADAPTIVE = {"Stella": True, "Stella-Amplify": True}

COMPARISONS = [
    ("Stella", "AdamW"),
    ("Stella-FixedGamma", "Stella"),
    ("Stella-Amplify", "Stella"),
]

# ==============================================================================
#  UNIT TESTS -- copied verbatim from the original file (pure math, domain-free)
# ==============================================================================
def run_unit_tests():
    results = {}
    torch.manual_seed(0)
    sigma2 = torch.zeros(5); gamma_t = torch.full((5,), 3.7)
    gate = (1.0 - gamma_t * sigma2).clamp(min=1e-6)
    results["1_sigma2_zero_gate_is_one"] = bool(torch.allclose(gate, torch.ones(5), atol=1e-8))

    g0 = 0.0
    sigma2 = torch.tensor([0.0, 0.5, 2.0, 100.0])
    gamma_t = g0 / (1.0 + g0 * sigma2)
    gate = (1.0 - gamma_t * sigma2).clamp(min=1e-6)
    results["2_gamma0_zero_disables_correction"] = bool(
        torch.allclose(gamma_t, torch.zeros(4), atol=1e-8) and
        torch.allclose(gate, torch.ones(4), atol=1e-8))

    g0 = 25.0
    sigma2 = torch.tensor([0.0, 0.01, 0.1, 1.0, 10.0])
    gamma_t = g0 / (1.0 + g0 * sigma2)
    raw_wp = 1.0 / (1.0 + gamma_t * sigma2)
    well_pos_target1 = raw_wp / 1.0
    results["3_target_wp_one_reduces_to_raw_gate"] = bool(torch.allclose(well_pos_target1, raw_wp))

    torch.manual_seed(1)
    g0_grid = torch.tensor([0.0, 1.0, 10.0, 25.0, 500.0, 5000.0])
    sigma2_grid = torch.tensor([0.0, 1e-6, 1e-3, 0.01, 0.1, 1.0, 10.0, 1e3, 1e6])
    target_wp = TARGET_WP; wp_max = WELL_POS_MAX
    all_raw_wp, all_well_pos, all_gain = [], [], []
    wpmax_would_bind = False
    for g0v in g0_grid.tolist():
        for s2 in sigma2_grid.tolist():
            gamma_t = g0v / (1.0 + g0v * s2)
            raw_wp = 1.0 / (1.0 + gamma_t * s2)
            well_pos_raw = raw_wp / target_wp
            all_raw_wp.append(raw_wp)
            all_well_pos.append(min(well_pos_raw, wp_max))
            all_gain.append(math.sqrt(min(well_pos_raw, wp_max)))
            if well_pos_raw > wp_max:
                wpmax_would_bind = True
    all_raw_wp = np.array(all_raw_wp); all_well_pos = np.array(all_well_pos); all_gain = np.array(all_gain)
    results["4_raw_wp_in_0_1"] = bool(((all_raw_wp > 0) & (all_raw_wp <= 1.0 + 1e-9)).all())
    results["5_well_pos_in_0_1_over_targetwp"] = bool(
        ((all_well_pos > 0) & (all_well_pos <= 1.0 / target_wp + 1e-9)).all())
    results["6_gain_le_amplification_ceiling"] = bool(
        (all_gain <= 1.0 / math.sqrt(target_wp) + 1e-9).all())
    results["7_well_pos_max_never_binds_under_valid_inputs"] = bool(not wpmax_would_bind)
    results["7_max_observed_well_pos_raw"] = float((all_raw_wp / target_wp).max())

    with open(UNITTEST_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"[UNIT TESTS] {json.dumps(results, indent=2)}")
    all_pass = all(v for k, v in results.items() if isinstance(v, bool))
    log(f"[UNIT TESTS] ALL PASS = {all_pass}")
    return results

# ==============================================================================
#  DOMAIN: FINANCE  (verbatim pipeline from the original file, budget reduced
#  to N_TICKERS_TASK=3 selected eval tickers + a disjoint tuning pool)
# ==============================================================================
FIN_TICKER_POOL = {
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
FIN_SECTOR_OF = {t: s for s, lst in FIN_TICKER_POOL.items() for t in lst}
FIN_ALL_CANDIDATES = [t for lst in FIN_TICKER_POOL.values() for t in lst]
_FIN_DATA_CACHE = {}
_FIN_FAILED = set()

class FinanceMLP(nn.Module):
    def __init__(self, hidden=MODEL_HIDDEN):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(5, h1), nn.GELU(),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

def fin_try_load_ticker(ticker):
    if ticker in _FIN_DATA_CACHE:
        return True
    if ticker in _FIN_FAILED:
        return False
    try:
        df = yf.download(ticker, start="2010-01-01", end="2025-01-01",
                          progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty or len(df) < 500:
            raise ValueError("insufficient data")
        ohlcv = df[['Open','High','Low','Close','Volume']].values.astype(np.float64)
        close = ohlcv[:, 3]
        ratio = close[1:] / close[:-1]
        if not np.isfinite(ratio).any():
            raise ValueError("no finite returns")
        log_ret = np.log(np.maximum(ratio, 1e-8))
        X, y = ohlcv[:-1], log_ret
        n = len(X)
        if n < 300:
            raise ValueError("too few usable rows")
        n_tr, n_va = int(n * 0.8), int(n * 0.1)
        X_tr, y_tr = X[:n_tr], y[:n_tr]
        X_va, y_va = X[n_tr:n_tr+n_va], y[n_tr:n_tr+n_va]
        X_te, y_te = X[n_tr+n_va:], y[n_tr+n_va:]
        if min(len(X_tr), len(X_va), len(X_te)) < 20:
            raise ValueError("split too small")
        mu = X_tr.mean(0, keepdims=True); std = X_tr.std(0, keepdims=True) + 1e-8
        X_tr = (X_tr - mu) / std; X_va = (X_va - mu) / std; X_te = (X_te - mu) / std
        _FIN_DATA_CACHE[ticker] = (X_tr, y_tr, X_va, y_va, X_te, y_te)
        return True
    except Exception as e:
        log(f"    [SKIP finance ticker {ticker}] {e}")
        _FIN_FAILED.add(ticker)
        return False

def fin_get_loaders(ticker, batch_size):
    X_tr, y_tr, X_va, y_va, X_te, y_te = _FIN_DATA_CACHE[ticker]
    def ds(Xd, yd):
        return TensorDataset(torch.tensor(Xd, dtype=torch.float32),
                              torch.tensor(yd, dtype=torch.float32).unsqueeze(1))
    return (DataLoader(ds(X_tr, y_tr), batch_size=batch_size, shuffle=True),
            DataLoader(ds(X_va, y_va), batch_size=batch_size),
            DataLoader(ds(X_te, y_te), batch_size=batch_size))

def fin_select_task_tickers(n_target=N_TICKERS_TASK, seed=1234):
    """The 3 finance eval task instances -- deterministic subset of the pool."""
    rng = random.Random(seed)
    pool = FIN_ALL_CANDIDATES[:]; rng.shuffle(pool)
    chosen = []
    for t in pool:
        if len(chosen) >= n_target:
            break
        if fin_try_load_ticker(t):
            chosen.append(t)
    if len(chosen) < n_target:
        log(f"[WARN] finance: only found {len(chosen)}/{n_target} valid task tickers.")
    return chosen

def fin_select_tuning_tickers(exclude, n=N_TUNING_TICKERS, seed=99):
    """Tuning pool for LR search, disjoint from the 3 eval tickers."""
    rng = random.Random(seed)
    pool = [t for t in FIN_ALL_CANDIDATES if t not in exclude]
    rng.shuffle(pool)
    chosen = []
    for t in pool:
        if len(chosen) >= n:
            break
        if fin_try_load_ticker(t):
            chosen.append(t)
    return chosen

# ==============================================================================
#  DOMAIN: VISION  (CIFAR-100 -> LightCNN, per spec PART V1/V2 benchmark config)
#  Only ONE dataset exists. The 3 "task instances" are paired re-splits of the
#  SAME CIFAR-100 pool via VISION_SPLIT_SEEDS -- explicitly NOT independent
#  datasets. Labeled as such in every output.
# ==============================================================================
class LightCNN(nn.Module):
    def __init__(self, width=VISION_MODEL_WIDTH, num_classes=VISION_NUM_CLASSES):
        super().__init__()
        c1, c2 = width
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.GELU()
        self.fc = nn.Linear(c2 * 8 * 8, num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)

_VISION_RAW = {"X": None, "y_fine": None}

def vision_load_raw():
    """Pools official CIFAR-100 train+test (60,000 images), per spec: split
    is a random permutation, NOT chronological, done later per SPLIT_SEED."""
    if _VISION_RAW["X"] is not None:
        return
    root = "/tmp/cifar100_data"
    tr = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    te = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    X = np.concatenate([tr.data, te.data], axis=0).astype(np.float32) / 255.0  # (60000,32,32,3)
    X = np.transpose(X, (0, 3, 1, 2))  # -> (N,3,32,32)
    y = np.array(tr.targets + te.targets, dtype=np.int64)
    _VISION_RAW["X"] = X
    _VISION_RAW["y_fine"] = y

_VISION_SPLIT_CACHE = {}

def vision_get_loaders(split_seed, batch_size):
    """80/10/10 random permutation split, per-channel z-score fit on TRAIN
    only -- exactly per spec. Cached per split_seed (= task instance id)."""
    if split_seed in _VISION_SPLIT_CACHE:
        pass
    else:
        vision_load_raw()
        X, y = _VISION_RAW["X"], _VISION_RAW["y_fine"]
        n = X.shape[0]
        perm = np.random.RandomState(seed=split_seed).permutation(n)
        n_tr, n_va = int(n * 0.8), int(n * 0.1)
        tr_idx, va_idx, te_idx = perm[:n_tr], perm[n_tr:n_tr+n_va], perm[n_tr+n_va:]
        mu = X[tr_idx].mean(axis=(0, 2, 3), keepdims=True)
        std = X[tr_idx].std(axis=(0, 2, 3), keepdims=True) + 1e-8
        Xn = (X - mu) / std
        _VISION_SPLIT_CACHE[split_seed] = (
            Xn[tr_idx], y[tr_idx], Xn[va_idx], y[va_idx], Xn[te_idx], y[te_idx])
    X_tr, y_tr, X_va, y_va, X_te, y_te = _VISION_SPLIT_CACHE[split_seed]
    def ds(Xd, yd):
        return TensorDataset(torch.tensor(Xd, dtype=torch.float32),
                              torch.tensor(yd, dtype=torch.long))
    return (DataLoader(ds(X_tr, y_tr), batch_size=batch_size, shuffle=True),
            DataLoader(ds(X_va, y_va), batch_size=batch_size),
            DataLoader(ds(X_te, y_te), batch_size=batch_size))

VISION_TUNING_SPLIT_SEED = 7777  # held out, disjoint from the 3 task-instance split seeds

# ==============================================================================
#  DOMAIN: LANGUAGE (WikiText-2 -> NPLM-MLP, per spec PART L1 benchmark config)
#  10 canonical deterministic shards exist; we use 3 of them (shard_id 0,1,2)
#  as the eval task instances, and the remaining 7 for LR-search tuning.
# ==============================================================================
class NPLM_MLP(nn.Module):
    def __init__(self, vocab_size=LANGUAGE_VOCAB_SIZE, embed_dim=LANGUAGE_EMBED_DIM,
                 k=LANGUAGE_CONTEXT_K, hidden=LANGUAGE_MODEL_HIDDEN):
        super().__init__()
        h1, h2 = hidden
        self.embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.net = nn.Sequential(
            nn.Linear(k * embed_dim, h1), nn.GELU(),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, vocab_size))
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        self.k = k
    def forward(self, x):
        e = self.embed(x).flatten(1)  # (B, k*embed_dim)
        return self.net(e)

_LANG_RAW = {"train_shards": None, "val_ids": None, "test_ids": None, "vocab": None}

_WORD_RE = re.compile(r"\S+")

def _tokenize(text):
    return _WORD_RE.findall(text)

def lang_load_raw():
    if _LANG_RAW["train_shards"] is not None:
        return
    ds = hf_datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    def join_lines(split):
        return ds[split]["text"]
    train_lines = join_lines("train")
    val_lines = join_lines("validation")
    test_lines = join_lines("test")

    # partition canonical train split into N_SHARDS_TARGET shards along
    # level-1 "= Title =" headings, per spec, greedily balanced by token count
    heading_re = re.compile(r"^\s*=\s[^=].*[^=]\s=\s*$")
    articles, cur = [], []
    for line in train_lines:
        if heading_re.match(line) and cur:
            articles.append(cur); cur = [line]
        else:
            cur.append(line)
    if cur:
        articles.append(cur)
    article_tokens = [_tokenize(" ".join(a)) for a in articles]
    article_tokens = [a for a in article_tokens if len(a) > 0]

    shard_tokens = [[] for _ in range(LANGUAGE_N_SHARDS)]
    shard_counts = [0] * LANGUAGE_N_SHARDS
    for toks in sorted(article_tokens, key=len, reverse=True):
        i = int(np.argmin(shard_counts))
        shard_tokens[i].extend(toks)
        shard_counts[i] += len(toks)

    all_train_tokens = [tok for shard in shard_tokens for tok in shard]
    val_tokens = _tokenize(" ".join(val_lines))
    test_tokens = _tokenize(" ".join(test_lines))

    # vocab fit on pooled TRAIN text only, top VOCAB_SIZE-1 words + <unk>=0
    freq = defaultdict(int)
    for tok in all_train_tokens:
        freq[tok] += 1
    top_words = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:LANGUAGE_VOCAB_SIZE - 1]]
    vocab = {"<unk>": 0}
    for w in top_words:
        vocab[w] = len(vocab)

    def to_ids(tokens):
        return np.array([vocab.get(t, 0) for t in tokens], dtype=np.int64)

    _LANG_RAW["train_shards"] = [to_ids(shard) for shard in shard_tokens]
    _LANG_RAW["val_ids"] = to_ids(val_tokens)
    _LANG_RAW["test_ids"] = to_ids(test_tokens)
    _LANG_RAW["vocab"] = vocab
    log(f"[LANGUAGE] vocab={len(vocab)}  shards_sizes={[len(s) for s in _LANG_RAW['train_shards']]}")

def _windows_from_ids(ids, k=LANGUAGE_CONTEXT_K):
    if len(ids) <= k:
        return np.zeros((0, k), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    n = len(ids) - k
    X = np.stack([ids[i:i+k] for i in range(n)], axis=0)
    y = ids[k:k+n]
    return X, y

_LANG_LOADER_CACHE = {}

def lang_get_loaders(shard_id, batch_size):
    if shard_id not in _LANG_LOADER_CACHE:
        lang_load_raw()
        shard_ids = _LANG_RAW["train_shards"][shard_id]
        X_tr, y_tr = _windows_from_ids(shard_ids)
        X_va, y_va = _windows_from_ids(_LANG_RAW["val_ids"])
        X_te, y_te = _windows_from_ids(_LANG_RAW["test_ids"])
        if len(X_tr) < 50:
            raise ValueError(f"language shard {shard_id} has too few windows ({len(X_tr)})")
        def ds(Xd, yd):
            return TensorDataset(torch.tensor(Xd, dtype=torch.long),
                                  torch.tensor(yd, dtype=torch.long))
        _LANG_LOADER_CACHE[shard_id] = (
            DataLoader(ds(X_tr, y_tr), batch_size=batch_size, shuffle=True),
            DataLoader(ds(X_va, y_va), batch_size=batch_size),
            DataLoader(ds(X_te, y_te), batch_size=batch_size))
    return _LANG_LOADER_CACHE[shard_id]

LANGUAGE_TUNING_SHARDS = [i for i in range(LANGUAGE_N_SHARDS) if i not in LANGUAGE_SHARD_IDS][:N_TUNING_SHARDS]

# ==============================================================================
#  DOMAIN DISPATCH TABLE
# ==============================================================================
def get_loaders(domain, task_id, batch_size):
    if domain == "finance":
        return fin_get_loaders(task_id, batch_size)
    elif domain == "vision":
        return vision_get_loaders(task_id, batch_size)
    elif domain == "language":
        return lang_get_loaders(task_id, batch_size)
    raise ValueError(domain)

def build_model(domain):
    if domain == "finance":
        return FinanceMLP(MODEL_HIDDEN).to(device)
    elif domain == "vision":
        return LightCNN(VISION_MODEL_WIDTH, VISION_NUM_CLASSES).to(device)
    elif domain == "language":
        return NPLM_MLP(LANGUAGE_VOCAB_SIZE, LANGUAGE_EMBED_DIM, LANGUAGE_CONTEXT_K,
                         LANGUAGE_MODEL_HIDDEN).to(device)
    raise ValueError(domain)

def build_criterion(domain):
    if domain == "finance":
        return nn.MSELoss()
    elif domain in ("vision", "language"):
        return nn.CrossEntropyLoss()
    raise ValueError(domain)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
def evaluate(domain, model, loader, criterion):
    model.eval()
    tot_loss, tot_n = 0.0, 0
    preds_all, tgts_all = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        p = model(X)
        if domain == "finance":
            tot_loss += criterion(p, y).item() * X.size(0)
        else:
            tot_loss += criterion(p, y).item() * X.size(0)
        tot_n += X.size(0)
        preds_all.append(p.cpu()); tgts_all.append(y.cpu())
    loss = tot_loss / max(tot_n, 1)
    pred, tgt = torch.cat(preds_all), torch.cat(tgts_all)
    if domain == "finance":
        mask = tgt.abs() > 1e-9
        dir_acc = ((pred[mask] * tgt[mask]) > 0).float().mean().item() if mask.sum() > 0 else float('nan')
        return loss, dir_acc
    else:
        pred_cls = pred.argmax(dim=1)
        if domain == "language":
            mask = tgt != 0  # exclude <unk> targets, per spec
            acc = (pred_cls[mask] == tgt[mask]).float().mean().item() if mask.sum() > 0 else float('nan')
        else:
            acc = (pred_cls == tgt).float().mean().item()
        return loss, acc

# ==============================================================================
#  CALIBRATION: gamma_fixed for Stella-FixedGamma, per domain, on a DISJOINT
#  calibration seed and a tuning pool disjoint from the eval task instances.
# ==============================================================================
def calibrate_fixed_gamma(domain, tuning_tasks, lr):
    set_seed(CALIBRATION_SEED)
    sigma2_estimates = []
    for task_id in tuning_tasks:
        train_loader, _, _ = get_loaders(domain, task_id, BATCH_SIZE)
        model = build_model(domain)
        criterion = build_criterion(domain)
        params = list(model.parameters())
        fast_m1 = [torch.zeros_like(p) for p in params]
        fast_m2 = [torch.zeros_like(p) for p in params]
        loader_iter = itertools.cycle(train_loader)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
        for step in range(CALIBRATE_STEP + 1):
            X_batch, y_batch = next(loader_iter)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
            with torch.no_grad():
                for i, p in enumerate(params):
                    if p.grad is None:
                        continue
                    fast_m1[i].mul_(FAST_EMA_BETA).add_(p.grad, alpha=1.0 - FAST_EMA_BETA)
                    fast_m2[i].mul_(FAST_EMA_BETA).addcmul_(p.grad, p.grad, value=1.0 - FAST_EMA_BETA)
            optimizer.step()
        with torch.no_grad():
            for i in range(len(params)):
                s2 = (fast_m2[i] - fast_m1[i].pow(2)).clamp(min=0.0)
                sigma2_estimates.append(s2.mean().item())
    mean_s2 = max(float(np.mean(sigma2_estimates)), 1e-12)
    gamma_fixed = float(np.clip((1.0 / TARGET_WP - 1.0) / mean_s2, 10.0, 5000.0))
    log(f"[CALIBRATION:{domain}] gamma_fixed={gamma_fixed:.2f} (mean_sigma2={mean_s2:.2e}, "
        f"calibration_seed={CALIBRATION_SEED}, disjoint from eval seeds {EVAL_SEEDS})")
    return gamma_fixed

# ==============================================================================
#  LR GRID SEARCH -- per domain, per arm, on the disjoint tuning pool +
#  LR_SEARCH_SEED only. Never touches EVAL_SEEDS or the eval task instances.
# ==============================================================================
def lr_grid_search(domain, arm_name, tuning_tasks, gamma_fixed_value=None):
    val_by_lr = defaultdict(list)
    for lr in LR_GRID:
        for task_id in tuning_tasks:
            set_seed(LR_SEARCH_SEED)
            train_loader, val_loader, _ = get_loaders(domain, task_id, BATCH_SIZE)
            model = build_model(domain)
            criterion = build_criterion(domain)
            optimizer = build_optimizer(arm_name, model, lr, gamma_fixed_value)
            loader_iter = itertools.cycle(train_loader)
            for step in range(LR_SEARCH_STEPS):
                model.train()
                X_batch, y_batch = next(loader_iter)
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                if (arm_name in IS_STELLA_FAMILY_ADAPTIVE) and step == CALIBRATE_STEP:
                    optimizer.calibrate_gamma0()
            val_loss, _ = evaluate(domain, model, val_loader, criterion)
            if not math.isnan(val_loss):
                val_by_lr[lr].append(val_loss)
    median_by_lr = {lr: (float(np.median(v)) if v else float('inf')) for lr, v in val_by_lr.items()}
    best_lr = min(median_by_lr, key=median_by_lr.get)
    log(f"  [LR-SEARCH:{domain}] {arm_name}: medians="
        f"{ {f'{k:.0e}': round(v,6) for k,v in median_by_lr.items()} } -> selected lr={best_lr:.0e}")
    return best_lr

# ==============================================================================
#  SINGLE EXPERIMENT (one domain, one arm, one task instance, one seed)
# ==============================================================================
def run_experiment(domain, arm_name, lr, task_id, seed, gamma_fixed_value=None, max_steps=MAX_STEPS):
    set_seed(seed)
    reset_peak_memory()
    eval_every = EVAL_EVERY_BY_DOMAIN[domain]

    train_loader, val_loader, test_loader = get_loaders(domain, task_id, BATCH_SIZE)
    model = build_model(domain)
    criterion = build_criterion(domain)
    n_params = count_params(model)

    optimizer = build_optimizer(arm_name, model, lr, gamma_fixed_value, collect_diagnostics=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.1)

    X0, y0 = next(iter(train_loader))
    with torch.no_grad():
        init_loss = criterion(model(X0.to(device)), y0.to(device)).item()

    step_sizes = []
    steps_to_half = steps_to_eps = None
    clip_trigger_count = 0
    final_train_loss = init_loss
    calibrated_once = False
    loader_iter = itertools.cycle(train_loader)
    best_val = float('inf'); best_state = None

    t0 = time.time()
    model.train()
    for step in range(max_steps):
        X_batch, y_batch = next(loader_iter)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        with torch.no_grad():
            params_before = [p.data.clone() for p in model.parameters()]

        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        if float(pre_clip_norm) > GRAD_CLIP_NORM:
            clip_trigger_count += 1

        optimizer.step()
        scheduler.step()

        if (arm_name in IS_STELLA_FAMILY_ADAPTIVE) and (not calibrated_once) and step == CALIBRATE_STEP:
            optimizer.calibrate_gamma0()
            calibrated_once = True

        lv = loss.item()
        with torch.no_grad():
            delta = torch.cat([(p.data - pb).flatten() for p, pb in zip(model.parameters(), params_before)])
        del params_before
        step_sizes.append(math.sqrt(delta.pow(2).sum().item() / max(n_params, 1)))

        if steps_to_half is None and lv <= 0.5 * init_loss:
            steps_to_half = step
        if steps_to_eps is None and lv <= TARGET_EPS_FRAC * init_loss:
            steps_to_eps = step
        final_train_loss = lv

        if step % eval_every == 0:
            val_loss, _ = evaluate(domain, model, val_loader, criterion)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_metric = evaluate(domain, model, test_loader, criterion)

    wall = time.time() - t0
    mem = peak_memory_mb()
    mean_step_size = float(np.mean(step_sizes)) if step_sizes else float('nan')
    clip_trigger_freq = clip_trigger_count / max(max_steps, 1)

    diag_summary = {}
    d = getattr(optimizer, "diag", {})
    for key, vals in d.items():
        if vals:
            diag_summary[f"diag_{key}_mean"] = float(np.mean(vals))
            diag_summary[f"diag_{key}_max"] = float(np.max(vals))

    # domain-specific "final metric" naming: finance/language use loss-based
    # (MSE / CE), vision uses accuracy-based, all per spec metric definitions
    if domain == "finance":
        primary_metric = test_loss                       # test MSE (lower better)
        secondary_metric = test_metric                    # dir_acc
    elif domain == "vision":
        primary_metric = 1.0 - test_metric                # 1 - top1 acc (lower better, paired w/ finance/lang sign convention)
        secondary_metric = test_metric                     # top1 acc
    else:  # language
        primary_metric = test_loss                        # test CE loss (lower better)
        secondary_metric = test_metric                     # top1acc (next-word, excl. <unk>)

    result = {
        'primary_metric': primary_metric, 'secondary_metric': secondary_metric,
        'test_loss': test_loss,
        'steps_half': steps_to_half if steps_to_half is not None else max_steps,
        'steps_eps': steps_to_eps if steps_to_eps is not None else max_steps,
        'wall_s': wall, 'peak_mem_mb': mem,
        'mean_step_sz': mean_step_size, 'clip_trigger_freq': clip_trigger_freq,
        'n_params': n_params, 'lr_used': lr,
    }
    result.update(diag_summary)
    return result

METRIC_KEYS = ['primary_metric', 'secondary_metric', 'test_loss', 'steps_half', 'steps_eps',
               'wall_s', 'peak_mem_mb', 'mean_step_sz', 'clip_trigger_freq']

# ==============================================================================
#  CHECKPOINTING
# ==============================================================================
def load_checkpoint():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {"task_ids": {}, "lr_used": {}, "gamma_fixed": {}, "completed": {}, "raw": {}}

def save_checkpoint(ckpt):
    tmp = CKPT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    os.replace(tmp, CKPT_PATH)

# ==============================================================================
#  ANALYSIS -- same statistical machinery as the original file (Wilcoxon +
#  Holm + paired bootstrap + rank-biserial), run per DOMAIN, over the 3 task
#  instances as the paired unit (analogous to "tickers"/"shards" in the
#  original). Reported strictly as DESCRIPTIVE given n=3 seeds / n=3 tasks --
#  no significance claims are made in the writeup, per instruction.
# ==============================================================================
def rank_biserial_effect_size(diffs):
    diffs = np.asarray(diffs); diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return float('nan')
    ranks = np.argsort(np.argsort(np.abs(diffs))) + 1
    r_plus = ranks[diffs > 0].sum(); r_minus = ranks[diffs < 0].sum()
    total = r_plus + r_minus
    return float((r_plus - r_minus) / total) if total > 0 else float('nan')

def paired_bootstrap_ci(challenger_vals, baseline_vals, n_boot=10000, alpha=0.05, rng=None):
    c = np.asarray(challenger_vals); b = np.asarray(baseline_vals)
    diffs = b - c
    n = len(diffs)
    if rng is None:
        rng = np.random.default_rng(12345)
    if n == 0:
        return (float('nan'), float('nan'))
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = diffs[idx].mean()
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)

def holm_correct(pvals_dict):
    items = [(k, p) for k, p in pvals_dict.items() if not math.isnan(p)]
    items_sorted = sorted(items, key=lambda x: x[1])
    m = len(items_sorted)
    corrected = {k: float('nan') for k in pvals_dict}
    running_max = 0.0
    for rank0, (k, p) in enumerate(items_sorted):
        val = min(p * (m - rank0), 1.0)
        running_max = max(running_max, val)
        corrected[k] = running_max
    return corrected

def analyze_domain(domain, agg_by_task_seed, task_ids, seeds):
    """agg_by_task_seed[arm][task_id][seed] = result dict.
    Primary unit = TASK INSTANCE (mean over the 3 seeds), n=3 per comparison.
    NOTE: with n=3, no significance claim is made -- see report text."""
    task_level_mean = {}
    for arm in agg_by_task_seed:
        task_level_mean[arm] = {}
        for t in task_ids:
            vals = [agg_by_task_seed[arm].get(t, {}).get(s, {}).get('primary_metric') for s in seeds]
            vals = [v for v in vals if v is not None and not math.isnan(v)]
            task_level_mean[arm][t] = float(np.mean(vals)) if vals else float('nan')

    raw_p, diffs_store, pairs_store = {}, {}, {}
    for challenger, baseline in COMPARISONS:
        key = f"{challenger}_vs_{baseline}"
        c_vals, b_vals = [], []
        for t in task_ids:
            c = task_level_mean.get(challenger, {}).get(t)
            b = task_level_mean.get(baseline, {}).get(t)
            if c is not None and b is not None and not (math.isnan(c) or math.isnan(b)):
                c_vals.append(c); b_vals.append(b)
        diffs = [b - c for c, b in zip(c_vals, b_vals)]  # positive = challenger better (lower metric)
        diffs_store[key] = diffs; pairs_store[key] = (c_vals, b_vals)
        try:
            _, p = wilcoxon(c_vals, b_vals) if len(diffs) >= 1 else (None, float('nan'))
        except ValueError:
            p = float('nan')
        raw_p[key] = p
    corrected_p = holm_correct(raw_p)

    report = {}
    for challenger, baseline in COMPARISONS:
        key = f"{challenger}_vs_{baseline}"
        diffs = diffs_store[key]; c_vals, b_vals = pairs_store[key]
        eff = rank_biserial_effect_size(diffs)
        ci = paired_bootstrap_ci(c_vals, b_vals) if diffs else (float('nan'), float('nan'))
        report[key] = {
            "domain": domain,
            "statistical_unit": "task instance (mean primary_metric over 3 eval seeds)",
            "n_task_instances": len(diffs),
            "mean_diff": float(np.mean(diffs)) if diffs else float('nan'),
            "median_diff": float(np.median(diffs)) if diffs else float('nan'),
            "p_raw": raw_p[key], "p_holm": corrected_p[key],
            "rank_biserial_effect_size": eff,
            "bootstrap_ci95": ci,
            "significance_claim": "NONE -- n=3 seeds / n=3 task instances; "
                                   "descriptive mechanism/robustness comparison only.",
        }
    return report, task_level_mean

# ==============================================================================
#  MAIN
# ==============================================================================
DOMAINS = ["finance", "vision", "language"]

def get_task_ids_for_domain(domain, ckpt):
    if domain in ckpt["task_ids"]:
        return ckpt["task_ids"][domain]
    if domain == "finance":
        ids = fin_select_task_tickers(N_TICKERS_TASK, seed=1234)
    elif domain == "vision":
        ids = list(VISION_SPLIT_SEEDS)   # 3 paired re-split instances of the SAME CIFAR-100 dataset
    elif domain == "language":
        ids = list(LANGUAGE_SHARD_IDS)   # 3 of the 10 canonical deterministic shards
    else:
        raise ValueError(domain)
    ckpt["task_ids"][domain] = ids
    return ids

def get_tuning_tasks_for_domain(domain, eval_task_ids):
    if domain == "finance":
        return fin_select_tuning_tickers(exclude=set(eval_task_ids), n=N_TUNING_TICKERS, seed=99)
    elif domain == "vision":
        return [VISION_TUNING_SPLIT_SEED]      # held-out re-split, never one of the 3 eval instances
    elif domain == "language":
        return LANGUAGE_TUNING_SHARDS          # 7 remaining canonical shards, disjoint from the 3 eval shards
    raise ValueError(domain)

def main():
    log_sep("=")
    log(f"  STELLA MECHANISM-ISOLATION ABLATION -- MULTI-DOMAIN  |  device={device}  torch={torch.__version__}")
    log(f"  Domains: {DOMAINS}  |  Arms: {ARM_NAMES}  |  EVAL_SEEDS: {EVAL_SEEDS}")
    log_sep("=")

    unit_test_results = run_unit_tests()
    ckpt = load_checkpoint()

    all_stat_reports = {}
    all_raw_rows = []
    all_diag_rows = []
    per_task_rows = []

    for domain in DOMAINS:
        log_sep("-")
        log(f"[DOMAIN] {domain}")
        task_ids = get_task_ids_for_domain(domain, ckpt)
        save_checkpoint(ckpt)
        tuning_tasks = get_tuning_tasks_for_domain(domain, task_ids)
        log(f"  eval task instances (n={len(task_ids)}): {task_ids}")
        log(f"  tuning pool (disjoint, n={len(tuning_tasks)}): {tuning_tasks}")
        if domain == "vision":
            log("  NOTE: vision task instances are 3 PAIRED RE-SPLITS of the SAME "
                "CIFAR-100 dataset (different SPLIT_SEED), not independent datasets.")

        # ---- calibrate gamma_fixed on disjoint calibration seed + tuning pool ----
        if domain not in ckpt["gamma_fixed"]:
            gamma_fixed_value = calibrate_fixed_gamma(domain, tuning_tasks, lr=1e-3)
            ckpt["gamma_fixed"][domain] = gamma_fixed_value
            save_checkpoint(ckpt)
        else:
            gamma_fixed_value = ckpt["gamma_fixed"][domain]

        # ---- LR search per arm, frozen before pairing, on tuning pool only ----
        if domain not in ckpt["lr_used"]:
            lr_used = {}
            for arm in ARM_NAMES:
                lr_used[arm] = lr_grid_search(domain, arm, tuning_tasks, gamma_fixed_value)
            ckpt["lr_used"][domain] = lr_used
            save_checkpoint(ckpt)
        else:
            lr_used = ckpt["lr_used"][domain]
        log(f"  [LR-SEARCH:{domain}] Frozen LRs: {lr_used}")

        # ---- full paired sweep: 3 task instances x 3 seeds x 4 arms ----
        raw = {arm: {t: {} for t in task_ids} for arm in ARM_NAMES}
        for arm in ARM_NAMES:
            for t in task_ids:
                for s in EVAL_SEEDS:
                    key = f"{domain}|{arm}|{t}|{s}"
                    if key in ckpt["raw"]:
                        raw[arm][t][s] = ckpt["raw"][key]

        for t in task_ids:
            for arm in ARM_NAMES:
                for seed in EVAL_SEEDS:
                    key = f"{domain}|{arm}|{t}|{seed}"
                    if key in ckpt["completed"]:
                        continue
                    lr = lr_used[arm]
                    r = run_experiment(domain, arm, lr, t, seed, gamma_fixed_value)
                    raw[arm][t][seed] = r
                    ckpt["raw"][key] = r
                    ckpt["completed"][key] = True
                    save_checkpoint(ckpt)
            log(f"  [PROGRESS:{domain}] task {t} done (all arms x all seeds).")

        # ---- collect raw + diagnostics rows ----
        for arm in ARM_NAMES:
            for t in task_ids:
                for s in EVAL_SEEDS:
                    r = raw[arm][t].get(s)
                    if r is None:
                        continue
                    row = {"domain": domain, "arm": arm, "task_id": t, "seed": s,
                           "lr_used": lr_used[arm], **r}
                    all_raw_rows.append(row)
                    all_diag_rows.append(row)

        # ---- per-task summary (mean/std across 3 seeds, per arm, per task) ----
        for arm in ARM_NAMES:
            for t in task_ids:
                vals = [raw[arm][t].get(s, {}).get('primary_metric') for s in EVAL_SEEDS]
                vals = [v for v in vals if v is not None and not math.isnan(v)]
                base_vals = [raw["AdamW"][t].get(s, {}).get('primary_metric') for s in EVAL_SEEDS]
                base_vals = [v for v in base_vals if v is not None and not math.isnan(v)]
                paired_improve = (float(np.mean(base_vals)) - float(np.mean(vals))
                                  if vals and base_vals else float('nan'))
                per_task_rows.append({
                    "domain": domain, "arm": arm, "task_id": t,
                    "n_seeds": len(vals),
                    "mean_primary_metric": float(np.mean(vals)) if vals else float('nan'),
                    "std_primary_metric": float(np.std(vals)) if vals else float('nan'),
                    "paired_improvement_vs_adamw": paired_improve,
                    "vision_note": ("repeated CIFAR-100 task instance (paired re-split), "
                                     "not an independent dataset") if domain == "vision" else "",
                })

        # ---- statistical analysis (descriptive only, n=3) ----
        stat_report, task_level_mean = analyze_domain(domain, raw, task_ids, EVAL_SEEDS)
        all_stat_reports[domain] = stat_report
        log(json.dumps(stat_report, indent=2))

    # ================== WRITE DELIVERABLE 1: raw_results.csv ==================
    raw_fieldnames = ["domain", "arm", "task_id", "seed", "lr_used"] + METRIC_KEYS
    diag_key_set = sorted({k for row in all_raw_rows for k in row if k.startswith("diag_")})
    with open(RAW_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fieldnames + diag_key_set, extrasaction="ignore")
        writer.writeheader()
        for row in all_raw_rows:
            writer.writerow(row)
    log(f"[CSV] Saved -> {RAW_CSV_PATH}")

    # ============ WRITE DELIVERABLE 2: per_task_summary.csv ============
    with open(PER_TASK_CSV, "w", newline="") as f:
        fieldnames = ["domain", "arm", "task_id", "n_seeds", "mean_primary_metric",
                      "std_primary_metric", "paired_improvement_vs_adamw", "vision_note"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_task_rows:
            writer.writerow(row)
    log(f"[CSV] Saved -> {PER_TASK_CSV}")

    # ============ WRITE DELIVERABLE 5: mechanism_diagnostics.csv ============
    with open(DIAG_PATH, "w", newline="") as f:
        fieldnames = ["domain", "arm", "task_id", "seed"] + diag_key_set
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_diag_rows:
            writer.writerow(row)
    log(f"[CSV] Saved -> {DIAG_PATH}")

    # ============ WRITE DELIVERABLE 3: ablation_summary.csv ============
    ablation_rows = []
    for domain in DOMAINS:
        for key, rep in all_stat_reports[domain].items():
            ablation_rows.append({
                "domain": domain, "comparison": key,
                "n_task_instances": rep["n_task_instances"],
                "mean_diff": rep["mean_diff"], "median_diff": rep["median_diff"],
                "p_raw": rep["p_raw"], "p_holm": rep["p_holm"],
                "rank_biserial_effect_size": rep["rank_biserial_effect_size"],
                "bootstrap_ci95_lo": rep["bootstrap_ci95"][0],
                "bootstrap_ci95_hi": rep["bootstrap_ci95"][1],
                "significance_claim": rep["significance_claim"],
            })
    with open(ABLATION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()) if ablation_rows else [])
        writer.writeheader()
        for row in ablation_rows:
            writer.writerow(row)
    log(f"[CSV] Saved -> {ABLATION_CSV}")

    # ============ WRITE DELIVERABLE 4: ablation_summary.tex ============
    with open(ABLATION_TEX, "w") as f:
        f.write("% Auto-generated. Descriptive ablation table -- n=3 seeds per task instance,\n")
        f.write("% n=3 task instances per domain. NO significance claims (see paper text).\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write("Domain & Comparison & $n$ & Mean $\\Delta$ & Rank-biserial & 95\\% CI \\\\\n\\midrule\n")
        for row in ablation_rows:
            ci = f"[{row['bootstrap_ci95_lo']:.3g}, {row['bootstrap_ci95_hi']:.3g}]"
            f.write(f"{row['domain']} & {row['comparison'].replace('_', ' ')} & "
                    f"{row['n_task_instances']} & {row['mean_diff']:.3g} & "
                    f"{row['rank_biserial_effect_size']:.3f} & {ci} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Mechanism-isolation ablation across finance, vision, and language "
                "(3 paired task instances $\\times$ 3 seeds per domain). Positive mean $\\Delta$ "
                "favors the first-named (challenger) arm. Given $n=3$, this table reports "
                "descriptive effect sizes only and makes no significance claim.}\n")
        f.write("\\end{table}\n")
    log(f"[TEX] Saved -> {ABLATION_TEX}")

    # ============ WRITE DELIVERABLE 7: appendix table (markdown) ============
    with open(APPENDIX_TABLE, "w") as f:
        f.write("# Appendix: Mechanism-Isolation Ablation (compact)\n\n")
        f.write("3 domains (finance, vision, language) x 3 paired task instances x 3 seeds x 4 arms "
                "= 108 runs. Vision's 3 task instances are re-splits of the single CIFAR-100 "
                "benchmark dataset (different split seed), not independent datasets. "
                "No significance claim is made from n=3; effect sizes/CIs are descriptive.\n\n")
        f.write("| Domain | Comparison | n | Mean Δ | Rank-biserial | 95% CI |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in ablation_rows:
            ci = f"[{row['bootstrap_ci95_lo']:.3g}, {row['bootstrap_ci95_hi']:.3g}]"
            f.write(f"| {row['domain']} | {row['comparison']} | {row['n_task_instances']} | "
                    f"{row['mean_diff']:.3g} | {row['rank_biserial_effect_size']:.3f} | {ci} |\n")
    log(f"[MD] Saved -> {APPENDIX_TABLE}")

    # ============ WRITE DELIVERABLE 6: paired improvement figure ============
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(DOMAINS), figsize=(4 * len(DOMAINS), 4), sharey=False)
        if len(DOMAINS) == 1:
            axes = [axes]
        for ax, domain in zip(axes, DOMAINS):
            rows = [r for r in per_task_rows if r["domain"] == domain and r["arm"] != "AdamW"]
            arms_here = [a for a in ARM_NAMES if a != "AdamW"]
            xs = np.arange(len(arms_here))
            means, stds = [], []
            for arm in arms_here:
                vals = [r["paired_improvement_vs_adamw"] for r in rows if r["arm"] == arm]
                vals = [v for v in vals if not math.isnan(v)]
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals) if vals else np.nan)
            ax.bar(xs, means, yerr=stds, capsize=4)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xticks(xs); ax.set_xticklabels(arms_here, rotation=30, ha="right")
            ax.set_title(domain + (" (repeated instances)" if domain == "vision" else ""))
            ax.set_ylabel("Paired improvement vs AdamW\n(mean over task instances, n=3 seeds)")
        fig.suptitle("Paired improvement over AdamW across domains (descriptive, n=3 seeds)")
        fig.tight_layout()
        fig.savefig(FIGURE_PATH, dpi=150)
        plt.close(fig)
        log(f"[FIGURE] Saved -> {FIGURE_PATH}")
    except Exception as e:
        log(f"[FIGURE] Skipped due to error: {e}")

    log_sep("=")
    log(f"  ALL DONE  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_sep("=")
    _log_fh.close()

if __name__ == "__main__":
    main()

