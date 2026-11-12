import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import subprocess, sys
def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
try:
    import torchvision
except ImportError:
    _pip("torchvision")
    import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings, time, math, csv, json, itertools, copy, random
from datetime import datetime
from collections import defaultdict
from scipy.stats import wilcoxon
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# ==============================================================================
#  GLOBAL CONFIG  (per spec, ported from finance batchsize_runs.py + Run-1 Part A)
# ==============================================================================
SEEDS                 = [42, 43]
EVAL_EVERY            = 100
MODEL_WIDTH_DEFAULT   = (32, 64)

# ---- Run-2 batch-size sweep axis ----
BATCH_SIZES           = [16]                  # this file only runs batch=16 (32 skipped, Run-1 baseline)
FULL_STEPS_BY_BATCH   = {16: 10000}
LR_SEARCH_SAMPLES_SEEN = 3200                 # fixed budget: 100 * 32 / batch steps
HESS_FREQ             = 5
N_HUTCH               = 4
GRAD_CLIP_NORM        = 5.0
WEIGHT_DECAY          = 1e-4
TARGET_EPS_FRAC       = 0.1
SOPHIA_BETA_H         = 0.99

LR_GRID               = [1e-4, 3e-4, 1e-3, 3e-3]
LR_SEARCH_SEED        = 42
N_TUNING_SUPERCLASSES  = 10
N_TUNING_FINE_PER_SUPER = 2

CLIP_METRIC_THRESHOLD = 1.0
TARGET_WP             = 0.85
WELL_POS_MAX          = 8.0
CALIBRATE_STEP        = 125          # in [100,150] per spec
FAST_EMA_BETA         = 0.9          # separate fast EMA used ONLY for gamma0 calibration

SPLIT_SEED            = 2024
NUM_CLASSES_MAIN      = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT  = "/tmp/cifar100_data"
OUT_DIR    = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH  = os.path.join(OUT_DIR, "vision_run2_b16_checkpoint.json")
CSV_PATH   = os.path.join(OUT_DIR, "vision_run2_b16_results_raw.csv")
JSON_PATH  = os.path.join(OUT_DIR, "vision_run2_b16_results_agg.json")
LOG_PATH   = os.path.join(OUT_DIR, "vision_run2_b16_log.txt")

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
#  DATA  (CIFAR-100, chronological-agnostic 80/10/10 random split, no leakage:
#  scaler fit on TRAIN split only)
# ==============================================================================
def _fine_to_coarse_map():
    coarse = np.zeros(100, dtype=np.int64)
    groups = [
        [4,30,55,72,95],[1,32,67,73,91],[54,62,70,82,92],[9,10,16,28,61],
        [0,51,53,57,83],[22,39,40,86,87],[5,20,25,84,94],[6,7,14,18,24],
        [3,42,43,88,97],[12,17,37,68,76],[23,33,49,60,71],[15,19,21,31,38],
        [34,63,64,66,75],[26,45,77,79,99],[2,11,35,46,98],[27,29,44,78,93],
        [36,50,65,74,80],[47,52,56,59,96],[8,13,48,58,90],[41,69,81,85,89],
    ]
    for coarse_idx, fine_list in enumerate(groups):
        for f in fine_list:
            coarse[f] = coarse_idx
    return coarse

def _download_cifar100():
    os.makedirs(DATA_ROOT, exist_ok=True)
    train_ds = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=True)
    test_ds  = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True)
    X = np.concatenate([train_ds.data, test_ds.data], axis=0).astype(np.float32)
    y_fine = np.array(list(train_ds.targets) + list(test_ds.targets), dtype=np.int64)
    coarse_map = _fine_to_coarse_map()
    y_coarse = coarse_map[y_fine]
    X = np.transpose(X, (0, 3, 1, 2))
    return X, y_fine, y_coarse

_SPLIT_CACHE = {}
def build_splits(seed=SPLIT_SEED):
    if _SPLIT_CACHE: return
    X, y_fine, y_coarse = _download_cifar100()
    n = len(X)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_tr, n_va = int(n * 0.8), int(n * 0.1)
    tr_idx, va_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    mu  = X[tr_idx].mean(axis=(0, 2, 3), keepdims=True)
    std = X[tr_idx].std(axis=(0, 2, 3), keepdims=True) + 1e-8

    def make(idxs):
        Xn = (X[idxs] - mu) / std
        return (torch.tensor(Xn, dtype=torch.float32),
                torch.tensor(y_fine[idxs], dtype=torch.long),
                torch.tensor(y_coarse[idxs], dtype=torch.long))

    _SPLIT_CACHE['train'] = make(tr_idx)
    _SPLIT_CACHE['val']   = make(va_idx)
    _SPLIT_CACHE['test']  = make(te_idx)

_TUNING_CACHE = {}
def build_tuning_subset():
    """Fixed tuning subset (10 superclasses x 2 fine classes each), same
    across every batch size — the vision analog of the finance sweep's
    fixed 10-ticker tuning_subset."""
    if _TUNING_CACHE: return _TUNING_CACHE['classes']
    build_splits()
    coarse_map = _fine_to_coarse_map()

    chosen_fine = []
    for sc in range(N_TUNING_SUPERCLASSES):
        fines_in_sc = sorted([f for f in range(100) if coarse_map[f] == sc])
        chosen_fine.extend(fines_in_sc[:N_TUNING_FINE_PER_SUPER])
    tuning_classes = sorted(chosen_fine)
    remap = {c: i for i, c in enumerate(tuning_classes)}

    for split in ('train', 'val', 'test'):
        X, y_fine, _ = _SPLIT_CACHE[split]
        y_np = y_fine.numpy()
        mask = np.isin(y_np, tuning_classes)
        Xs = X[mask]
        ys = torch.tensor([remap[int(v)] for v in y_np[mask]], dtype=torch.long)
        _TUNING_CACHE[split] = (Xs, ys)
    _TUNING_CACHE['classes'] = tuning_classes
    return tuning_classes

def get_main_loaders(batch_size):
    build_splits()
    def ds(split):
        X, y, _ = _SPLIT_CACHE[split]
        return TensorDataset(X, y)
    return (DataLoader(ds('train'), batch_size=batch_size, shuffle=True),
            DataLoader(ds('val'), batch_size=batch_size),
            DataLoader(ds('test'), batch_size=batch_size))

def get_tuning_loaders(batch_size):
    build_tuning_subset()
    def ds(split):
        X, y = _TUNING_CACHE[split]
        return TensorDataset(X, y)
    return (DataLoader(ds('train'), batch_size=batch_size, shuffle=True),
            DataLoader(ds('val'), batch_size=batch_size),
            DataLoader(ds('test'), batch_size=batch_size))

# ==============================================================================
#  MODEL
# ==============================================================================
class LightCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES_MAIN, width=MODEL_WIDTH_DEFAULT):
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
        x = x.flatten(1)
        return self.fc(x)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_fwd_flops(model, batch_size, input_hw=32):
    flops = 0
    h = input_hw
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            flops += 2 * m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] * h * h * batch_size
            h = h // 2
        elif isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features * batch_size
    return int(flops)

# ==============================================================================
#  OPTIMIZERS  (unchanged from finance/Run-1 — same 5 optimizers, same update
#  rules; only the model/data pipeline differs between finance and vision)
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
            log(f"    [Evie] gamma0={new_g0:.1f}  fast_sigma2={mean_s2:.2e}")

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
                well_pos = (1.0 - gamma_t * sigma2).clamp_(min=1e-6)
                denom = m2c.sqrt().clamp_(min=eps)
                p.data.addcdiv_(m1c.mul(well_pos.sqrt()), denom, value=-lr)
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
#  HESSIAN ESTIMATOR (Hutchinson diagonal, consumed only by Sophia & AdaHessian
#  every HESS_FREQ steps — same single estimator used for all 5 optimizers,
#  passive probe for the other three).
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
    model.eval()
    tot_loss, tot_n, top1_correct, top5_correct = 0.0, 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        tot_loss += criterion(logits, y).item() * X.size(0)
        tot_n += X.size(0)
        top1 = logits.argmax(dim=1)
        top1_correct += (top1 == y).sum().item()
        k = min(5, logits.size(1))
        top5 = logits.topk(k, dim=1).indices
        top5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
    avg_loss = tot_loss / max(tot_n, 1)
    top1_pct = 100.0 * top1_correct / max(tot_n, 1)
    top5_pct = 100.0 * top5_correct / max(tot_n, 1)
    return avg_loss, top1_pct, top5_pct

# ==============================================================================
#  LR GRID SEARCH  — fixed tuning subset, ALL 5 optimizers, frozen per batch size
# ==============================================================================
def _lr_grid_search_unsafe(opt_name, batch_size, width, n_search_steps):
    build_tuning_subset()
    val_by_lr = defaultdict(list)
    for lr in LR_GRID:
        set_seed(LR_SEARCH_SEED)
        train_loader, val_loader, _ = get_tuning_loaders(batch_size)
        tuning_classes = _TUNING_CACHE['classes']
        model = LightCNN(num_classes=len(tuning_classes), width=width).to(device)
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
        val_loss, _, _ = evaluate(model, val_loader, criterion)
        if not math.isnan(val_loss):
            val_by_lr[lr].append(val_loss)
    median_by_lr = {lr: (float(np.median(v)) if v else float('inf'))
                     for lr, v in val_by_lr.items()}
    best_lr = min(median_by_lr, key=median_by_lr.get)
    log(f"  [LR-SEARCH] {opt_name}: medians={ {f'{k:.0e}': round(v,6) for k,v in median_by_lr.items()} } "
        f"-> selected lr={best_lr:.0e}")
    return best_lr

# ==============================================================================
#  SINGLE EXPERIMENT (one optimizer, one seed, one batch size)
# ==============================================================================
def _run_experiment_unsafe(opt_name, lr, seed, batch_size, width=MODEL_WIDTH_DEFAULT, max_steps=None):
    set_seed(seed)
    reset_peak_memory()

    train_loader, val_loader, test_loader = get_main_loaders(batch_size)
    model = LightCNN(num_classes=NUM_CLASSES_MAIN, width=width).to(device)
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
    clip_trigger_count = 0
    hess_norms = []
    cum_regret = 0.0
    cumulative_flops = 0.0
    fwd_flops_per_batch = estimate_fwd_flops(model, batch_size)
    is_hessian_consumer = opt_name in HESSIAN_CONSUMERS
    is_evie_family = opt_name in ("Evie", "Evie-Amplify")

    loader_iter = itertools.cycle(train_loader)
    best_val_loss = float('inf'); best_state = None
    best_val_top1_so_far = -float('inf')
    calibrated_once = False

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
        if float(pre_clip_norm) > CLIP_METRIC_THRESHOLD:
            clip_trigger_count += 1

        ran_probe = (step % HESS_FREQ == 0) and is_hessian_consumer
        h_list = None
        if ran_probe:
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

        if is_evie_family and not calibrated_once and step == CALIBRATE_STEP:
            optimizer.calibrate_gamma0()
            calibrated_once = True

        with torch.no_grad():
            delta = torch.cat([(p.data - pb).flatten()
                                for p, pb in zip(model.parameters(), params_before)])
        del params_before
        delta_sq = delta.pow(2).sum().item()
        step_sizes.append(math.sqrt(delta_sq / max(n_params, 1)))

        if step % EVAL_EVERY == 0 or step == max_steps - 1:
            val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
            cum_regret += max(best_val_top1_so_far - val_top1, 0.0)
            best_val_top1_so_far = max(best_val_top1_so_far, val_top1)
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_top1, test_top5 = evaluate(model, test_loader, criterion)

    wall = time.time() - t0
    mem = peak_memory_mb()
    mean_step_size = float(np.mean(step_sizes)) if step_sizes else float('nan')
    gflops = cumulative_flops / max(max_steps, 1) / 1e9
    exaflops = cumulative_flops / 1e18
    clip_trigger_freq = clip_trigger_count / max(max_steps, 1)

    if hess_norms:
        hess_diag_norm = float(np.mean(hess_norms))
        hn_std = float(np.std(hess_norms))
        hess_diag_cv = hn_std / hess_diag_norm if hess_diag_norm > 0 else float('nan')
    else:
        hess_diag_norm = hess_diag_cv = float('nan')

    return {
        'test_top1': test_top1, 'test_top5': test_top5, 'test_loss': test_loss,
        'val_loss': best_val_loss,
        'wall_s': wall, 'peak_mem_mb': mem,
        'gflops': gflops, 'exaflops': exaflops,
        'mean_step_sz': mean_step_size,
        'clip_trigger_freq': clip_trigger_freq,
        'hess_diag_norm': hess_diag_norm, 'hess_diag_cv': hess_diag_cv,
        'regret': cum_regret,
        'n_params': n_params, 'lr_used': lr,
    }

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
            'test_top1': float('nan'), 'test_top5': float('nan'),
            'test_loss': float('nan'), 'val_loss': float('nan'),
            'wall_s': 0.0, 'peak_mem_mb': 0.0,
            'gflops': 0.0, 'exaflops': 0.0,
            'mean_step_sz': float('nan'),
            'clip_trigger_freq': float('nan'),
            'hess_diag_norm': float('nan'), 'hess_diag_cv': float('nan'),
            'regret': float('nan'),
            'n_params': 0, 'lr_used': args[1] if len(args) > 1 else 1e-3,
        }

METRIC_KEYS = ['test_top1', 'test_top5', 'test_loss', 'val_loss', 'wall_s',
               'peak_mem_mb', 'gflops', 'exaflops', 'mean_step_sz',
               'clip_trigger_freq', 'hess_diag_norm', 'hess_diag_cv', 'regret']
# direction: True = lower is better
LOWER_BETTER = {
    'test_top1': False, 'test_top5': False, 'test_loss': True, 'val_loss': True,
    'wall_s': True, 'peak_mem_mb': True, 'gflops': True, 'exaflops': True,
    'mean_step_sz': None, 'clip_trigger_freq': True,
    'hess_diag_norm': None, 'hess_diag_cv': True, 'regret': True,
}

# ==============================================================================
#  CHECKPOINTING
# ==============================================================================
def load_checkpoint():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {"by_batch": {}}

def save_checkpoint(ckpt):
    tmp = CKPT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    os.replace(tmp, CKPT_PATH)

# ==============================================================================
#  AGGREGATION / STATS
# ==============================================================================
def aggregate_seed_results(seed_results):
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
#  (a) test_top1 mean+-std, rows=batch size, cols=optimizer
#  (b) Evie-Amplify vs AdamW %delta (+ p_holm, corrected jointly across the
#      batch-size rows) on test_top1 (higher is better; positive %delta means
#      Evie-Amplify better), rows=batch size
# ==============================================================================
def build_batch_trend_tables(raw_by_batch_opt_seed, opt_names, seeds, batch_sizes):
    challenger = "Evie-Amplify"

    # (a) mean+-std test_top1 per (batch, opt), across seeds
    top1_stats = {}
    for b in batch_sizes:
        top1_stats[b] = {}
        for opt in opt_names:
            vals = [raw_by_batch_opt_seed[b][opt][s]['test_top1'] for s in seeds
                    if s in raw_by_batch_opt_seed[b][opt]
                    and not math.isnan(raw_by_batch_opt_seed[b][opt][s]['test_top1'])]
            top1_stats[b][opt] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float('nan'), float('nan'))

    # (b) Evie-Amplify vs AdamW %delta on test_top1 + Wilcoxon,
    # Holm-Bonferroni corrected jointly across all batch-size rows
    raw_p = {}
    delta_pct = {}
    for b in batch_sizes:
        c_vals, b_vals = [], []
        for s in seeds:
            c = raw_by_batch_opt_seed[b].get(challenger, {}).get(s, {}).get('test_top1', float('nan'))
            a = raw_by_batch_opt_seed[b].get("AdamW", {}).get(s, {}).get('test_top1', float('nan'))
            if not (math.isnan(c) or math.isnan(a)):
                c_vals.append(c); b_vals.append(a)
        if len(c_vals) >= 3:
            try:
                _, p = wilcoxon(c_vals, b_vals)
            except ValueError:
                p = float('nan')
        else:
            p = float('nan')
        raw_p[b] = p
        if b_vals and c_vals:
            a_med, c_med = float(np.median(b_vals)), float(np.median(c_vals))
            # higher top1 is better -> positive %delta means Evie-Amplify better
            delta_pct[b] = (c_med - a_med) / abs(a_med) * 100.0 if a_med != 0 else float('nan')
        else:
            delta_pct[b] = float('nan')
    corrected_p = holm_bonferroni(raw_p)

    return top1_stats, delta_pct, corrected_p

def print_batch_trend_tables(top1_stats, delta_pct, corrected_p, opt_names, batch_sizes):
    log_sep("=", 160)
    log("  RUN 2 (VISION) — BATCH-SIZE SWEEP: test top-1 accuracy %% (mean +/- std across seeds)")
    log_sep("-", 160)
    header = f"  {'Batch':<8}"
    for opt in opt_names:
        header += f"{opt:>22}"
    log(header)
    log("  " + "-" * (8 + 22 * len(opt_names)))
    for b in batch_sizes:
        row = f"  {b:<8}"
        for opt in opt_names:
            m, sd = top1_stats[b][opt]
            cell = f"{m:.4g}+/-{sd:.2g}" if not math.isnan(m) else "nan"
            row += f"{cell:>22}"
        log(row)
    log_sep("-", 160)
    log("  Evie-Amplify vs AdamW on test top-1 (%delta, positive = Evie-Amplify better; Holm-Bonferroni p across batch sizes)")
    log(f"  {'Batch':<8}{'%delta':>12}{'p_holm':>14}")
    for b in batch_sizes:
        d = delta_pct.get(b, float('nan'))
        p = corrected_p.get(b, float('nan'))
        log(f"  {b:<8}{d:>+11.2f}%{p:>14.3e}")
    log_sep("=", 160)

# ==============================================================================
#  CSV
# ==============================================================================
def write_csv(raw_by_batch_opt_seed, opt_names, seeds, lr_used_by_batch, batch_sizes):
    fieldnames = ['batch_size', 'opt', 'seed', 'lr_used'] + METRIC_KEYS
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b in batch_sizes:
            for opt in opt_names:
                for seed in seeds:
                    r = raw_by_batch_opt_seed.get(b, {}).get(opt, {}).get(seed)
                    if r is None:
                        continue
                    row = {'batch_size': b, 'opt': opt, 'seed': seed,
                           'lr_used': lr_used_by_batch.get(b, {}).get(opt, '')}
                    row.update({k: r.get(k, float('nan')) for k in METRIC_KEYS})
                    writer.writerow(row)
    log(f"[CSV] Saved -> {CSV_PATH}")

# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    log_sep("=")
    log(f"  RUN 2 (VISION) — BATCH-SIZE SWEEP  |  device={device}  torch={torch.__version__}")
    log(f"  batch sizes: {BATCH_SIZES}  |  full-run steps: {FULL_STEPS_BY_BATCH}")
    log_sep("=")

    ckpt = load_checkpoint()
    build_tuning_subset()

    raw_by_batch = {}
    lr_used_by_batch = {}

    for batch_size in BATCH_SIZES:
        bkey = str(batch_size)
        full_steps = FULL_STEPS_BY_BATCH[batch_size]
        n_search_steps = round(LR_SEARCH_SAMPLES_SEEN / batch_size)
        log_sep("=")
        log(f"  BATCH SIZE {batch_size}  |  lr-search steps={n_search_steps} (samples seen={n_search_steps*batch_size})"
            f"  |  full-run steps={full_steps} (samples seen={full_steps*batch_size})")
        log_sep("=")

        bckpt = ckpt["by_batch"].setdefault(bkey, {"lr_used": {}, "completed": {}, "raw": {}})

        # ---- 1. LR search at this batch size (resume if checkpointed) ----
        if not bckpt["lr_used"]:
            lr_used = {}
            for opt_name in OPT_NAMES:
                try:
                    lr_used[opt_name] = lr_grid_search(opt_name, batch_size,
                                                         MODEL_WIDTH_DEFAULT, n_search_steps)
                except Exception as e:
                    log(f"[ERROR] lr_grid_search({opt_name}) crashed: {e}; using 1e-3")
                    lr_used[opt_name] = 1e-3
            bckpt["lr_used"] = lr_used
            save_checkpoint(ckpt)
        else:
            lr_used = bckpt["lr_used"]
        log(f"[LR-SEARCH batch={batch_size}] Frozen LRs: {lr_used}")
        lr_used_by_batch[batch_size] = lr_used

        # ---- 2. full 3-seed sweep at this batch size, checkpointed per (opt,seed) ----
        raw = {opt: {} for opt in OPT_NAMES}
        for opt in OPT_NAMES:
            for seed in SEEDS:
                key = f"{opt}|{seed}"
                if key in bckpt["raw"]:
                    raw[opt][seed] = bckpt["raw"][key]

        total = len(OPT_NAMES) * len(SEEDS)
        done = sum(len(v) for v in raw.values())
        log(f"[SWEEP batch={batch_size}] {done}/{total} runs already complete (resumed).")

        for opt in OPT_NAMES:
            for seed in SEEDS:
                key = f"{opt}|{seed}"
                if key in bckpt["completed"]:
                    continue
                lr = lr_used[opt]
                try:
                    r = run_experiment(opt, lr, seed, batch_size=batch_size,
                                        width=MODEL_WIDTH_DEFAULT, max_steps=full_steps)
                except Exception as e:
                    log(f"[ERROR] run_experiment({opt}, seed={seed}) crashed: {e}")
                    import traceback; traceback.print_exc()
                    r = {
                        'test_top1': float('nan'), 'test_top5': float('nan'),
                        'test_loss': float('nan'), 'val_loss': float('nan'),
                        'wall_s': 0.0, 'peak_mem_mb': 0.0,
                        'gflops': 0.0, 'exaflops': 0.0,
                        'mean_step_sz': float('nan'),
                        'clip_trigger_freq': float('nan'),
                        'hess_diag_norm': float('nan'), 'hess_diag_cv': float('nan'),
                        'regret': float('nan'),
                        'n_params': 0, 'lr_used': lr,
                    }
                raw[opt][seed] = r
                bckpt["raw"][key] = r
                bckpt["completed"][key] = True
                save_checkpoint(ckpt)
                log(f"[PROGRESS batch={batch_size}] {opt} seed={seed} done "
                    f"(top1={r['test_top1']:.2f}%).")

        raw_by_batch[batch_size] = raw

    # ---- 3. aggregate + report across all batch sizes ----
    write_csv(raw_by_batch, OPT_NAMES, SEEDS, lr_used_by_batch, BATCH_SIZES)
    agg_by_batch_opt = {b: {opt: aggregate_seed_results(list(raw_by_batch[b][opt].values()))
                             for opt in OPT_NAMES}
                         for b in BATCH_SIZES}
    with open(JSON_PATH, 'w') as jf:
        json.dump({
            'metadata': {'seeds': SEEDS, 'batch_sizes': BATCH_SIZES,
                         'full_steps_by_batch': FULL_STEPS_BY_BATCH,
                         'lr_search_samples_seen': LR_SEARCH_SAMPLES_SEEN,
                         'model_width': MODEL_WIDTH_DEFAULT,
                         'tuning_classes': _TUNING_CACHE['classes'],
                         'lr_used_by_batch': lr_used_by_batch, 'timestamp': datetime.now().isoformat()},
            'agg_by_batch_opt': agg_by_batch_opt,
            'raw_by_batch': raw_by_batch,
        }, jf, indent=2)
    log(f"[JSON] Saved -> {JSON_PATH}")

    top1_stats, delta_pct, corrected_p = build_batch_trend_tables(raw_by_batch, OPT_NAMES, SEEDS, BATCH_SIZES)
    print_batch_trend_tables(top1_stats, delta_pct, corrected_p, OPT_NAMES, BATCH_SIZES)

    log_sep("=")
    log(f"  ALL DONE  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_sep("=")
    _log_fh.close()

if __name__ == '__main__':
    main()
