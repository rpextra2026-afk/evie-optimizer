"""
vision_benchmark_run1.py
================================================================================
RUN 1 — HEADLINER BENCHMARK (PART A)
Evie vs Evie-Amplify vs AdamW vs Sophia vs AdaHessian
CIFAR-100 -> lightweight 2-conv-layer CNN image classification
Multi-seed (42, 43, 44), batch=32 fixed, 5000 steps.
================================================================================
"""
# ==============================================================================
# E1 COPY (AdaBelief baseline, PART D). Generated from the part-A script by
# evie_exp/scripts/build_e1_scripts.py; every difference from the source is listed in
# evie_exp/e1_diffs/*.diff. Experiment code (data, model, loop, LR search, aggregation) is untouched.
# Adds: AdaBelief, 9h55 budget + resume-from-zip, error handling, heartbeat every 10 min.
# ==============================================================================
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
#  GLOBAL CONFIG  (per spec)
# ==============================================================================
SEEDS               = [42, 43, 44]
MAX_STEPS           = 5000
EVAL_EVERY          = 100
BATCH_SIZE_DEFAULT  = 32
MODEL_WIDTH_DEFAULT = (32, 64)
HESS_FREQ           = 5
N_HUTCH             = 4
GRAD_CLIP_NORM      = 5.0
WEIGHT_DECAY        = 1e-4
TARGET_EPS_FRAC     = 0.1
SOPHIA_BETA_H       = 0.99


LR_GRID             = [1e-4, 3e-4, 1e-3, 3e-3]
LR_SEARCH_STEPS     = 100
CALIBRATION_SEEDS   = [42, 43, 44]
N_TUNING_SUPERCLASSES = 10
N_TUNING_FINE_PER_SUPER = 2


CLIP_METRIC_THRESHOLD = 1.0
TARGET_WP            = 0.85
WELL_POS_MAX         = 8.0
CALIBRATE_STEP       = 125
FAST_EMA_BETA        = 0.9


SPLIT_SEED           = 2024
NUM_CLASSES_MAIN     = 100


device = torch.device(os.environ.get("EVIE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
if device.type == "cuda":
    torch.cuda.set_device(device)   # so peak-memory stats read the device actually used (EVIE_DEVICE=cuda:1)


DATA_ROOT  = "/tmp/cifar100_data"
OUT_DIR    = os.environ.get("EVIE_E1_OUT") or (("/kaggle/working/evie_e1_vision") if os.path.isdir("/kaggle/working") else "/mnt/user-data/outputs")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH  = os.path.join(OUT_DIR, "vision_run1d_adabelief_checkpoint.json")
CSV_PATH   = os.path.join(OUT_DIR, "vision_run1d_adabelief_results_raw.csv")
JSON_PATH  = os.path.join(OUT_DIR, "vision_run1d_adabelief_results_agg.json")
LOG_PATH   = os.path.join(OUT_DIR, "vision_run1d_adabelief_log.txt")


# ==============================================================================
#  E1 RESUME-FROM-PREVIOUS-SESSION (runs before the log file is opened)
#  Looks for the checkpoint of an earlier session in: OUT_DIR, $EVIE_RESUME_DIRS (colon separated) and
#  everything under /kaggle/input (loose files or *.zip packs written by this script). Picks the one
#  with the most completed units and restores checkpoint + log + data cache into OUT_DIR.
# ==============================================================================
import signal, threading, zipfile, shutil, gc
E1_DOMAIN = "vision"
TIME_BUDGET_MIN = float(os.environ.get("EVIE_TIME_BUDGET_MIN", "595"))        # 9 h 55 min
ELAPSED_OFFSET_MIN = float(os.environ.get("EVIE_ELAPSED_OFFSET_MIN", "0"))    # minutes of this Kaggle session already used
_T_START = time.time() - ELAPSED_OFFSET_MIN * 60.0
SAFETY_S = 120.0                    # margin kept for the final checkpoint + zip
MAX_UNIT_ATTEMPTS = 3               # a unit that fails this many times (over all sessions) is recorded as NaN
E1_DATA_CACHE = os.path.join(OUT_DIR, E1_DOMAIN + "_e1_data_cache.npz")
_E1_KNOWN = [os.path.basename(p) for p in (CKPT_PATH, CKPT_PATH + ".bak", LOG_PATH, CSV_PATH, JSON_PATH, E1_DATA_CACHE)]
_PACK_DIR = os.path.dirname(OUT_DIR) if os.path.abspath(OUT_DIR).startswith("/kaggle/working/") else OUT_DIR
PACK_PATH = os.path.join(_PACK_DIR, "evie_e1_" + E1_DOMAIN + "_pack.zip")
_EARLY = []                          # messages produced before log() exists


def _ckpt_done(d):
    try:
        return len(d.get("completed", {}))
    except Exception:
        return -1


def _find_resume_candidates():
    base = os.path.basename(CKPT_PATH)
    roots = [OUT_DIR] + [p for p in os.environ.get("EVIE_RESUME_DIRS", "").split(":") if p]
    if os.path.isdir("/kaggle/input"):
        roots.append("/kaggle/input")
    found, seen = [], set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, dn, fn in os.walk(root):
            if dp[len(root):].count(os.sep) > 6:
                dn[:] = []
                continue
            for f in fn:
                full = os.path.join(dp, f)
                if full in seen:
                    continue
                seen.add(full)
                try:
                    if f == base:
                        with open(full) as fh:
                            d = json.load(fh)
                        found.append({"kind": "file", "path": full, "done": _ckpt_done(d), "mtime": os.path.getmtime(full),
                                      "own": os.path.abspath(dp) == os.path.abspath(OUT_DIR)})
                    elif f.endswith(".zip"):
                        with zipfile.ZipFile(full) as z:
                            for nm in z.namelist():
                                if os.path.basename(nm) == base:
                                    d = json.loads(z.read(nm))
                                    found.append({"kind": "zip", "path": full, "member": nm, "done": _ckpt_done(d),
                                                  "mtime": os.path.getmtime(full), "own": False})
                except Exception as e:
                    _EARLY.append(f"[RESUME] ignoring unreadable candidate {full}: {e!r}")
    return found


def _restore_from_previous_session():
    info = {"source": None, "done": 0}
    cands = _find_resume_candidates()
    for c in sorted(cands, key=lambda c: -c["done"]):
        _EARLY.append(f"[RESUME] candidate: {c['kind']} {c['path']}" + (f"::{c['member']}" if c.get("member") else "")
                      + f"  completed_units={c['done']}")
    if not cands:
        _EARLY.append("[RESUME] no previous checkpoint found anywhere -> starting FRESH")
        return info
    best = sorted(cands, key=lambda c: (c["done"], c["own"], c["mtime"]))[-1]
    info.update(source=best["path"], done=best["done"])
    if best["own"]:
        _EARLY.append(f"[RESUME] using the checkpoint already in OUT_DIR ({best['done']} units done) -> CONTINUING from it")
        return info
    n_copied = 0
    if best["kind"] == "file":
        src_dir = os.path.dirname(best["path"])
        for nm in _E1_KNOWN:
            s, d = os.path.join(src_dir, nm), os.path.join(OUT_DIR, nm)
            if os.path.isfile(s) and (nm == os.path.basename(CKPT_PATH) or not os.path.exists(d)):
                shutil.copy2(s, d); n_copied += 1
    else:
        with zipfile.ZipFile(best["path"]) as z:
            for nm in z.namelist():
                b = os.path.basename(nm)
                d = os.path.join(OUT_DIR, b)
                if b in _E1_KNOWN and (b == os.path.basename(CKPT_PATH) or not os.path.exists(d)):
                    with open(d, "wb") as fh:
                        fh.write(z.read(nm))
                    n_copied += 1
    _EARLY.append(f"[RESUME] restored {n_copied} file(s) from {best['path']} into {OUT_DIR}; "
                  f"{best['done']} units already complete -> the sweep will CONTINUE from that checkpoint")
    return info


try:
    RESUME_INFO = _restore_from_previous_session()
except Exception as _e:   # a failed restore must never turn into a silent fresh start
    print(f"[RESUME][FATAL] could not inspect/restore previous sessions: {_e!r}. Aborting instead of starting fresh "
          f"(set EVIE_RESUME_DIRS or fix permissions).", flush=True)
    raise SystemExit(5)


_log_fh = open(LOG_PATH, "a", buffering=1)
def log(msg):
   line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
   print(line, flush=True)
   _log_fh.write(line + "\n")
def log_sep(c="=", w=140):
   log(c * w)

for _m in _EARLY:
    log(_m)
log(f"[E1] domain={E1_DOMAIN} OUT_DIR={OUT_DIR} budget={TIME_BUDGET_MIN:.0f} min (offset {ELAPSED_OFFSET_MIN:.0f}) pack={PACK_PATH}")


# ==============================================================================
#  E1 SESSION / BUDGET / CHECKPOINT / ERROR-HANDLING INFRASTRUCTURE (added; not part of the experiment)
# ==============================================================================
class _SessionEnd(Exception):
    """Raised to leave the sweep cleanly: the checkpoint is saved and the zip written by main()."""
    def __init__(self, reason, code=0):
        super().__init__(reason); self.reason, self.code = reason, code


_STOP = {"requested": False, "n": 0}
_PROGRESS = {"done": 0, "total": 0, "current": "-", "failed": 0}
_UNIT_WALL = []
_CONSEC_FAIL = [0]
_SESSION_RAN = [0]


def _elapsed_s():
    return time.time() - _T_START


def _budget_left_s():
    return TIME_BUDGET_MIN * 60.0 - _elapsed_s() - SAFETY_S


def _check_budget(next_key, default_est_s):
    """Raise _SessionEnd if a stop was requested or the next unit would not finish inside the budget."""
    est = 1.3 * max(_UNIT_WALL[-20:]) if _UNIT_WALL else default_est_s
    if _STOP["requested"]:
        raise _SessionEnd("stopped by signal (SIGINT/SIGTERM) between units")
    if _budget_left_s() < est:
        raise _SessionEnd(f"BUDGET REACHED: {_budget_left_s() / 60:.1f} min left < estimated {est / 60:.1f} min for {next_key}")


def _on_signal(signum, frame):
    _STOP["n"] += 1
    _STOP["requested"] = True
    log(f"[SIGNAL] {signum} received: finishing the current unit, then saving + zipping (send again to abort now)")
    if _STOP["n"] >= 2:
        raise KeyboardInterrupt


def _install_signal_handlers():
    if threading.current_thread() is threading.main_thread():
        for s in (signal.SIGINT, signal.SIGTERM):
            try: signal.signal(s, _on_signal)
            except Exception: pass


def _start_heartbeat(every_s=600.0):
    def loop():
        t0 = time.time()
        while True:
            time.sleep(every_s)
            print(f"[HEARTBEAT {(time.time() - t0) / 60:6.1f} min] {E1_DOMAIN}: {_PROGRESS['done']}/{_PROGRESS['total']} units done, "
                  f"current={_PROGRESS['current']}, failed={_PROGRESS['failed']}, budget left {_budget_left_s() / 60:.0f} min", flush=True)
    threading.Thread(target=loop, daemon=True, name="e1-heartbeat").start()


def write_resume_pack(reason=""):
    """Zip checkpoint(+.bak), log, csv/json, data cache -> PACK_PATH (atomic). Never raises."""
    try:
        try: _log_fh.flush()
        except Exception: pass
        tmp = PACK_PATH + ".tmp"
        n = 0
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for nm in _E1_KNOWN:
                p = os.path.join(OUT_DIR, nm)
                if os.path.isfile(p):
                    z.write(p, nm); n += 1
            z.writestr("E1_PACK_MANIFEST.json", json.dumps({"domain": E1_DOMAIN, "reason": reason,
                       "created": datetime.now().isoformat(), "files": n}))
        os.replace(tmp, PACK_PATH)
        log(f"[PACK] wrote {PACK_PATH} ({n} files, {os.path.getsize(PACK_PATH) / 2**20:.1f} MB)")
        return PACK_PATH
    except Exception as e:
        log(f"[ERROR] could not write the resume pack: {e!r}")
        return None


def _start_watchdog():
    """If a unit hangs past budget+10 min, write the pack from the on-disk checkpoint and hard-exit."""
    def fire():
        log("[WATCHDOG] budget + 10 min exceeded while a unit was still running: writing pack from the last checkpoint and exiting")
        write_resume_pack("watchdog")
        os._exit(0)
    delay = TIME_BUDGET_MIN * 60.0 + 600.0 - _elapsed_s()
    t = threading.Timer(max(delay, 1.0), fire)
    t.daemon = True
    t.start()


def _retry(fn, what, tries=4, base_s=5.0):
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            log(f"[RETRY] {what} failed ({e!r}); attempt {k + 1}/{tries}")
            time.sleep(base_s * (k + 1))
    raise RuntimeError(f"{what} failed after {tries} attempts: {last!r}")


def run_guarded(fn, label, *args, **kwargs):
    """Run fn; on an exception log the traceback, free GPU memory, retry once. Returns (result, None) or (None, error)."""
    import traceback
    last = None
    for attempt in (1, 2):
        try:
            return fn(*args, **kwargs), None
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last = e
            log(f"[ERROR] {label}: attempt {attempt}/2 failed: {e!r}")
            log(traceback.format_exc())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)
    return None, last


def _run_unit_guarded(ckpt, key, nan_result_fn, fn, *args, **kwargs):
    """Run one checkpoint unit with error handling. Returns the result, or None if the unit failed this session
    (it is then retried next session, up to MAX_UNIT_ATTEMPTS in total; the 3rd failure is stored as NaN)."""
    t0 = time.time()
    r, err = run_guarded(fn, key, *args, **kwargs)
    if err is None:
        _UNIT_WALL.append(time.time() - t0)
        _CONSEC_FAIL[0] = 0
        _SESSION_RAN[0] += 1
        return r
    f = ckpt.setdefault("failed", {}).setdefault(key, {"attempts": 0})
    f["attempts"] += 1
    f["last_error"] = repr(err)
    _PROGRESS["failed"] += 1
    _CONSEC_FAIL[0] += 1
    save_checkpoint(ckpt)
    if f["attempts"] >= MAX_UNIT_ATTEMPTS:
        log(f"[PERMANENT FAILURE] {key} failed {f['attempts']} times; stored as NaN (it will be counted and reported, not hidden)")
        return nan_result_fn()
    log(f"[FAILED] {key} (attempt {f['attempts']}/{MAX_UNIT_ATTEMPTS}); it will be retried when this script is rerun")
    if _CONSEC_FAIL[0] >= 5:
        raise _SessionEnd("circuit breaker: 5 consecutive unit failures (likely a code/data problem, not a flaky run)", 3)
    return None


def _check_result(key, r, primary_key):
    v = r.get(primary_key, float("nan"))
    if isinstance(v, float) and math.isnan(v):
        log(f"[WARN] {key}: primary metric {primary_key} is NaN (diverged run); kept as in the original scripts and counted by the comparison")


def begin_session(ckpt, total):
    ss = ckpt.setdefault("sessions", [])
    n_done = len(ckpt.get("completed", {}))
    ss.append({"n": len(ss) + 1, "start": datetime.now().isoformat(), "resumed_from": RESUME_INFO["source"],
               "completed_at_start": n_done, "budget_min": TIME_BUDGET_MIN, "elapsed_offset_min": ELAPSED_OFFSET_MIN})
    _PROGRESS["total"], _PROGRESS["done"] = total, n_done
    if len(ss) > 1 or n_done:
        log(f"[SESSION {len(ss)}] RESUMED FROM CHECKPOINT: {n_done}/{total} units were already complete "
            f"(source: {RESUME_INFO['source'] or 'OUT_DIR'}); only the remaining units will run")
    else:
        log(f"[SESSION 1] fresh start (no checkpoint); {total} units to run")
    save_checkpoint(ckpt)


def finish_session(ckpt, reason, code=0):
    ss = ckpt.get("sessions", [])
    if ss:
        ss[-1].update(end=datetime.now().isoformat(), reason=reason, units_run=_SESSION_RAN[0],
                      elapsed_min=round(_elapsed_s() / 60, 1), completed_at_end=len(ckpt.get("completed", {})))
    save_checkpoint(ckpt)
    left = _PROGRESS["total"] - len(ckpt.get("completed", {}))
    log_sep("=")
    log(f"[SESSION END] {reason} | ran {_SESSION_RAN[0]} unit(s) this session | {len(ckpt.get('completed', {}))}/{_PROGRESS['total']} complete | {left} left")
    pack = write_resume_pack(reason)
    if left > 0:
        log(f"[NEXT] DOWNLOAD {pack}, attach it (as a dataset or as the previous notebook's output) to the next session and rerun this same script: it resumes from the checkpoint.")
    else:
        log("[NEXT] nothing left to run.")
    log_sep("=")
    return code




def set_seed(seed):
   torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)


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
   train_ds = _retry(lambda: torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=True), "CIFAR-100 train download")
   test_ds  = _retry(lambda: torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True), "CIFAR-100 test download")
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


def _spatial_avg(h):
   if h.dim() == 4:
       return h.mean(dim=[2,3], keepdim=True).expand_as(h).clone()
   elif h.dim() == 2:
       return h.mean(dim=1, keepdim=True).expand_as(h).clone()
   elif h.dim() == 1:
       return h.mean().expand_as(h).clone()
   return h.clone()


class EvieOptimizer(torch.optim.Optimizer):
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
           if group['calibrated']: continue
           target = group['target_wp']
           sigma2_list = []
           for p in group['params']:
               s = self.state.get(p, {})
               if 'fast_m1' not in s: continue
               sigma2 = (s['fast_m2'] - s['fast_m1'].pow(2)).clamp_(min=0.0)
               sigma2_list.append(sigma2.mean().item())
           if not sigma2_list: continue
           mean_s2 = max(float(np.mean(sigma2_list)), 1e-12)
           new_g0 = float(np.clip((1.0 / target - 1.0) / mean_s2, 10.0, 5000.0))
           group['gamma0'] = new_g0
           group['calibrated'] = True


   @torch.no_grad()
   def step(self, closure=None):
       loss = None
       if closure is not None:
           with torch.enable_grad(): loss = closure()
       for group in self.param_groups:
           lr, g0, b1, b2, eps, wd = (group['lr'], group['gamma0'], group['beta1'],
                                       group['beta2'], group['eps'], group['weight_decay'])
           fb = group['fast_beta']
           for p in group['params']:
               if p.grad is None: continue
               if wd > 0.0: p.data.mul_(1.0 - lr * wd)
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
           if group['calibrated']: continue
           target = group['target_wp']
           sigma2_list = []
           for p in group['params']:
               s = self.state.get(p, {})
               if 'fast_m1' not in s: continue
               sigma2 = (s['fast_m2'] - s['fast_m1'].pow(2)).clamp_(min=0.0)
               sigma2_list.append(sigma2.mean().item())
           if not sigma2_list: continue
           mean_s2 = max(float(np.mean(sigma2_list)), 1e-12)
           new_g0 = float(np.clip((1.0 / target - 1.0) / mean_s2, 10.0, 5000.0))
           group['gamma0'] = new_g0
           group['calibrated'] = True


   @torch.no_grad()
   def step(self, closure=None):
       loss = None
       if closure is not None:
           with torch.enable_grad(): loss = closure()
       for group in self.param_groups:
           lr, g0, b1, b2 = group['lr'], group['gamma0'], group['beta1'], group['beta2']
           eps, wd, target = group['eps'], group['weight_decay'], group['target_wp']
           wp_max = group['well_pos_max']
           fb = group['fast_beta']
           for p in group['params']:
               if p.grad is None: continue
               if wd > 0.0: p.data.mul_(1.0 - lr * wd)
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
           if 'h_hat' not in s: s['h_hat'] = h_new.clone()
           else: s['h_hat'].mul_(beta_h).add_(h_new, alpha=1.0 - beta_h)
           sq_sum += h_new.pow(2).sum().item()
       self.last_h_norm = math.sqrt(sq_sum)


   @torch.no_grad()
   def step(self, closure=None):
       loss = None
       if closure is not None:
           with torch.enable_grad(): loss = closure()
       for group in self.param_groups:
           lr, b1, rho, eps, wd = (group['lr'], group['beta1'], group['rho'], group['eps'], group['weight_decay'])
           for p in group['params']:
               if p.grad is None: continue
               if wd > 0.0: p.data.mul_(1.0 - lr * wd)
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
           if p not in self.state: self.state[p] = {}
           s = self.state[p]
           h_smooth = _spatial_avg(h.detach().abs()).clamp(min=1e-8)
           if 'h_hat' not in s: s['h_hat'] = h_smooth.clone()
           else: s['h_hat'].mul_(beta_h).add_(h_smooth, alpha=1.0 - beta_h)
           sq_sum += h_smooth.pow(2).sum().item()
       self.last_h_norm = math.sqrt(sq_sum)


   @torch.no_grad()
   def step(self, closure=None):
       loss = None
       if closure is not None:
           with torch.enable_grad(): loss = closure()
       for group in self.param_groups:
           lr, b1, b2, eps, wd, k = (group['lr'], group['beta1'], group['beta2'],
                                      group['eps'], group['weight_decay'], group['hessian_power'])
           for p in group['params']:
               if p.grad is None: continue
               if wd > 0.0: p.data.mul_(1.0 - lr * wd)
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


class AdaBeliefOptimizer(torch.optim.Optimizer):
    """AdaBelief (Zhuang et al. 2020), decoupled weight decay, no rectification, no amsgrad.
    Matches adabelief_pytorch 0.2.1 with weight_decouple=True, rectify=False, eps=1e-16:
        p <- p*(1-lr*wd);  m <- b1 m + (1-b1) g;  s <- b2 s + (1-b2)(g-m)^2 + eps (in place, as the reference)
        p <- p - (lr/bc1) * m / (sqrt(s)/sqrt(bc2) + eps)"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group['lr'], group['betas'], group['eps'], group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad.data
                if wd > 0.0: p.data.mul_(1.0 - lr * wd)
                s = self.state[p]
                if 'step' not in s:
                    s['step'] = 0
                    s['exp_avg'] = torch.zeros_like(p.data)
                    s['exp_avg_var'] = torch.zeros_like(p.data)
                s['step'] += 1
                t = s['step']
                bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
                m, v = s['exp_avg'], s['exp_avg_var']
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                resid = g - m
                v.mul_(b2).addcmul_(resid, resid, value=1.0 - b2).add_(eps)
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.data.addcdiv_(m, denom, value=-lr / bc1)
        return loss


def build_optimizer(opt_name, model, lr):
   if opt_name == "Evie":
       return EvieOptimizer(model.parameters(), lr=lr, gamma0=1.0, target_wp=TARGET_WP, weight_decay=WEIGHT_DECAY)
   elif opt_name == "Evie-Amplify":
       return EvieAmplifyOptimizer(model.parameters(), lr=lr, gamma0=1.0, target_wp=TARGET_WP, weight_decay=WEIGHT_DECAY, well_pos_max=WELL_POS_MAX)
   elif opt_name == "Sophia":
       return SophiaOptimizer(model.parameters(), lr=lr, betas=(0.965, 0.99), rho=0.04, weight_decay=WEIGHT_DECAY)
   elif opt_name == "AdaHessian":
       return AdaHessianOptimizer(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-4, weight_decay=WEIGHT_DECAY, hessian_power=1.0)
   elif opt_name == "AdaBelief":
       return AdaBeliefOptimizer(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-16, weight_decay=WEIGHT_DECAY)
   else:
       return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)


OPT_NAMES = ["AdaBelief"]  # PART D (E1): AdaBelief baseline column
HESSIAN_CONSUMERS = {"Sophia", "AdaHessian"}


def hutchinson_hessian(model, params, X, y, criterion, n_hutch=N_HUTCH, optimizer=None, feed_optimizer=False):
   with torch.enable_grad(): loss_h = criterion(model(X), y)
   grads = torch.autograd.grad(loss_h, params, create_graph=True, retain_graph=True, allow_unused=True)
   h_acc = [torch.zeros_like(p.data) for p in params]
   for k in range(n_hutch):
       vs = [(torch.randint(0, 2, p.shape, device=p.device, dtype=torch.float32) * 2 - 1).to(p.dtype) for p in params]
       gv = sum((g * v).sum() for g, v in zip(grads, vs) if g is not None)
       retain = (k < n_hutch - 1)
       hvps = torch.autograd.grad(gv, params, retain_graph=retain, allow_unused=True)
       for i, (v, hv) in enumerate(zip(vs, hvps)):
           if hv is not None: h_acc[i].add_(v * hv.detach())
       del vs, gv, hvps
   del grads
   if torch.cuda.is_available(): torch.cuda.empty_cache()
   for h in h_acc: h.div_(n_hutch); h.abs_()
   if feed_optimizer and optimizer is not None:
       optimizer.set_hessian(list(zip(params, h_acc)), beta_h=SOPHIA_BETA_H)
   return h_acc


def reset_peak_memory():
   if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()


def peak_memory_mb():
   if torch.cuda.is_available(): return torch.cuda.max_memory_allocated() / (1024 ** 2)
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


def _lr_grid_search_unsafe(opt_name, batch_size, width):
   tuning_classes = build_tuning_subset()
   val_by_lr = defaultdict(list)
   for lr in LR_GRID:
       for cal_seed in CALIBRATION_SEEDS:
           set_seed(cal_seed)
           train_loader, val_loader, _ = get_tuning_loaders(batch_size)
           model = LightCNN(num_classes=len(tuning_classes), width=width).to(device)
           criterion = nn.CrossEntropyLoss()
           optimizer = build_optimizer(opt_name, model, lr)
           assert type(optimizer).__name__ == 'AdaBeliefOptimizer', type(optimizer).__name__
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
                   hutchinson_hessian(model, params, X_batch, y_batch, criterion, N_HUTCH, optimizer=optimizer, feed_optimizer=True)
               optimizer.step()
           val_loss, _, _ = evaluate(model, val_loader, criterion)
           if not math.isnan(val_loss):
               val_by_lr[lr].append(val_loss)
   median_by_lr = {lr: (float(np.median(v)) if v else float('inf')) for lr, v in val_by_lr.items()}
   best_lr = min(median_by_lr, key=median_by_lr.get)
   log(f"  [LR-SEARCH] {opt_name} (width={width}): medians={ {f'{k:.0e}': round(v,6) for k,v in median_by_lr.items()} } -> selected lr={best_lr:.0e}")
   return best_lr


def _run_experiment_unsafe(opt_name, lr, seed, batch_size=BATCH_SIZE_DEFAULT, width=MODEL_WIDTH_DEFAULT, max_steps=MAX_STEPS):
   set_seed(seed)
   reset_peak_memory()
   train_loader, val_loader, test_loader = get_main_loaders(batch_size)
   model = LightCNN(num_classes=NUM_CLASSES_MAIN, width=width).to(device)
   criterion = nn.CrossEntropyLoss()
   n_params = count_params(model)
   params = [p for p in model.parameters() if p.requires_grad]


   optimizer = build_optimizer(opt_name, model, lr)
   assert type(optimizer).__name__ == 'AdaBeliefOptimizer', type(optimizer).__name__
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.1)


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
   val_history = []
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


       ran_probe = (step % HESS_FREQ == 0)
       h_list = None
       if ran_probe:
           h_list = hutchinson_hessian(model, params, X_batch, y_batch, criterion, N_HUTCH, optimizer=optimizer, feed_optimizer=is_hessian_consumer)
           if is_hessian_consumer and not math.isnan(optimizer.last_h_norm):
               hess_norms.append(optimizer.last_h_norm)
           else:
               hess_norms.append(math.sqrt(sum(h.pow(2).sum().item() for h in h_list)))


       step_flops = fwd_flops_per_batch * 3
       if ran_probe: step_flops += N_HUTCH * fwd_flops_per_batch * 3
       cumulative_flops += step_flops


       optimizer.step()
       scheduler.step()


       if is_evie_family and not calibrated_once and step == CALIBRATE_STEP:
           optimizer.calibrate_gamma0()
           calibrated_once = True


       with torch.no_grad():
           delta = torch.cat([(p.data - pb).flatten() for p, pb in zip(model.parameters(), params_before)])
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
           val_history.append((step, val_top1))
           model.train()


   if best_state is not None:
       model.load_state_dict(best_state)
   test_loss, test_top1, test_top5 = evaluate(model, test_loader, criterion)


   wall = time.time() - t0
   mem = peak_memory_mb()
   mean_step_size = float(np.mean(step_sizes)) if step_sizes else float('nan')
   gflops_st = cumulative_flops / max(max_steps, 1) / 1e9
   exaflops = cumulative_flops / 1e18
   clipfreq = clip_trigger_count / max(max_steps, 1)


   if hess_norms:
       hessnorm = float(np.mean(hess_norms))
       hn_std = float(np.std(hess_norms))
       hesscv = hn_std / hessnorm if hessnorm > 0 else float('nan')
   else:
       hessnorm = hesscv = float('nan')


   return {
       'top1': test_top1, 'top5': test_top5, 'valloss': best_val_loss,
       'iterstotgt': float('nan'),
       'wall': wall, 'peakmemmb': mem,
       'gflops_st': gflops_st, 'exaflops': exaflops,
       'stepsize': mean_step_size, 'clipfreq': clipfreq,
       'hessnorm': hessnorm, 'hesscv': hesscv, 'regret': cum_regret,
       'val_history': val_history,
       'n_params': n_params, 'lr_used': lr,
   }


def compute_iters_to_target(val_history, target_top1, max_steps):
   if target_top1 is None or math.isnan(target_top1): return float('nan')
   for step, v in val_history:
       if v >= target_top1: return step
   return max_steps


METRIC_KEYS = ['top1', 'top5', 'valloss', 'iterstotgt', 'wall', 'peakmemmb',
              'gflops_st', 'exaflops', 'stepsize', 'clipfreq', 'hessnorm', 'hesscv', 'regret']
METRIC_LABELS = {
   'top1': 'top1(%)', 'top5': 'top5(%)', 'valloss': 'valloss',
   'iterstotgt': 'iterstotgt', 'wall': 'wall(s)', 'peakmemmb': 'peakmemmb',
   'gflops_st': 'gflops/st', 'exaflops': 'exaflops', 'stepsize': 'stepsize',
   'clipfreq': 'clipfreq', 'hessnorm': 'hessnorm', 'hesscv': 'hesscv', 'regret': 'regret'
}
LOWER_BETTER = {
   'top1': False, 'top5': False, 'valloss': True, 'iterstotgt': True,
   'wall': True, 'peakmemmb': True, 'gflops_st': True, 'exaflops': True,
   'stepsize': None, 'clipfreq': True, 'hessnorm': None, 'hesscv': True, 'regret': True
}


def load_checkpoint():
    """Primary file, else the .bak copy; a corrupt file is quarantined (never silently ignored)."""
    for path, tag in ((CKPT_PATH, "primary"), (CKPT_PATH + ".bak", "backup")):
        if os.path.exists(path):
            try:
                with open(path) as f:
                    d = json.load(f)
                if not isinstance(d, dict) or "completed" not in d or "raw" not in d:
                    raise ValueError("missing keys")
                if tag == "backup":
                    log(f"[WARN] primary checkpoint unreadable; using the .bak copy ({len(d['completed'])} units)")
                return d
            except Exception as e:
                q = path + ".corrupt-" + datetime.now().strftime("%H%M%S")
                log(f"[WARN] checkpoint {path} unreadable ({e!r}); quarantined as {q}")
                try: os.replace(path, q)
                except OSError: pass
    return {"lr_used": {}, "completed": {}, "raw": {}}


def save_checkpoint(ckpt):
    tmp = CKPT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(CKPT_PATH):
        shutil.copy2(CKPT_PATH, CKPT_PATH + ".bak")
    os.replace(tmp, CKPT_PATH)


def aggregate_seeds(raw_by_opt_seed, opt_names, seeds):
   stats = {}
   for opt in opt_names:
       stats[opt] = {}
       for k in METRIC_KEYS:
           vals = [raw_by_opt_seed[opt][s][k] for s in seeds if not math.isnan(raw_by_opt_seed[opt][s][k])]
           stats[opt][k] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float('nan'), float('nan'))
   return stats


def holm_bonferroni(pvals_dict):
   items = [(k, p) for k, p in pvals_dict.items() if not math.isnan(p)]
   items_sorted = sorted(items, key=lambda x: x[1])
   m = len(items_sorted)
   corrected = {k: float('nan') for k in pvals_dict}
   for rank0, (k, p) in enumerate(items_sorted):
       corrected[k] = min(p * (m - rank0), 1.0)
   return corrected


def build_final_table(raw, opt_names, seeds):
   challenger = "Evie"
   stats = aggregate_seeds(raw, opt_names, seeds)
  
   winners = {}
   for k in METRIC_KEYS:
       lb = LOWER_BETTER.get(k)
       candidates = {opt: stats[opt][k][0] for opt in opt_names if not math.isnan(stats[opt][k][0])}
       if not candidates or lb is None:
           winners[k] = "n/a"
           continue
       winners[k] = min(candidates, key=candidates.get) if lb else max(candidates, key=candidates.get)


   raw_p, delta_pct = {}, {}
   can_compare = (challenger in opt_names) and ("AdamW" in opt_names)
   for k in METRIC_KEYS:
       if not can_compare:
           raw_p[k] = float('nan'); delta_pct[k] = float('nan')
           continue
       c_vals = [raw[challenger][s][k] for s in seeds if not math.isnan(raw[challenger][s][k])]
       b_vals = [raw["AdamW"][s][k] for s in seeds if not math.isnan(raw["AdamW"][s][k])]
       if len(c_vals) >= 3 and len(b_vals) >= 3:
           try:
               _, p = wilcoxon(c_vals, b_vals)
           except ValueError:
               p = float('nan')
       else:
           p = float('nan')
       raw_p[k] = p
      
       lb = LOWER_BETTER.get(k)
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
   return stats, winners, delta_pct, corrected_p


def print_final_table(stats, winners, delta_pct, corrected_p, opt_names):
   opt_col_w = max(18, max((len(o) for o in opt_names), default=18) + 4)
   log_sep("=", 200)
   log("  RUN 1 RESULTS  (mean +/- std across seeds)")
   log_sep("-", 200)
   header = f"  {'Metric':<18}"
   for opt in opt_names:
       header += f"{opt:>{opt_col_w}}"
   header += f"{'WINNER':>{opt_col_w}}"
   log(header)
   log("  " + "-" * (18 + opt_col_w * (len(opt_names) + 1)))
   for k in METRIC_KEYS:
       row = f"  {METRIC_LABELS[k]:<18}"
       for opt in opt_names:
           m, sd = stats[opt][k]
           cell = f"{m:.4g}+/-{sd:.2g}" if not math.isnan(m) else "nan"
           row += f"{cell:>{opt_col_w}}"
       row += f"{winners[k]:>{opt_col_w}}"
       log(row)
   log_sep("-", 200)
   log("  Challenger vs AdamW  (median %delta, direction-aware; Wilcoxon + Holm-Bonferroni)")
   can_compare = ("Evie" in opt_names) and ("AdamW" in opt_names)
   if can_compare:
       for k in METRIC_KEYS:
           d = delta_pct.get(k, float('nan'))
           p = corrected_p.get(k, float('nan'))
           log(f"    {METRIC_LABELS[k]:<18}  %delta={d:+7.2f}%   p_holm={p:.3e}")
   else:
       log("    (skipped: this run does not include both sides of the comparison)")
   log_sep("=", 200)


def write_csv(raw, opt_names, seeds, lr_used, target_top1):
   fieldnames = ['opt', 'seed', 'lr_used', 'target_top1'] + METRIC_KEYS
   with open(CSV_PATH, 'w', newline='') as f:
       writer = csv.DictWriter(f, fieldnames=fieldnames)
       writer.writeheader()
       for opt in opt_names:
           for seed in seeds:
               r = raw.get(opt, {}).get(seed)
               if r is None: continue
               row = {'opt': opt, 'seed': seed, 'lr_used': lr_used.get(opt, ''), 'target_top1': target_top1}
               row.update({k: r.get(k, float('nan')) for k in METRIC_KEYS})
               writer.writerow(row)
   log(f"[CSV] Saved -> {CSV_PATH}")




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
           'top1': float('nan'), 'top5': float('nan'), 'valloss': float('nan'),
           'iterstotgt': float('nan'), 'wall': 0.0, 'peakmemmb': 0.0,
           'gflops_st': 0.0, 'exaflops': 0.0, 'stepsize': float('nan'),
           'clipfreq': float('nan'), 'hessnorm': float('nan'), 'hesscv': float('nan'),
           'regret': float('nan'), 'val_history': [], 'n_params': 0, 'lr_used': 1e-3
       }


# ==============================================================================
#  E1 MAIN (replaces the original main(); same sequence: LR search -> runs -> table)
# ==============================================================================
def _nan_result_vision():
    r = {k: float('nan') for k in METRIC_KEYS}
    r.update(val_history=[], n_params=0, lr_used=float('nan'), wall=0.0, peakmemmb=0.0, _failed=True)
    return r


def _vision_body(ckpt):
    n_total = len(OPT_NAMES) * len(SEEDS)
    if not ckpt.get("lr_used"):
        build_tuning_subset()
        lr_used = {}
        for opt_name in OPT_NAMES:
            lr, err = run_guarded(_lr_grid_search_unsafe, f"LR search {opt_name}", opt_name, BATCH_SIZE_DEFAULT, MODEL_WIDTH_DEFAULT)
            if err is not None:
                raise _SessionEnd(f"LR search failed twice ({err!r}); refusing to fall back to a default LR", 2)
            lr_used[opt_name] = lr
        ckpt["lr_used"] = lr_used
        save_checkpoint(ckpt)
    else:
        lr_used = ckpt["lr_used"]
        log(f"[RESUME] LR search NOT repeated: frozen LRs restored from the checkpoint: {lr_used}")
    log(f"[LR-SEARCH] Frozen LRs: {lr_used}")
    target_top1 = float('nan')     # the AdamW-derived target is not available in this copy (see compare_adabelief.py)
    raw = {opt: {} for opt in OPT_NAMES}
    n_restored = sum(1 for o in OPT_NAMES for s in SEEDS if f"{o}|{s}" in ckpt["completed"])
    log(f"[SWEEP] {n_restored}/{n_total} runs already complete ({'RESUMING from checkpoint' if n_restored else 'fresh start'})")
    for opt in OPT_NAMES:
        for seed in SEEDS:
            key = f"{opt}|{seed}"
            if key in ckpt["completed"]:
                raw[opt][seed] = ckpt["raw"][key]
                log(f"  [RESUME] {key}: restored from checkpoint, not rerun")
                continue
            _PROGRESS["current"] = key
            _check_budget(key, 150.0)
            r = _run_unit_guarded(ckpt, key, _nan_result_vision, _run_experiment_unsafe, opt, lr_used[opt], seed,
                                  batch_size=BATCH_SIZE_DEFAULT, width=MODEL_WIDTH_DEFAULT, max_steps=MAX_STEPS)
            if r is None:
                continue
            _check_result(key, r, 'top1')
            raw[opt][seed] = r
            ckpt["raw"][key] = r
            ckpt["completed"][key] = True
            save_checkpoint(ckpt)
            _PROGRESS["done"] = len(ckpt["completed"])
            log(f"[PROGRESS {_PROGRESS['done']}/{n_total}] {key} done (optimizer class AdaBeliefOptimizer) top1={r['top1']:.2f} wall={r['wall']:.0f}s")
    missing = [f"{o}|{s}" for o in OPT_NAMES for s in SEEDS if f"{o}|{s}" not in ckpt["completed"]]
    if missing:
        raise _SessionEnd(f"incomplete: {missing} failed this session; rerun to retry them", 4)
    for opt in OPT_NAMES:
        for seed in SEEDS:
            if math.isnan(raw[opt][seed].get('iterstotgt', float('nan'))):
                raw[opt][seed]['iterstotgt'] = compute_iters_to_target(raw[opt][seed]['val_history'], target_top1, MAX_STEPS)
                ckpt["raw"][f"{opt}|{seed}"] = raw[opt][seed]
    save_checkpoint(ckpt)
    write_csv(raw, OPT_NAMES, SEEDS, lr_used, target_top1)
    with open(JSON_PATH, 'w') as jf:
        json.dump({
            'metadata': {'seeds': SEEDS, 'max_steps': MAX_STEPS, 'batch_size': BATCH_SIZE_DEFAULT,
                         'model_width': MODEL_WIDTH_DEFAULT, 'lr_used': lr_used, 'target_top1': target_top1,
                         'sessions': ckpt.get('sessions', []), 'failed': ckpt.get('failed', {})},
            'raw': raw
        }, jf, indent=2)
    try:
        stats, winners, delta_pct, corrected_p = build_final_table(raw, OPT_NAMES, SEEDS)
        print_final_table(stats, winners, delta_pct, corrected_p, OPT_NAMES)
    except Exception as e:
        log(f"[WARN] final table printing failed ({e!r}); results are safe in {JSON_PATH} / {CSV_PATH}")


def main():
    log_sep("=")
    log(f"  RUN 1 PART D (AdaBelief, E1: {E1_DOMAIN})  |  device={device}  torch={torch.__version__}")
    log_sep("=")
    _probe = build_optimizer("AdaBelief", torch.nn.Linear(2, 2), 1e-3)
    log(f"[E1] factory check: build_optimizer('AdaBelief') -> {type(_probe).__name__}")
    assert type(_probe).__name__ == "AdaBeliefOptimizer"
    ckpt = load_checkpoint()
    begin_session(ckpt, len(OPT_NAMES) * len(SEEDS))
    _install_signal_handlers()
    _start_heartbeat()
    _start_watchdog()
    code = 0
    try:
        _vision_body(ckpt)
        code = finish_session(ckpt, "complete")
    except _SessionEnd as e:
        code = finish_session(ckpt, e.reason, e.code)
    except KeyboardInterrupt:
        code = finish_session(ckpt, "interrupted (KeyboardInterrupt)", 130)
    except BaseException as e:
        import traceback
        log("[CRASH] " + repr(e)); log(traceback.format_exc())
        finish_session(ckpt, f"crash: {e!r}", 1)
        raise
    finally:
        try: _log_fh.close()
        except Exception: pass
    sys.exit(code)


if __name__ == '__main__':
    main()
