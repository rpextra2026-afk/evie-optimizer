"""Wall-clock budget, graceful stop and 10-minute heartbeat shared by every long CLI.

Kaggle sessions are killed hard, so every long job (a) stops *between* runs when the next
run would not finish inside the budget, (b) stops between runs on SIGINT/SIGTERM, and
(c) prints a heartbeat line every 10 minutes so a silent hang is visible.
"""
from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Callable, Optional

log = logging.getLogger("evie_exp")

DEFAULT_BUDGET_MIN = 595.0  # 9 h 55 min, the owner's cutoff
SAFETY_S = 120.0            # margin for the final checkpoint + zip


class Deadline:
    """Absolute deadline. ``budget_min`` counts from construction unless ``deadline_epoch`` is given."""

    def __init__(self, budget_min: float = DEFAULT_BUDGET_MIN, deadline_epoch: Optional[float] = None,
                 elapsed_offset_min: float = 0.0, safety_s: float = SAFETY_S) -> None:
        self.safety_s = float(safety_s)
        if deadline_epoch is not None:
            self.deadline = float(deadline_epoch)
        else:
            self.deadline = time.time() + budget_min * 60.0 - elapsed_offset_min * 60.0
        self.start = time.time()

    def remaining_s(self) -> float:
        return self.deadline - time.time() - self.safety_s

    def can_start(self, est_run_s: float) -> bool:
        """True if a run expected to take ``est_run_s`` seconds still fits."""
        return self.remaining_s() >= est_run_s

    def elapsed_s(self) -> float:
        return time.time() - self.start


class GracefulStop:
    """First SIGINT/SIGTERM sets ``requested``; the second one raises KeyboardInterrupt."""

    def __init__(self) -> None:
        self.requested = False
        self._n = 0

    def install(self) -> "GracefulStop":
        if threading.current_thread() is not threading.main_thread():
            return self

        def _h(signum, _frame):
            self._n += 1
            self.requested = True
            log.warning("signal %s received: finishing the current run, then saving and stopping "
                        "(send again to abort immediately)", signum)
            if self._n >= 2:
                raise KeyboardInterrupt

        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, _h)
            except (ValueError, OSError):  # pragma: no cover - non-main thread / unsupported
                pass
        return self


class Heartbeat:
    """Prints ``status()`` every ``every_s`` seconds (default 10 min) from a daemon thread."""

    def __init__(self, status: Callable[[], str], every_s: float = 600.0, name: str = "heartbeat") -> None:
        self._status, self._every, self._name = status, every_s, name
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True, name=name)

    def _loop(self) -> None:
        t0 = time.time()
        while not self._stop.wait(self._every):
            try:
                msg = self._status()
            except Exception as e:  # never let the heartbeat kill a run
                msg = f"(status failed: {e!r})"
            print(f"[HEARTBEAT {(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)

    def __enter__(self) -> "Heartbeat":
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
