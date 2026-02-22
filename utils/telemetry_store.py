# utils/telemetry_store.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque
from threading import Lock
from time import time
from typing import Deque, Dict, Any, Optional


@dataclass
class TelemetryEvent:
    t: float
    state: str
    label: str
    conf: float
    seconds_left: float
    armed_progress: float
    armed_ready: bool


class TelemetryStore:
    """
    Thread-safe in-memory telemetry:
    - Current: Last state + live prediction + arming progress
    - History: Last N committed events
    """

    def __init__(self, max_history: int = 25):
        self._lock = Lock()
        self._history: Deque[TelemetryEvent] = deque(maxlen=max_history)

        self._current_state: str = "idle"
        self._current_label: str = "-"
        self._current_conf: float = 0.0
        self._seconds_left: float = 0.0
        self._last_update_t: float = 0.0

        self._armed_progress: float = 0.0  # 0..1
        self._armed_ready: bool = False

    def update(
        self,
        state: str,
        label: str,
        conf: float,
        seconds_left: float,
        armed_progress: float = 0.0,
        armed_ready: bool = False,
        ts: Optional[float] = None,
        push_history: bool = False,
    ) -> None:
        ts = time() if ts is None else float(ts)
        conf = float(conf)
        seconds_left = float(seconds_left)
        armed_progress = float(armed_progress)
        armed_ready = bool(armed_ready)

        # clamp
        if armed_progress < 0.0:
            armed_progress = 0.0
        if armed_progress > 1.0:
            armed_progress = 1.0

        with self._lock:
            self._current_state = str(state)
            self._current_label = str(label)
            self._current_conf = conf
            self._seconds_left = seconds_left
            self._armed_progress = armed_progress
            self._armed_ready = armed_ready
            self._last_update_t = ts

            if push_history:
                self._history.append(
                    TelemetryEvent(
                        t=ts,
                        state=str(state),
                        label=str(label),
                        conf=conf,
                        seconds_left=seconds_left,
                        armed_progress=armed_progress,
                        armed_ready=armed_ready,
                    )
                )

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "current": {
                    "state": self._current_state,
                    "label": self._current_label,
                    "conf": self._current_conf,
                    "seconds_left": self._seconds_left,
                    "armed_progress": self._armed_progress,
                    "armed_ready": self._armed_ready,
                    "last_update_t": self._last_update_t,
                },
                "history": [
                    asdict(e) for e in list(self._history)[::-1]
                ],  # newest first
            }


TELEMETRY = TelemetryStore(max_history=30)
