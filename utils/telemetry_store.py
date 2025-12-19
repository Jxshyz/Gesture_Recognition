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


class TelemetryStore:
    """
    Thread-safe In-Memory Telemetrie:
    - current: letzter Zustand + letzte Prediction
    - history: letzte N committed Events
    """

    def __init__(self, max_history: int = 25):
        self._lock = Lock()
        self._history: Deque[TelemetryEvent] = deque(maxlen=max_history)

        self._current_state: str = "idle"
        self._current_label: str = "-"
        self._current_conf: float = 0.0
        self._seconds_left: float = 0.0
        self._last_update_t: float = 0.0

    def update(
        self,
        state: str,
        label: str,
        conf: float,
        seconds_left: float,
        ts: Optional[float] = None,
        push_history: bool = True,
    ) -> None:
        ts = time() if ts is None else float(ts)
        conf = float(conf)
        seconds_left = float(seconds_left)

        with self._lock:
            self._current_state = str(state)
            self._current_label = str(label)
            self._current_conf = conf
            self._seconds_left = seconds_left
            self._last_update_t = ts

            if push_history:
                self._history.append(
                    TelemetryEvent(
                        t=ts,
                        state=str(state),
                        label=str(label),
                        conf=conf,
                        seconds_left=seconds_left,
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
                    "last_update_t": self._last_update_t,
                },
                "history": [asdict(e) for e in list(self._history)[::-1]],  # newest first
            }


TELEMETRY = TelemetryStore(max_history=30)
