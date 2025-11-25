from __future__ import annotations
import time
from collections import defaultdict
from typing import Dict, Tuple, Optional

class PredictionAggregator:
    """
    Sammelt (label, conf)-Vorhersagen über eine Zeitspanne.
    - Drosselt per min_interval_s (z. B. 50–100 ms), um nicht jeden Frame zu zählen.
    - Ergebnis: Majority-Label, mittlere Confidence dieses Labels, Anzahl akzeptierter Samples.
    """
    def __init__(self, min_interval_s: float = 0.075):
        self.min_interval_s = float(min_interval_s)
        self.reset()

    def reset(self):
        self._last_t = 0.0
        self._counts: Dict[str, int] = defaultdict(int)
        self._conf_sum: Dict[str, float] = defaultdict(float)
        self._n_total = 0

    def feed(self, label: str, conf: float, now: float) -> bool:
        # Zeitliche Drosselung
        if now - self._last_t < self.min_interval_s:
            return False
        self._last_t = now
        self._counts[label] += 1
        self._conf_sum[label] += float(conf)
        self._n_total += 1
        return True

    def result(self) -> Tuple[Optional[str], float, int]:
        if self._n_total == 0 or not self._counts:
            return None, 0.0, 0
        # Majority-Label
        label = max(self._counts.items(), key=lambda kv: kv[1])[0]
        n_label = self._counts[label]
        conf_avg = self._conf_sum[label] / max(1, n_label)
        return label, conf_avg, self._n_total
