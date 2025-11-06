# utils/feature_extractor.py
from __future__ import annotations
from collections import deque
from typing import Deque, List, Tuple
import numpy as np

# Erwartet 21 Tuples (x,y,z) in Normal-Koordinaten (0..1, z relativ). Kann NaN enthalten.
def normalize_landmarks(lm: List[Tuple[float, float, float]]) -> np.ndarray:
    arr = np.array(lm, dtype=float)  # (21,3)
    # Fehlwerte grob auffüllen: simple Vorwärts-/Rückwärts-Ersatz
    if np.isnan(arr).any():
        # Fülle NaN mit 0.0 als Fallback; robustere Strategien möglich
        arr = np.nan_to_num(arr, nan=0.0)

    # Translation: relativ zum Wrist (Index 0)
    wrist = arr[0, :2].copy()
    arr[:, 0] -= wrist[0]
    arr[:, 1] -= wrist[1]

    # Skalierung: Handgröße (Distanz Wrist zu MCap (Index 9) oder mittlerer Abstand)
    ref = arr[9, :2]
    scale = np.linalg.norm(ref) + 1e-6
    arr[:, 0] /= scale
    arr[:, 1] /= scale
    # z optional leicht skalieren (gleiche Skala)
    arr[:, 2] /= scale
    return arr.reshape(-1)  # (63,)

# Bildet aus einem Zeitfenster robuste, kleine Feature-Vektoren (geeignet für kleine Datensätze).
# Für jede der 63 Dimensionen: mean, std, last-first (Delta) => 189 Features.
def window_features(window: np.ndarray) -> np.ndarray:
    # window: (T, 63)
    mean = np.nanmean(window, axis=0)
    std = np.nanstd(window, axis=0)
    delta = window[-1] - window[0]
    feats = np.concatenate([mean, std, delta], axis=0)
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e3, neginf=-1e3)
    return feats  # (189,)

class LandmarkBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buf: Deque[np.ndarray] = deque(maxlen=maxlen)

    def push(self, lm_norm_flat: np.ndarray):
        self.buf.append(lm_norm_flat)

    def full(self) -> bool:
        return len(self.buf) == self.maxlen

    def as_array(self) -> np.ndarray:
        if not self.full():
            raise ValueError("Buffer not full")
        return np.stack(list(self.buf), axis=0)  # (T,63)
