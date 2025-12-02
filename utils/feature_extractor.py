# utils/feature_extractor.py

from __future__ import annotations
from collections import deque
from typing import Deque, List, Tuple
import numpy as np


# ------------------------------------------------------------
# Normalisieren der Mediapipe-Landmarks auf (63,) Flat-Vector
# ------------------------------------------------------------
def normalize_landmarks(lm: List[Tuple[float, float, float]]) -> np.ndarray:
    arr = np.array(lm, dtype=float)  # (21,3)

    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)

    # Wrist als Referenzpunkt (Translation)
    wrist = arr[0, :2].copy()
    arr[:, 0] -= wrist[0]
    arr[:, 1] -= wrist[1]

    # Normierung auf ungefähre Handgröße (Skalierung)
    ref = arr[9, :2]
    scale = np.linalg.norm(ref) + 1e-6
    arr[:, 0] /= scale
    arr[:, 1] /= scale
    arr[:, 2] /= scale

    return arr.reshape(-1)  # (63,)


# ------------------------------------------------------------
# Zeitfenster → 189 Feature-Vektor
# ------------------------------------------------------------
def window_features(window: np.ndarray) -> np.ndarray:
    """
    window: (T,63)
    Output: 189 Features (mean, std, delta)
    """
    mean = np.nanmean(window, axis=0)
    std = np.nanstd(window, axis=0)
    delta = window[-1] - window[0]

    feats = np.concatenate([mean, std, delta], axis=0)
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e3, neginf=-1e3)
    return feats  # (189,)


# ------------------------------------------------------------
# Sliding Window Buffer
# ------------------------------------------------------------
class LandmarkBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buf: Deque[np.ndarray] = deque(maxlen=maxlen)

    def push(self, lm_norm_flat: np.ndarray):
        self.buf.append(lm_norm_flat)

    def full(self) -> bool:
        return len(self.buf) == self.maxlen

    def as_array(self) -> np.ndarray:
        return np.stack(list(self.buf), axis=0)  # (T,63)
