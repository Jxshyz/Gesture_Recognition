"""
Feature extraction utilities for real-time gesture inference.

This module provides:

- Landmark normalization for MediaPipe hand landmarks
- Temporal feature aggregation over sliding windows
- A sliding window buffer for sequential landmark data

The output features are designed for classical ML models
(e.g., NN, HMM, or similarity-based classification).
"""

from __future__ import annotations
from collections import deque
from typing import Deque, List, Tuple
import numpy as np


def normalize_landmarks(lm: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    Normalize a single MediaPipe hand landmark frame.

    The function performs:
        1. NaN handling (replace with zeros)
        2. Translation normalization (wrist as origin)
        3. Scale normalization using landmark 9 as reference
        4. Flattening to a 63-dimensional feature vector

    Parameters:
        lm (List[Tuple[float, float, float]]):
            List of 21 landmark points (x, y, z).

    Returns:
        np.ndarray:
            Flattened normalized landmark vector of shape (63,).
    """
    arr = np.array(lm, dtype=float)  # (21,3)

    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)

    # Wrist as reference point (Translation)
    wrist = arr[0, :2].copy()
    arr[:, 0] -= wrist[0]
    arr[:, 1] -= wrist[1]

    # Standardization to approximate hand size (scaling)
    ref = arr[9, :2]
    scale = np.linalg.norm(ref) + 1e-6
    arr[:, 0] /= scale
    arr[:, 1] /= scale
    arr[:, 2] /= scale

    return arr.reshape(-1)  # (63,)


def window_features(window: np.ndarray) -> np.ndarray:
    """
    Extract temporal features from a sliding window of landmarks.

    Given a window of normalized landmarks with shape (T, 63),
    the following features are computed:

        - Mean over time  → 63 features
        - Standard deviation over time → 63 features
        - Temporal delta (last - first frame) → 63 features

    These are concatenated into a 189-dimensional feature vector.

    NaN and infinite values are safely replaced.

    Parameters:
        window (np.ndarray):
            Array of shape (T, 63) representing T time steps.

    Returns:
        np.ndarray:
            Feature vector of shape (189,).
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
    """
    Sliding window buffer for normalized landmark vectors.

    Stores a fixed-length sequence of flattened landmark frames
    and provides utilities to check completeness and retrieve
    the window as a NumPy array.
    """

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buf: Deque[np.ndarray] = deque(maxlen=maxlen)

    def push(self, lm_norm_flat: np.ndarray):
        self.buf.append(lm_norm_flat)

    def full(self) -> bool:
        return len(self.buf) == self.maxlen

    def as_array(self) -> np.ndarray:
        return np.stack(list(self.buf), axis=0)  # (T,63)
