# utils/train_gesture_model.py
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

from utils.feature_extractor import window_features


DATA_ROOT = Path("./data_raw")
MODELS = Path("./models")
MODELS.mkdir(exist_ok=True)

WINDOW_SIZE = 12


def _resample_to_T(seq: np.ndarray, T: int) -> np.ndarray:
    """seq: (N,63) -> (T,63) via linear interpolation"""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq.reshape(1, -1)
    N, D = seq.shape
    if N == T:
        return seq
    xs = np.linspace(0, 1, N)
    xt = np.linspace(0, 1, T)
    out = np.zeros((T, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(xt, xs, seq[:, d])
    return out


def load_sequences() -> Tuple[List[np.ndarray], List[str]]:
    """
    Lädt alle *.npy aus data_raw/<gesture>/<person>/*.npy

    Erwartet pro Datei:
      - neu: (T,63)
      - alt: (63,) -> wird zu (1,63)
    """
    seqs: List[np.ndarray] = []
    labels: List[str] = []

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"{DATA_ROOT} nicht gefunden. Erst aufnehmen: python main.py record_data ...")

    for gesture_dir in sorted(DATA_ROOT.glob("*")):
        if not gesture_dir.is_dir():
            continue
        gesture = gesture_dir.name

        for person_dir in gesture_dir.glob("*"):
            if not person_dir.is_dir():
                continue

            for npy_file in person_dir.glob("*.npy"):
                arr = np.load(npy_file)
                arr = np.asarray(arr, dtype=np.float32)

                # normalize shape to (T,63)
                if arr.ndim == 1:
                    if arr.shape[0] != 63:
                        print(f"[WARN] Skip {npy_file} shape={arr.shape}")
                        continue
                    arr = arr.reshape(1, 63)
                elif arr.ndim == 2:
                    if arr.shape[1] != 63:
                        print(f"[WARN] Skip {npy_file} shape={arr.shape}")
                        continue
                else:
                    print(f"[WARN] Skip {npy_file} shape={arr.shape}")
                    continue

                # remove NaNs/Infs just in case
                arr = np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)

                seqs.append(arr)
                labels.append(gesture)

    return seqs, labels


def build_windows(
    seqs: List[np.ndarray], labels: List[str], window_size: int = WINDOW_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aus jeder Sequenz werden Trainingswindows generiert:
      - wenn len >= window_size: sliding windows (step=1)
      - wenn len < window_size: resample to window_size (1 window)

    Output:
      X: (N,189)
      y: (N,)
    """
    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    for seq, lab in zip(seqs, labels):
        T = seq.shape[0]
        if T >= window_size:
            step = 1
            for start in range(0, T - window_size + 1, step):
                win = seq[start : start + window_size]  # (12,63)
                feat = window_features(win)  # (189,)
                X_list.append(feat)
                y_list.append(lab)
        else:
            win = _resample_to_T(seq, window_size)
            feat = window_features(win)
            X_list.append(feat)
            y_list.append(lab)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list)
    return X, y


def _cap_garbage(
    X: np.ndarray, y: np.ndarray, garbage_label: str = "garbage", max_ratio: float = 1.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Begrenzt die Anzahl Garbage-Samples, damit es nicht dominiert.
    max_ratio: garbage <= max_ratio * mean(count(other classes))
    """
    counts = Counter(y)
    if garbage_label not in counts:
        return X, y

    other = [c for lab, c in counts.items() if lab != garbage_label]
    if not other:
        return X, y

    cap = int(max_ratio * (sum(other) / len(other)))
    g_count = counts[garbage_label]
    if g_count <= cap:
        return X, y

    # indices
    idx_g = np.where(y == garbage_label)[0]
    idx_o = np.where(y != garbage_label)[0]
    np.random.shuffle(idx_g)
    idx_g = idx_g[:cap]
    idx = np.concatenate([idx_o, idx_g])
    np.random.shuffle(idx)

    return X[idx], y[idx]


def train_and_save():
    print("[INFO] Lade Sequenzen...")
    seqs, labels = load_sequences()
    print(f"[INFO] Geladen: {len(seqs)} Sequenzen")

    print("[INFO] Baue Windows (189 Features)...")
    X, y = build_windows(seqs, labels, WINDOW_SIZE)
    print(f"[INFO] Windows: X={X.shape}, y={y.shape}")

    # Garbage deckeln
    X, y = _cap_garbage(X, y, garbage_label="garbage", max_ratio=1.3)

    # Stats
    counts = Counter(y)
    print("[INFO] Class counts (after cap):")
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:>12}: {v}")

    print("[INFO] Label-Encoding...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("[INFO] Trainiere GradientBoostingClassifier...")
    model = GradientBoostingClassifier(n_estimators=350, random_state=42)
    model.fit(X, y_enc)

    joblib.dump(model, MODELS / "gesture_model.joblib")
    joblib.dump(le, MODELS / "label_encoder.joblib")
    print("[OK] Modell gespeichert in ./models/")
