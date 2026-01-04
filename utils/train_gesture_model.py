# utils/train_gesture_model.py
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

from utils.feature_extractor import window_features

DATA_ROOT = Path("./data/recordings")
MODELS = Path("./models")
MODELS.mkdir(exist_ok=True)

WINDOW_SIZE = 12

# komplett raus aus Training:
EXCLUDE_LABELS = {"garbage", "swipe_up", "rotate_right", "neutral_peace"}


def _resample_to_T(seq: np.ndarray, T: int) -> np.ndarray:
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
    data/recordings/<name>/<hand>/<gesture>/*.npz
    """
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"{DATA_ROOT} nicht gefunden.")

    seqs: List[np.ndarray] = []
    labels: List[str] = []

    for name_dir in sorted(DATA_ROOT.glob("*")):
        if not name_dir.is_dir():
            continue
        for hand_dir in sorted(name_dir.glob("*")):
            if not hand_dir.is_dir():
                continue
            for gesture_dir in sorted(hand_dir.glob("*")):
                if not gesture_dir.is_dir():
                    continue

                gesture = gesture_dir.name
                if gesture in EXCLUDE_LABELS:
                    continue

                for npz_file in gesture_dir.glob("*.npz"):
                    try:
                        data = np.load(npz_file, allow_pickle=True)
                    except Exception:
                        continue

                    if "seq12" in data:
                        arr = np.asarray(data["seq12"], dtype=np.float32)
                    elif "seq" in data:
                        arr = np.asarray(data["seq"], dtype=np.float32)
                    else:
                        continue

                    if arr.ndim != 2 or arr.shape[1] != 63:
                        continue

                    arr = np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)
                    seqs.append(arr)
                    labels.append(gesture)

    return seqs, labels


def build_windows(seqs: List[np.ndarray], labels: List[str], window_size: int = WINDOW_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    for seq, lab in zip(seqs, labels):
        T = seq.shape[0]
        if T >= window_size:
            for start in range(0, T - window_size + 1, 1):
                win = seq[start:start + window_size]
                feat = window_features(win)
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


def train_and_save():
    print("[INFO] Lade Sequenzen...")
    seqs, labels = load_sequences()
    print(f"[INFO] Geladen: {len(seqs)} Sequenzen")

    if len(seqs) == 0:
        raise RuntimeError("Keine Sequenzen gefunden. Check data/recordings/...")

    print("[INFO] Baue Windows (189 Features)...")
    X, y = build_windows(seqs, labels, WINDOW_SIZE)
    print(f"[INFO] Windows: X={X.shape}, y={y.shape}")

    counts = Counter(y)
    print("[INFO] Class counts:")
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:>14}: {v}")

    print("[INFO] Label-Encoding...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("[INFO] Trainiere GradientBoostingClassifier...")
    model = GradientBoostingClassifier(n_estimators=350, random_state=42)
    model.fit(X, y_enc)

    joblib.dump(model, MODELS / "gesture_model.joblib")
    joblib.dump(le, MODELS / "label_encoder.joblib")
    print("[OK] Modell gespeichert in ./models/")
