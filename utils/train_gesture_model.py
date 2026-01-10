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

DATA_ROOT = Path("./data")
MODELS = Path("./models")
MODELS.mkdir(exist_ok=True)

WINDOW_SIZE = 12

# ✅ trainiere NUR diese Labels
ALLOWED_LABELS = {
    "swipe_left",
    "swipe_right",
    "swipe_down",      # optional, falls du es nutzt
    "rotate_left",     # optional, falls du es nutzt
    "close_fist",
    "neutral_palm",
    "finger_pistol",
    "pinch",
}

# alles andere wird ignoriert (auch garbage, swipe_up, rotate_right, neutral_peace, ...)
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


def _load_npz_any(npz_path: Path) -> np.ndarray | None:
    preferred_keys = ("seq12", "seq", "data", "arr_0")
    npz = np.load(npz_path, allow_pickle=True)

    arr = None
    for k in preferred_keys:
        if k in npz.files:
            arr = npz[k]
            break
    if arr is None and len(npz.files) > 0:
        arr = npz[npz.files[0]]
    return arr


def load_sequences() -> Tuple[List[np.ndarray], List[str]]:
    seqs: List[np.ndarray] = []
    labels: List[str] = []

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"{DATA_ROOT} nicht gefunden.")

    search_root = DATA_ROOT / "recordings" if (DATA_ROOT / "recordings").exists() else DATA_ROOT
    files = list(search_root.rglob("*.npy")) + list(search_root.rglob("*.npz"))
    if not files:
        return seqs, labels

    for f in sorted(files):
        try:
            gesture_label = f.parent.name

            if gesture_label in EXCLUDE_LABELS:
                continue
            if gesture_label not in ALLOWED_LABELS:
                continue

            if f.suffix.lower() == ".npy":
                arr = np.load(f, allow_pickle=True)
            else:
                arr = _load_npz_any(f)
                if arr is None:
                    continue

            arr = np.asarray(arr, dtype=np.float32)

            # normalize shape: (63,) or (T,63) or (T,21,3)
            if arr.ndim == 1:
                if arr.shape[0] != 63:
                    continue
                arr = arr.reshape(1, 63)

            elif arr.ndim == 2:
                if arr.shape[1] != 63:
                    continue

            elif arr.ndim == 3:
                if arr.shape[1:] == (21, 3):
                    arr = arr.reshape(arr.shape[0], 63)
                else:
                    continue
            else:
                continue

            arr = np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)

            seqs.append(arr)
            labels.append(gesture_label)

        except Exception:
            continue

    return seqs, labels


def build_windows(seqs: List[np.ndarray], labels: List[str], window_size: int = WINDOW_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    for seq, lab in zip(seqs, labels):
        T = seq.shape[0]
        if T >= window_size:
            for start in range(0, T - window_size + 1):
                win = seq[start : start + window_size]
                feat = window_features(win)
                X_list.append(feat)
                y_list.append(lab)
        else:
            win = _resample_to_T(seq, window_size)
            feat = window_features(win)
            X_list.append(feat)
            y_list.append(lab)

    if not X_list:
        return np.zeros((0, 189), dtype=np.float32), np.asarray([], dtype=str)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=str)
    return X, y


def train_and_save():
    print("[INFO] Lade Sequenzen...")
    seqs, labels = load_sequences()
    print(f"[INFO] Geladen: {len(seqs)} Sequenzen")

    if len(seqs) == 0:
        print("[ERROR] Keine Samples gefunden. Prüfe ./data/recordings/... und Labels in ALLOWED_LABELS.")
        return

    print("[INFO] Baue Windows (189 Features)...")
    X, y = build_windows(seqs, labels, WINDOW_SIZE)
    print(f"[INFO] Windows: X={X.shape}, y={y.shape}")

    if X.shape[0] == 0:
        print("[ERROR] Keine Windows erzeugt.")
        return

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
