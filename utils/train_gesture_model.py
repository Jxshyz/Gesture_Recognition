"""
Training pipeline for the gesture classification model.

This module:
    1. Loads recorded gesture sequences from ./data (npy / npz).
    2. Filters labels via ALLOWED_LABELS / EXCLUDE_LABELS.
    3. Converts variable-length sequences into fixed-size windows.
    4. Extracts 189-dimensional features per window.
    5. Trains a GradientBoostingClassifier.
    6. Saves:
        - ./models/gesture_model.joblib
        - ./models/label_encoder.joblib

Data expectations:
    - Sequences stored as:
        (T,63)  flattened normalized landmarks
        (T,21,3) raw landmark format
        (63,)    single-frame fallback
    - Folder structure typically:
        ./data/recordings/<user>/<hand>/<gesture>/*.npz

Feature pipeline:
    window (T=12,63)
        -> window_features()
        -> 189-dim vector:
            [mean(63), std(63), delta(63)]

Model:
    - sklearn GradientBoostingClassifier
    - Incremental warm_start training for tqdm progress display
    - N_ESTIMATORS controls training duration/complexity

Intended usage:
    Run train_and_save() after collecting new recordings.
"""
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

# Optional: tqdm Fortschrittsbalken
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

DATA_ROOT = Path("./data")
MODELS = Path("./models")
MODELS.mkdir(exist_ok=True)

WINDOW_SIZE = 12
N_ESTIMATORS = 350  # <- hier steuerst du die Trainingsdauer

# ✅ trainiere NUR diese Labels
ALLOWED_LABELS = {
    "swipe_left",
    "swipe_right",
    "swipe_down",  # optional, falls du es nutzt
    "rotate_left",  # optional, falls du es nutzt
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

    search_root = (
        DATA_ROOT / "recordings" if (DATA_ROOT / "recordings").exists() else DATA_ROOT
    )
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


def build_windows(
    seqs: List[np.ndarray], labels: List[str], window_size: int = WINDOW_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
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


def _fit_gb_with_tqdm(
    X: np.ndarray,
    y_enc: np.ndarray,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = 42,
) -> GradientBoostingClassifier:
    """
    Train GradientBoostingClassifier incrementally (warm_start)
    so that tqdm can display an ETA/remaining time.
    """
    if tqdm is None:
        print("[WARN] tqdm ist nicht installiert. Installiere mit: pip install tqdm")
        print("[INFO] Trainiere ohne Progressbar...")
        model = GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
        model.fit(X, y_enc)
        return model

    model = GradientBoostingClassifier(
        n_estimators=1,  # Startwert, wird im Loop hochgezählt
        warm_start=True,
        random_state=random_state,
    )

    pbar = tqdm(total=n_estimators, desc="Training (GradientBoosting)", unit="tree")

    # Jede Iteration fügt genau einen weiteren Baum hinzu (warm_start)
    for i in range(1, n_estimators + 1):
        model.set_params(n_estimators=i)
        model.fit(X, y_enc)
        pbar.update(1)

    pbar.close()
    return model


def train_and_save():
    """
    Executes full training pipeline and stores model + label encoder.
    Aborts safely if no valid samples are found.
    """
    print("[INFO] Lade Sequenzen...")
    seqs, labels = load_sequences()
    print(f"[INFO] Geladen: {len(seqs)} Sequenzen")

    if len(seqs) == 0:
        print(
            "[ERROR] Keine Samples gefunden. Prüfe ./data/recordings/... und Labels in ALLOWED_LABELS."
        )
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
    model = _fit_gb_with_tqdm(X, y_enc, n_estimators=N_ESTIMATORS, random_state=42)

    joblib.dump(model, MODELS / "gesture_model.joblib")
    joblib.dump(le, MODELS / "label_encoder.joblib")
    print("[OK] Modell gespeichert in ./models/")
