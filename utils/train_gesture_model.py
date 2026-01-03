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

# ✅ Bei dir liegen die Daten unter data/recordings/...
DATA_ROOT = Path("./data")
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


def _load_npz_any(npz_path: Path) -> np.ndarray | None:
    """
    Robust .npz loader:
    - bevorzugt keys: "seq", "data", "arr_0"
    - fallback: erstes array im npz
    """
    preferred_keys = ("seq", "data", "arr_0")
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
    """
    Lädt alle *.npy und *.npz rekursiv unter:
      - ./data/recordings/**/<gesture>/*.npy|*.npz   (dein aktuelles Layout)
      - ./data/<gesture>/<person>/*.npy|*.npz       (alte Layouts werden mit unterstützt)

    Label = Name des Parent-Folders der Sample-Datei (also der Gesten-Ordnername).
    """
    seqs: List[np.ndarray] = []
    labels: List[str] = []

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"{DATA_ROOT} nicht gefunden. Ordner existiert nicht.")

    # ✅ Wenn recordings existiert, nutze den; sonst fallback auf data/
    search_root = DATA_ROOT / "recordings" if (DATA_ROOT / "recordings").exists() else DATA_ROOT

    files = list(search_root.rglob("*.npy")) + list(search_root.rglob("*.npz"))
    if not files:
        return seqs, labels

    for f in sorted(files):
        try:
            if f.suffix.lower() == ".npy":
                arr = np.load(f, allow_pickle=True)
            else:
                arr = _load_npz_any(f)
                if arr is None:
                    print(f"[WARN] Skip {f} (empty npz)")
                    continue

            arr = np.asarray(arr, dtype=np.float32)

            # --- normalize shape ---
            # erlaubt:
            #  (63,)             -> (1,63)
            #  (T,63)            -> ok
            #  (T,21,3)          -> (T,63)
            if arr.ndim == 1:
                if arr.shape[0] != 63:
                    print(f"[WARN] Skip {f} shape={arr.shape} (expected 63)")
                    continue
                arr = arr.reshape(1, 63)

            elif arr.ndim == 2:
                if arr.shape[1] != 63:
                    print(f"[WARN] Skip {f} shape={arr.shape} (expected (T,63))")
                    continue

            elif arr.ndim == 3:
                # (T,21,3) -> (T,63)
                if arr.shape[1:] == (21, 3):
                    arr = arr.reshape(arr.shape[0], 63)
                else:
                    print(f"[WARN] Skip {f} shape={arr.shape} (expected (T,21,3))")
                    continue
            else:
                print(f"[WARN] Skip {f} shape={arr.shape}")
                continue

            arr = np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)

            gesture_label = f.parent.name  # ✅ z.B. "pinch" / "finger_pistol" / "neutral_palm"
            seqs.append(arr)
            labels.append(gesture_label)

        except Exception as e:
            print(f"[WARN] Skip {f} ({e})")

    return seqs, labels


def build_windows(seqs: List[np.ndarray], labels: List[str], window_size: int = WINDOW_SIZE) -> Tuple[np.ndarray, np.ndarray]:
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
            for start in range(0, T - window_size + 1):
                win = seq[start : start + window_size]  # (12,63)
                feat = window_features(win)             # (189,)
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


def _cap_garbage(X: np.ndarray, y: np.ndarray, garbage_label: str = "garbage", max_ratio: float = 1.3) -> Tuple[np.ndarray, np.ndarray]:
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

    if len(seqs) == 0:
        print("[ERROR] Keine Samples gefunden. Prüfe, ob unter ./data/recordings/... Dateien *.npy oder *.npz liegen.")
        print("        Beispiel erwartet: data/recordings/josh/Right/pinch/<sample>.npz")
        return

    print("[INFO] Baue Windows (189 Features)...")
    X, y = build_windows(seqs, labels, WINDOW_SIZE)
    print(f"[INFO] Windows: X={X.shape}, y={y.shape}")

    if X.shape[0] == 0:
        print("[ERROR] Keine Windows erzeugt (evtl. Sample-Shape falsch).")
        return

    X, y = _cap_garbage(X, y, garbage_label="garbage", max_ratio=1.3)

    counts = Counter(y)
    print("[INFO] Class counts (after cap):")
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
