# utils/train_gesture_model.py

import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from .feature_extractor import window_features, LandmarkBuffer

DATA_ROOT = Path("./data_raw")
MODELS = Path("./models")
MODELS.mkdir(exist_ok=True)

WINDOW_SIZE = 12


def load_dataset():
    X_list = []
    Y_list = []

    gesture_dirs = list(DATA_ROOT.glob("*"))

    for geste_path in gesture_dirs:
        if not geste_path.is_dir():
            continue

        gesture_name = geste_path.name

        for person_dir in geste_path.glob("*"):
            for npy_file in person_dir.glob("*.npy"):
                arr = np.load(npy_file)
                X_list.append(arr)
                Y_list.append(gesture_name)

    X_arr = np.stack(X_list)  # (N,63)
    Y_arr = np.array(Y_list)
    return X_arr, Y_arr


def windowize_data(X_arr, Y_arr):
    """
    Erzeugt Sliding-Window (12 Frames) → 189-Features.
    """
    sequences = []
    labels = []

    buf = LandmarkBuffer(WINDOW_SIZE)

    # Wir gehen sampleweise durch, ordnen sie anhand label
    # (da die Videos nicht fortlaufend sind, nehmen wir je Geste alle 12er-Chunks)
    for i in range(len(X_arr) - WINDOW_SIZE):
        # nur gleiche Klasse im Fenster verwenden
        if Y_arr[i] == Y_arr[i + WINDOW_SIZE - 1]:
            chunk = X_arr[i : i + WINDOW_SIZE]
            feat = window_features(chunk)
            sequences.append(feat)
            labels.append(Y_arr[i])

    return np.array(sequences), np.array(labels)


def train_and_save():
    print("[INFO] Lade Rohdaten...")
    X_arr, Y_arr = load_dataset()

    print("[INFO] Baue 12-Frame-Fenster...")
    X_win, Y_win = windowize_data(X_arr, Y_arr)

    print(f"[INFO] Trainingssamples: {len(X_win)}")

    print("[INFO] Label-Encoding...")
    le = LabelEncoder()
    Y_enc = le.fit_transform(Y_win)

    print(f"[INFO] Trainiere GradientBoostingClassifier...")
    model = GradientBoostingClassifier(n_estimators=350)
    model.fit(X_win, Y_enc)

    joblib.dump(model, MODELS / "gesture_model.joblib")
    joblib.dump(le, MODELS / "label_encoder.joblib")
    print("[INFO] Modell gespeichert!")
