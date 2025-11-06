from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from joblib import dump

from utils.feature_extractor import normalize_landmarks, window_features

# Fenster-Konfiguration (Echtzeit-geeignet, kleine Datenmengen)
WINDOW = 16     # ~0.5 s bei ~30 FPS
STRIDE = 4

DATA_DIR = Path("./data")
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

GESTURE_COL = "label_text"
COLOR_COL = "square_color"

def _subject_from_filename(name: str) -> str:
    # Erwartete Patterns:
    #  - Gestures_<Name>.pkl
    #  - Gestures_<Name>_<N>.pkl
    m = re.match(r"^Gestures_([^_]+)(?:_\d+)?\.pkl$", name)
    if m:
        return m.group(1)
    # Fallback: alles zwischen "Gestures_" und ".pkl"
    m2 = re.match(r"^Gestures_(.+)\.pkl$", name)
    return m2.group(1) if m2 else name

def _lm_row_to_vec(row: pd.Series) -> np.ndarray:
    lms = [row[f"lm_{i}"] for i in range(21)]
    return normalize_landmarks(lms)  # (63,)

def _windows_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # Nur grüne Frames verwenden (saubere Labels)
    df = df[df[COLOR_COL] == "green"].sort_values("timestamp").copy()
    if df.empty:
        return np.zeros((0, 189)), np.zeros((0,))
    X_flat = np.stack([_lm_row_to_vec(r) for _, r in df.iterrows()], axis=0)  # (N,63)
    y = df[GESTURE_COL].to_numpy()

    X_seq, y_seq = [], []
    i, n = 0, len(df)
    while i + WINDOW <= n:
        win = X_flat[i:i+WINDOW]
        lab = y[i:i+WINDOW]
        # Mehrheitslabel im Fenster
        vals, cnt = np.unique(lab, return_counts=True)
        maj = vals[np.argmax(cnt)]
        feats = window_features(win)  # (189,)
        X_seq.append(feats)
        y_seq.append(maj)
        i += STRIDE

    if not X_seq:
        return np.zeros((0, 189)), np.zeros((0,))
    return np.stack(X_seq), np.array(y_seq)

def _load_all_grouped():
    files = sorted(DATA_DIR.glob("Gestures_*.pkl"))
    if not files:
        raise RuntimeError("Keine Trainingsdaten gefunden in ./data (Gestures_*.pkl).")

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    groups: List[str] = []

    for fp in files:
        df = pd.read_pickle(fp)
        # Mindest-Schema prüfen
        if not set([f"lm_{i}" for i in range(21)]).issubset(df.columns):
            continue
        Xf, yf = _windows_from_df(df)
        if len(Xf):
            X_list.append(Xf)
            y_list.append(yf)
            groups.append(_subject_from_filename(fp.name))

    if not X_list:
        raise RuntimeError("Es wurden Dateien gefunden, aber keine verwertbaren Fenster (nur rote Frames?).")

    # Stapeln + Gruppen-Index je Fenster erzeugen
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    group_idx = np.concatenate([np.full(len(x), i) for i, x in enumerate(X_list)], axis=0)
    return X, y, np.array(groups)[group_idx]

def train_and_save():
    # Daten laden und gruppierte Splits vorbereiten
    X, y, groups = _load_all_grouped()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Gruppierter Split (alle Fenster einer Aufnahme/Person zusammenhalten)
    unique_groups = np.unique(groups)
    if len(unique_groups) > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))
    else:
        # Fallback: wenn nur eine Gruppe vorhanden ist (z. B. erste Sammlung),
        # mache einen normalen Split, damit überhaupt ein Test-Report entsteht.
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.25, random_state=42, stratify=y_enc
        )

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

    # Schlankes, schnelles Modell für kleine Datensätze und Echtzeit
    pipe = Pipeline([
        ("clf", GradientBoostingClassifier(random_state=42))
    ])
    pipe.fit(X_tr, y_tr)

    # Echte Generalisierungsmetrik
    y_hat = pipe.predict(X_te)
    print("\n=== Test Report (group-wise split) ===")
    print(classification_report(y_te, y_hat, target_names=le.classes_))
    print("Confusion matrix:\n", confusion_matrix(y_te, y_hat))

    # OPTIONAL: Für das Deployment auf allen Daten refitten (mehr Signal)
    pipe.fit(X, y_enc)

    # Speichern
    dump(pipe, MODEL_DIR / "gesture_model.joblib")
    dump(le,   MODEL_DIR / "label_encoder.joblib")
    from joblib import dump as _dump
    _dump({"WINDOW": WINDOW, "STRIDE": STRIDE}, MODEL_DIR / "config.joblib")

    print(f"[OK] Modell gespeichert in: {MODEL_DIR.resolve()}")
    print("[Hinweis] Bei neuen Dateien in ./data Skript erneut ausführen, der Split verwendet immer ALLE aktuell verfügbaren Daten.")

if __name__ == "__main__":
    train_and_save()
