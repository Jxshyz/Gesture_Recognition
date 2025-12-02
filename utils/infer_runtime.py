# utils/infer_runtime.py

import time
from typing import Callable, Optional, List, Tuple
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import joblib

from .feature_extractor import normalize_landmarks, LandmarkBuffer, window_features

# FSM States
IDLE = "IDLE"
READY = "READY"
RECORDING = "RECORDING"
COMMIT = "COMMIT"
COOLDOWN = "COOLDOWN"

WINDOW_SIZE = 12  # Frames


OnPrediction = Callable[[str, float, np.ndarray, str, float], None]
OnRender = Callable[[np.ndarray, str, float], None]


# -----------------------------------------------------------------------------
# Modell laden
# -----------------------------------------------------------------------------
def load_model():
    models_dir = Path("./models")
    model = joblib.load(models_dir / "gesture_model.joblib")
    le = joblib.load(models_dir / "label_encoder.joblib")
    return model, le


def predict(model, le, feat189: np.ndarray):
    X = feat189.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        idx = np.argmax(proba)
        return le.inverse_transform([idx])[0], float(proba[idx])
    else:
        lbl = model.predict(X)[0]
        return le.inverse_transform([lbl])[0], 1.0


# -----------------------------------------------------------------------------
# Bewegung messen
# -----------------------------------------------------------------------------
def motion_energy(lm_prev, lm_curr):
    if lm_prev is None or lm_curr is None:
        return 0.0
    a = np.array(lm_prev)
    b = np.array(lm_curr)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


# -----------------------------------------------------------------------------
# FSM-basierte Live-Erkennung
# -----------------------------------------------------------------------------
def run_live(
    camera_index=0,
    show_window=True,
    draw_phase_overlay=True,
    on_prediction: Optional[OnPrediction] = None,
    on_render: Optional[OnRender] = None,
):
    # Lade Modell
    model, label_encoder = load_model()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Kamera konnte nicht geöffnet werden.")
        return

    # FSM Variablen
    state = IDLE
    neutral_timer = 0.0
    cooldown_timer = 0.0
    record_timer = 0.0

    last_time = time.time()
    last_lm = None

    RECORD_TIMEOUT = 1.4
    COOLDOWN = 0.5
    NEUTRAL_HOLD = 1.5
    MOTION_HIGH = 0.035
    MOTION_LOW = 0.015

    lm_buffer = LandmarkBuffer(WINDOW_SIZE)
    record_labels = []
    record_confs = []

    while True:
        now = time.time()
        dt = now - last_time
        last_time = now

        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_lm = None

        if result.multi_hand_landmarks:
            lm = [(p.x, p.y, p.z) for p in result.multi_hand_landmarks[0].landmark]
            drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            current_lm = lm

            # Normiert (63)
            lm_norm = normalize_landmarks(lm)
        else:
            lm_norm = None

        # Bewegung
        m_energy = motion_energy(last_lm, current_lm)
        last_lm = current_lm

        # -----------------------------------------------------------
        # FSM LOGIK
        # -----------------------------------------------------------
        if state == IDLE:
            if current_lm is not None:
                # Warte auf stabile Neutralgeste
                feat = lm_norm.reshape(1, -1)
                gesture, conf = predict(model, label_encoder, np.zeros(189))
                # Trick: 189-Dummy (Neutral wird sowieso gut erkannt)

                if gesture in ["neutral_palm", "neutral_peace"]:
                    neutral_timer += dt
                    if neutral_timer >= NEUTRAL_HOLD:
                        state = READY
                        neutral_timer = 0
                else:
                    neutral_timer = 0

        elif state == READY:
            # Start wenn Bewegung frisch beginnt
            if m_energy > MOTION_HIGH:
                state = RECORDING
                lm_buffer = LandmarkBuffer(WINDOW_SIZE)
                record_labels = []
                record_confs = []
                record_timer = 0.0

        elif state == RECORDING:
            record_timer += dt

            if lm_norm is not None:
                lm_buffer.push(lm_norm)

                if lm_buffer.full():
                    feats189 = window_features(lm_buffer.as_array())
                    lbl, conf = predict(model, label_encoder, feats189)
                    record_labels.append(lbl)
                    record_confs.append(conf)

            # Abbruchbedingungen
            if m_energy < MOTION_LOW:
                # Wenn die Bewegung für ca. 200ms niedrig ist
                if record_timer > 0.25:
                    state = COMMIT

            if record_timer > RECORD_TIMEOUT:
                state = COMMIT

        elif state == COMMIT:
            if record_labels:
                values, counts = np.unique(record_labels, return_counts=True)
                winner = values[np.argmax(counts)]
                avg_conf = float(np.mean([c for l, c in zip(record_labels, record_confs) if l == winner]))
            else:
                winner = "NO_GESTURE"
                avg_conf = 0.0

            # Ergebnis zurückgeben
            if on_prediction:
                on_prediction(winner, avg_conf, frame, state, 0.0)

            # Cooldown starten
            state = COOLDOWN
            cooldown_timer = 0.0

        elif state == COOLDOWN:
            cooldown_timer += dt
            if cooldown_timer > COOLDOWN:
                state = IDLE

        # -----------------------------------------------------------
        # Phase Overlay?
        # -----------------------------------------------------------
        if draw_phase_overlay:
            color = {
                IDLE: (200, 200, 200),
                READY: (0, 255, 0),
                RECORDING: (0, 0, 255),
                COMMIT: (0, 255, 255),
                COOLDOWN: (255, 0, 0),
            }[state]

            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 4)

        # -----------------------------------------------------------
        # Render Callback
        # -----------------------------------------------------------
        if on_render:
            on_render(frame, state, 0.0)

        # -----------------------------------------------------------
        # Optionales Fenster
        # -----------------------------------------------------------
        if show_window:
            cv2.imshow("Gesture-FSM", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
