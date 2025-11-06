# utils/infer_runtime.py
# Laufzeit: Kamera + MediaPipe + Modell; ruft bei jeder gültigen Vorhersage einen Callback auf,
# der das Frame z.B. mit einem Bild überlagern oder Aktionen triggern kann.
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, Callable, Tuple
import cv2
import numpy as np
from joblib import load

from utils.hand_tracking import HandTracker, put_hud
from utils.feature_extractor import LandmarkBuffer, normalize_landmarks, window_features

MODEL_DIR = Path("./models")

# Callback-Signatur:
# on_prediction(label: str, conf: float, frame_bgr: np.ndarray, phase_color: str, seconds_left: float) -> None
OnPredFn = Callable[[str, float, np.ndarray, str, float], None]

def _predict(pipe, le, window_arr: np.ndarray) -> Tuple[str, float]:
    feats = window_features(window_arr).reshape(1, -1)
    # Konfidenz schätzen per predict_proba (max probability)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(feats)[0]
        y_idx = int(np.argmax(proba))
        conf = float(np.max(proba))
    else:
        # Fallback, falls das Modell kein predict_proba hat
        y_idx = int(pipe.predict(feats)[0])
        conf = 1.0
    label = le.inverse_transform([y_idx])[0]
    return label, conf

def run_live(camera_index: int = 0,
             width: Optional[int] = 1280,
             height: Optional[int] = 720,
             on_prediction: Optional[OnPredFn] = None,
             green_dur: float = 1.0,
             red_dur: float = 1.0,
             conf_threshold: float = 0.0) -> None:
    # Modelle laden
    pipe = load(MODEL_DIR / "gesture_model.joblib")
    le   = load(MODEL_DIR / "label_encoder.joblib")
    cfg  = load(MODEL_DIR / "config.joblib")
    WINDOW = int(cfg["WINDOW"])

    # Kamera
    cap = cv2.VideoCapture(camera_index)
    if width is not None: cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print(f"[ERROR] Konnte Kamera {camera_index} nicht öffnen.")
        return

    tracker = HandTracker(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        draw_style=True,
    )

    buf = LandmarkBuffer(maxlen=WINDOW)

    # Phasensteuerung (Rot/Grün)
    phase_color = "red"
    phase_dur = red_dur
    phase_start = time.time()
    phase_end = phase_start + phase_dur

    print(f"[INFO] Live-Modus gestartet. 'q' zum Beenden. Takt: {green_dur:.1f}s Grün / {red_dur:.1f}s Rot.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Kein Frame gelesen – beende.")
                break

            frame = cv2.flip(frame, 1)

            # MediaPipe
            frame, hands = tracker.process_frame(frame, draw_landmarks=True)

            # Landmarks → normiert → Buffer
            if hands:
                lm = [tuple(x) for x in hands[0].coords_norm]
            else:
                lm = [(np.nan, np.nan, np.nan)] * 21
            lm_vec = normalize_landmarks(lm)  # (63,)
            buf.push(lm_vec)

            now = time.time()
            # Phase umschalten
            if now >= phase_end:
                if phase_color == "red":
                    phase_color = "green"
                    phase_dur = green_dur
                else:
                    phase_color = "red"
                    phase_dur = red_dur
                phase_start = now
                phase_end = phase_start + phase_dur

            secs_left = max(0.0, phase_end - now)

            # HUD
            put_hud(frame, [
                f"run_live | Phase: {phase_color} | time left: {secs_left:0.1f}s",
                "q: quit",
            ])

            # Klassifikation nur in Grün-Phase
            if phase_color == "green" and buf.full():
                window_arr = buf.as_array()
                label, conf = _predict(pipe, le, window_arr)
                if conf >= conf_threshold and on_prediction is not None:
                    # Callback darf das Frame in-place modifizieren (z. B. Bild einblenden, Rahmen, etc.)
                    on_prediction(label, conf, frame, phase_color, secs_left)

            # Anzeige
            cv2.imshow("Live – Gesture Runtime", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
