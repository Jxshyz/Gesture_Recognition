from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, Callable, Tuple
import cv2
import numpy as np
from joblib import load

from utils.hand_tracking import HandTracker, put_hud
from utils.feature_extractor import LandmarkBuffer, normalize_landmarks, window_features
from utils.prediction_utils import PredictionAggregator

MODEL_DIR = Path("./models")

OnPredFn = Callable[[str, float, np.ndarray, str, float], None]   # label, conf, frame, phase_color, secs_left
OnRenderFn = Callable[[np.ndarray, str, float], None]             # frame, phase_color, secs_left

def _predict(pipe, le, window_arr: np.ndarray) -> Tuple[int, float]:
    feats = window_features(window_arr).reshape(1, -1)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(feats)[0]
        y_idx = int(np.argmax(proba))
        conf = float(np.max(proba))
    else:
        y_idx = int(pipe.predict(feats)[0])
        conf = 1.0
    return y_idx, conf

def _draw_white_box_with_border(frame, phase_color: str):
    x0, y0 = 10, 10
    box_w, box_h = 160, 160
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), thickness=-1)
    col = (0, 200, 0) if phase_color == "green" else (0, 0, 200)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), col, thickness=3)

def run_live(camera_index: int = 0,
             width: Optional[int] = 1280,
             height: Optional[int] = 720,
             on_prediction: Optional[OnPredFn] = None,
             on_render: Optional[OnRenderFn] = None,
             green_dur: float = 2.0,
             red_dur: float = 2.0,
             conf_threshold: float = 0.0,
             final_min_votes: int = 3,
             final_min_conf: float = 0.55,
             downsample_interval_s: float = 0.075,
             enable_no_gesture: bool = True,
             no_gesture_label: str = "NO_GESTURE") -> None:

    pipe = load(MODEL_DIR / "gesture_model.joblib")
    le   = load(MODEL_DIR / "label_encoder.joblib")
    cfg  = load(MODEL_DIR / "config.joblib")
    WINDOW = int(cfg["WINDOW"])

    cap = cv2.VideoCapture(camera_index)
    if width is not None: cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print(f"[ERROR] Konnte Kamera {camera_index} nicht öffnen.")
        return

    tracker = HandTracker(
        static_image_mode=False, max_num_hands=1, model_complexity=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.6, draw_style=True,
    )

    buf = LandmarkBuffer(maxlen=WINDOW)
    aggr = PredictionAggregator(min_interval_s=downsample_interval_s)

    phase_color = "red"
    phase_dur = red_dur
    phase_start = time.time()
    phase_end = phase_start + phase_dur

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Kein Frame gelesen – beende.")
                break

            frame = cv2.flip(frame, 1)
            frame, hands = tracker.process_frame(frame, draw_landmarks=True)

            if hands:
                lm = [tuple(x) for x in hands[0].coords_norm]
            else:
                lm = [(np.nan, np.nan, np.nan)] * 21
            lm_vec = normalize_landmarks(lm)
            buf.push(lm_vec)

            now = time.time()
            if now >= phase_end:
                if phase_color == "green" and on_prediction is not None:
                    label_final, conf_avg, n_samples = aggr.result()
                    if label_final is None or n_samples < final_min_votes or conf_avg < final_min_conf:
                        if enable_no_gesture:
                            label_final, conf_avg = no_gesture_label, 0.0
                        else:
                            label_final = None
                    if label_final is not None:
                        on_prediction(label_final, float(conf_avg), frame, phase_color, 0.0)
                    aggr.reset()

                if phase_color == "red":
                    phase_color = "green"; phase_dur = green_dur
                else:
                    phase_color = "red"; phase_dur = red_dur
                phase_start = now
                phase_end = phase_start + phase_dur

            secs_left = max(0.0, phase_end - now)
            _draw_white_box_with_border(frame, phase_color)
            put_hud(frame, [f"run_live | Phase: {phase_color} | time left: {secs_left:0.1f}s", "q: quit"])

            if phase_color == "green" and buf.full():
                window_arr = buf.as_array()
                y_idx, conf = _predict(pipe, le, window_arr)
                if conf >= conf_threshold:
                    label = le.inverse_transform([y_idx])[0]
                    aggr.feed(label, conf, now)

            # NEU: persistente Darstellung pro Frame
            if on_render is not None:
                on_render(frame, phase_color, secs_left)

            cv2.imshow("Live – Gesture Runtime", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
