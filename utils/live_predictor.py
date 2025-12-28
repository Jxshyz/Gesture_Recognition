# utils/live_predictor.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple

import cv2
import numpy as np
import mediapipe as mp

from utils.feature_extractor import normalize_landmarks, window_features
from utils.prediction_utils import PredictionAggregator
from utils.model_io import load_model_and_encoder, predict_feat189, decode_label


@dataclass
class PredictorConfig:
    camera_index: int = 0
    window_size: int = 12                 # muss wie Training sein
    pred_min_interval_s: float = 0.06     # ~16-17 Predictions/s (flüssig)
    min_det_conf: float = 0.6
    min_track_conf: float = 0.6
    flip: bool = True
    draw_landmarks: bool = False
    show_window: bool = False


def run_live_predictions(
    on_pred: Callable[[str, float, float], None],
    on_frame: Optional[Callable[[np.ndarray], None]] = None,
    cfg: PredictorConfig = PredictorConfig(),
) -> None:
    """
    Ruft on_pred(label, conf, ts) in eurer gewünschten Rate auf.
    Keine FSM hier – nur "Pipeline liefert pro Frame pred_label".
    """
    model, le = load_model_and_encoder()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=cfg.min_det_conf,
        min_tracking_confidence=cfg.min_track_conf,
    )

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {cfg.camera_index} konnte nicht geöffnet werden.")

    window: List[np.ndarray] = []
    pred_agg = PredictionAggregator(min_interval_s=cfg.pred_min_interval_s)

    def extract_lm_list(results) -> Optional[List[Tuple[float, float, float]]]:
        if not results.multi_hand_landmarks:
            return None
        hand = results.multi_hand_landmarks[0]
        return [(p.x, p.y, p.z) for p in hand.landmark]

    def classify(win12: np.ndarray) -> Tuple[str, float]:
        feat189 = window_features(win12)
        y_enc, conf = predict_feat189(model, feat189)
        label = decode_label(y_enc, le)
        return label, float(conf)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if cfg.flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if cfg.draw_landmarks and results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            lm_list = extract_lm_list(results)
            if lm_list is not None:
                x63 = normalize_landmarks(lm_list)  # (63,)
                window.append(x63)
                if len(window) > cfg.window_size:
                    window.pop(0)

                if len(window) == cfg.window_size:
                    now = time.time()
                    tmp_label, tmp_conf = classify(np.asarray(window, dtype=np.float32))
                    if pred_agg.feed(tmp_label, tmp_conf, now):
                        on_pred(tmp_label, tmp_conf, now)

            if on_frame is not None:
                on_frame(frame)

            if cfg.show_window:
                cv2.imshow("Runner Predictor", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            hands.close()
        except Exception:
            pass
