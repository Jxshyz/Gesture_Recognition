"""
Lightweight real-time gesture prediction pipeline.

This module provides a minimal live inference loop that:

- Captures webcam frames
- Extracts MediaPipe hand landmarks
- Builds sliding window features (189-dim)
- Runs model inference
- Emits throttled predictions via callback

No state machine or commit logic is included.
Designed to be composed with higher-level controllers (e.g. FSM).
"""
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
    """
    Configuration parameters for the live predictor.

    Attributes:
        camera_index (int):
            Index of the webcam device.

        window_size (int):
            Number of frames used for sliding window classification.
            Must match the training configuration.

        pred_min_interval_s (float):
            Minimum time interval between emitted predictions
            (rate limiting / smoothing).

        min_det_conf (float):
            Minimum detection confidence for MediaPipe.

        min_track_conf (float):
            Minimum tracking confidence for MediaPipe.

        flip (bool):
            Whether to horizontally flip the webcam frame.

        draw_landmarks (bool):
            Whether to draw detected landmarks onto the frame.

        show_window (bool):
            Whether to display the OpenCV window.
    """

    camera_index: int = 0
    window_size: int = 12  # has to be like in training
    pred_min_interval_s: float = 0.06  # ~16-17 Predictions/s (smooth)
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
    Run a real-time gesture prediction loop without FSM logic.

    Pipeline:
        1. Capture webcam frame
        2. Extract MediaPipe landmarks
        3. Normalize to 63-dim feature vector
        4. Maintain sliding window (window_size frames)
        5. Generate 189-dim temporal features
        6. Run model inference
        7. Throttle predictions via PredictionAggregator
        8. Emit predictions via callback

    The function continuously calls:

        on_pred(label, conf, timestamp)
            When a prediction passes the rate limiter.

        on_frame(frame)
            Every frame (optional), useful for streaming or visualization.

    No state machine, commit logic, or segmentation is performed here.
    This function delivers raw model predictions at a controlled rate.

    Parameters:
        on_pred (Callable):
            Callback receiving (label, confidence, timestamp).

        on_frame (Optional[Callable]):
            Optional callback receiving the current frame.

        cfg (PredictorConfig):
            Runtime configuration parameters.

    Returns:
        None
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
        """
        Extract normalized landmark coordinates from MediaPipe results.

        Parameters:
            results:
                MediaPipe inference result object.

        Returns:
            Optional[List[Tuple[float, float, float]]]:
                List of 21 (x, y, z) normalized coordinates if a hand
                is detected, otherwise None.
        """
        if not results.multi_hand_landmarks:
            return None
        hand = results.multi_hand_landmarks[0]
        return [(p.x, p.y, p.z) for p in hand.landmark]

    def classify(win12: np.ndarray) -> Tuple[str, float]:
        """
        Perform model inference on a fixed-length landmark window.

        The input window must have shape (window_size, 63).
        Temporal features (189-dim) are computed and passed to
        the trained classifier.

        Parameters:
            win12 (np.ndarray):
                Sliding window array of shape (T, 63).

        Returns:
            Tuple[str, float]:
                Predicted label and associated confidence score.
        """
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
                mp_draw.draw_landmarks(
                    frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )

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
