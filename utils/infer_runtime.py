"""
Runtime module for real-time gesture inference with temporal segmentation.

This module implements a full live gesture pipeline:

- MediaPipe landmark extraction
- Sliding window feature generation
- Model inference (189-dim feature vector)
- Time-based arming (neutral hold)
- Segment-based gesture recording
- Early commit via majority aggregation
- Cooldown handling
- Optional UI callbacks and overlays

Designed for interactive real-time applications (e.g., games).
"""
# utils/infer_runtime.py
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
class LiveConfig:
    """
    Configuration container for live gesture inference.

    Parameters must match training setup where applicable.

    Attributes:
        window_size (int):
            Number of frames used for sliding window classification.

        neutral_label (str):
            Label used as neutral arming gesture.

        garbage_label (str):
            Label representing invalid/noise predictions.

        neutral_min_conf (float):
            Minimum confidence required to count as neutral.

        neutral_hold_s (float):
            Time (seconds) neutral must be held before entering ARMED state.

        record_min_frames (int):
            Minimum number of frames required for fallback segment classification.

        record_max_frames (int):
            Maximum frames allowed for one recording segment.

        end_hold_frames (int):
            Number of consecutive neutral frames required to end recording.

        commit_min_samples (int):
            Minimum number of aggregated predictions for early commit.

        commit_min_conf (float):
            Minimum average confidence for early commit.

        commit_frame_conf_gate (float):
            Minimum per-frame confidence required to enter commit aggregation.

        cooldown_s (float):
            Cooldown duration (seconds) after a committed gesture.

        pred_min_interval_s (float):
            Minimum time between model predictions (rate limiting).

        telemetry_interval_s (float):
            Interval for telemetry callback updates.
    """

    # Has to be exactly as in training
    window_size: int = 12

    neutral_label: str = "neutral_palm"
    garbage_label: str = "garbage"

    # Arming: neutral has to be held X seconds (time based)
    neutral_min_conf: float = 0.60
    neutral_hold_s: float = 0.5

    # Recording/Commit
    record_min_frames: int = 8
    record_max_frames: int = 80

    # Segment-Ende: neutral stable as end (frame based)
    end_hold_frames: int = 6

    # Early-Commit: n Samples + Conf
    commit_min_samples: int = 3
    commit_min_conf: float = 0.60  # avg-conf
    commit_frame_conf_gate: float = 0.55  # einzel-frame gate

    # Cooldown: short rest after Commit (time based)
    cooldown_s: float = 0.5  # <-- HIER: cooldown (0.5s)

    # Classification rate (max. Predictions/s)
    # 0.075 ≈ 13.3/s, 0.05 ≈ 20/s
    pred_min_interval_s: float = 0.075

    # How often telemetry (UI) should be updated
    telemetry_interval_s: float = 1.0 / 15.0  # 15 Hz


STATE_COLORS = {
    "IDLE": (180, 180, 180),
    "ARMED": (0, 255, 0),
    "RECORDING": (0, 150, 255),
    "COOLDOWN": (0, 0, 255),
}


def _draw_phase_overlay(
    frame_bgr,
    state: str,
    seconds_left: float,
    live_label: str,
    live_conf: float,
    armed_progress: float,
):
    """
    Draw a visual state overlay onto the frame.

    Displays:
        - Current FSM state
        - Remaining time in current phase
        - Arming progress (percentage)
        - Live prediction (label + confidence)

    Parameters:
        frame_bgr (np.ndarray): Frame modified in place.
        state (str): Current runtime state.
        seconds_left (float): Remaining time in current phase.
        live_label (str): Current predicted label.
        live_conf (float): Current prediction confidence.
        armed_progress (float): Arming progress in range [0,1].
    """
    h, w = frame_bgr.shape[:2]
    color = STATE_COLORS.get(state, (200, 200, 200))
    cv2.rectangle(frame_bgr, (10, 10), (420, 140), color, -1)

    cv2.putText(
        frame_bgr,
        f"STATE: {state}",
        (22, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"sec_left: {seconds_left:.2f}",
        (22, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame_bgr,
        f"ARM: {armed_progress*100:.0f}%",
        (22, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # live prediction bottom left
    cv2.rectangle(frame_bgr, (10, h - 60), (520, h - 10), (40, 40, 40), -1)
    cv2.putText(
        frame_bgr,
        f"LIVE: {live_label or '-'} ({live_conf:.2f})",
        (22, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _extract_lm_list(results) -> Optional[List[Tuple[float, float, float]]]:
    """
    Extract the first detected hand's landmarks from MediaPipe results.

    Parameters:
        results: MediaPipe inference result object.

    Returns:
        Optional[List[Tuple[float, float, float]]]:
            List of 21 (x, y, z) normalized coordinates,
            or None if no hand is detected.
    """
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    return [(p.x, p.y, p.z) for p in hand.landmark]


def _resample_to_T(seq: np.ndarray, T: int) -> np.ndarray:
    """
    Resample a variable-length sequence to fixed length T.

    Uses linear interpolation along the temporal axis.

    Parameters:
        seq (np.ndarray): Array of shape (N, 63).
        T (int): Target number of frames.

    Returns:
        np.ndarray: Resampled array of shape (T, 63).
    """
    N, D = seq.shape
    if N == T:
        return seq
    xs = np.linspace(0, 1, N)
    xt = np.linspace(0, 1, T)
    out = np.zeros((T, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(xt, xs, seq[:, d])
    return out


def run_live(
    camera_index: int = 0,
    show_window: bool = True,
    draw_phase_overlay: bool = True,
    on_prediction: Optional[
        Callable[[str, float, np.ndarray, str, float], None]
    ] = None,
    on_render: Optional[Callable[[np.ndarray, str, float], None]] = None,
    on_telemetry: Optional[
        Callable[[str, str, float, float, float, bool], None]
    ] = None,
    cfg: LiveConfig = LiveConfig(),
):
    """
    Run the full real-time gesture inference loop.

    Pipeline:
        1. Capture webcam frames
        2. Extract hand landmarks via MediaPipe
        3. Normalize landmarks (63-dim)
        4. Maintain sliding window (window_size frames)
        5. Perform model inference (189-dim features)
        6. Apply time-based FSM:
               IDLE → ARMED → RECORDING → COOLDOWN
        7. Trigger commit events for valid gestures
        8. Call optional UI callbacks

    State Logic:

        IDLE:
            Waits for stable neutral gesture (arming).

        ARMED:
            Neutral confirmed; waits for motion start.

        RECORDING:
            Collects frames for gesture segment.
            Supports early commit via aggregated predictions.
            Falls back to segment classification if needed.

        COOLDOWN:
            Short rest phase after a committed gesture.

    Callbacks:

        on_prediction(label, conf, frame, state, seconds_left):
            Triggered exactly once per committed gesture.

        on_render(frame, state, seconds_left):
            Called every frame before display.

        on_telemetry(state, live_label, live_conf,
                     seconds_left, armed_progress, armed_ready):
            Called periodically for UI updates.

    Parameters:
        camera_index (int):
            Camera device index.

        show_window (bool):
            Whether to display the OpenCV window.

        draw_phase_overlay (bool):
            Whether to draw the state overlay.

        cfg (LiveConfig):
            Runtime configuration parameters.

    Returns:
        None
    """
    model, le = load_model_and_encoder()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {camera_index} konnte nicht geöffnet werden.")

    # Optional: set FPS
    cap.set(cv2.CAP_PROP_FPS, 30)

    window: List[np.ndarray] = []
    state = "IDLE"
    rec_frames: List[np.ndarray] = []

    pred_agg = PredictionAggregator(min_interval_s=cfg.pred_min_interval_s)
    commit_agg = PredictionAggregator(min_interval_s=cfg.pred_min_interval_s)

    live_label, live_conf = "", 0.0

    # Arming (time based)
    neutral_hold_t = 0.0

    # Cooldown (time based)
    cooldown_until = 0.0

    # End detection (frames)
    end_counter = 0

    # Telemetry throttle
    last_tel_t = 0.0

    last_loop_t = time.time()

    def classify_from_window(win12: np.ndarray) -> Tuple[str, float]:
        feat189 = window_features(win12)
        y_enc, conf = predict_feat189(model, feat189)
        label = decode_label(y_enc, le)
        return label, conf

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.time()
            dt = now - last_loop_t
            last_loop_t = now

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )

            lm_list = _extract_lm_list(results)
            x63 = None
            if lm_list is not None:
                x63 = normalize_landmarks(lm_list)
                window.append(x63)
                if len(window) > cfg.window_size:
                    window.pop(0)

            # LIVE prediction (throttled)
            if x63 is not None and len(window) == cfg.window_size:
                win12 = np.asarray(window, dtype=np.float32)
                tmp_label, tmp_conf = classify_from_window(win12)
                if pred_agg.feed(tmp_label, tmp_conf, now):
                    live_label, live_conf = tmp_label, tmp_conf

            # Defaults for telemetry
            armed_progress = 0.0
            armed_ready = False
            seconds_left = 0.0

            # -----------------------------
            # FSM
            # -----------------------------
            if state == "IDLE":
                # neutral hold time accumulates only if confidently neutral
                if (
                    live_label == cfg.neutral_label
                    and live_conf >= cfg.neutral_min_conf
                ):
                    neutral_hold_t += dt
                else:
                    neutral_hold_t = 0.0

                armed_progress = min(1.0, neutral_hold_t / cfg.neutral_hold_s)
                seconds_left = max(0.0, cfg.neutral_hold_s - neutral_hold_t)

                if neutral_hold_t >= cfg.neutral_hold_s:
                    state = "ARMED"
                    armed_ready = True
                    commit_agg.reset()
                    rec_frames = []
                    end_counter = 0

            elif state == "ARMED":
                # bar remains full until movement starts
                armed_progress = 1.0
                armed_ready = True
                seconds_left = 0.0

                # as soon as neutral is left -> RECORDING
                if not (
                    live_label == cfg.neutral_label
                    and live_conf >= cfg.neutral_min_conf
                ):
                    state = "RECORDING"
                    neutral_hold_t = 0.0
                    rec_frames = []
                    commit_agg.reset()
                    end_counter = 0

            elif state == "RECORDING":
                # collec
                if x63 is not None:
                    rec_frames.append(x63)

                # early commit only for real gestures
                if (
                    live_label
                    and live_label not in (cfg.garbage_label, cfg.neutral_label)
                    and live_conf >= cfg.commit_frame_conf_gate
                ):
                    commit_agg.feed(live_label, live_conf, now)

                maj_label, maj_conf, maj_n = commit_agg.result()
                if (
                    maj_label is not None
                    and maj_n >= cfg.commit_min_samples
                    and maj_conf >= cfg.commit_min_conf
                ):
                    # COMMIT (exactly 1 gesture)
                    if on_prediction is not None:
                        on_prediction(maj_label, float(maj_conf), frame, "COMMIT", 0.0)

                    state = "COOLDOWN"
                    cooldown_until = now + cfg.cooldown_s
                    rec_frames = []
                    end_counter = 0
                    commit_agg.reset()

                else:
                    # end: back to neutral stable or max frames
                    if (
                        live_label == cfg.neutral_label
                        and live_conf >= cfg.neutral_min_conf
                    ):
                        end_counter += 1
                    else:
                        end_counter = 0

                    if (
                        len(rec_frames) >= cfg.record_max_frames
                        or end_counter >= cfg.end_hold_frames
                    ):
                        # Fallback segment classify
                        if len(rec_frames) >= cfg.record_min_frames:
                            seg = np.asarray(rec_frames, dtype=np.float32)
                            seg12 = _resample_to_T(seg, cfg.window_size)
                            seg_label, seg_conf = classify_from_window(seg12)

                            if (
                                seg_label not in (cfg.garbage_label, cfg.neutral_label)
                                and seg_conf >= cfg.commit_min_conf
                            ):
                                if on_prediction is not None:
                                    on_prediction(
                                        seg_label, float(seg_conf), frame, "COMMIT", 0.0
                                    )

                        state = "COOLDOWN"
                        cooldown_until = now + cfg.cooldown_s
                        rec_frames = []
                        end_counter = 0
                        commit_agg.reset()

            elif state == "COOLDOWN":
                remaining = cooldown_until - now
                seconds_left = max(0.0, remaining)
                if remaining <= 0:
                    state = "IDLE"
                    neutral_hold_t = 0.0
                    end_counter = 0
                    commit_agg.reset()

            # -----------------------------
            # callbacks
            # -----------------------------
            if on_render is not None:
                on_render(frame, state, seconds_left)

            if on_telemetry is not None:
                if (now - last_tel_t) >= cfg.telemetry_interval_s:
                    last_tel_t = now
                    on_telemetry(
                        state,
                        live_label or "-",
                        float(live_conf),
                        float(seconds_left),
                        float(armed_progress),
                        bool(armed_ready),
                    )

            if draw_phase_overlay:
                _draw_phase_overlay(
                    frame, state, seconds_left, live_label, live_conf, armed_progress
                )

            if show_window:
                cv2.imshow("Gesture Live", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            hands.close()
        except Exception:
            pass
