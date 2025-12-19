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
    window_size: int = 12  # MUSS wie Training sein
    neutral_label: str = "neutral_palm"  # nur DAS gilt als Startpose
    garbage_label: str = "garbage"

    # Arming: neutral muss stabil gehalten werden
    neutral_min_conf: float = 0.60
    neutral_hold_frames: int = 8

    # Aufnahme/Commit
    record_min_frames: int = 8
    record_max_frames: int = 80

    # Segment-Ende: zurück zu neutral stabil
    end_hold_frames: int = 6

    # Early-Commit: n Samples + Conf
    commit_min_samples: int = 3
    commit_min_conf: float = 0.60  # avg-conf
    commit_frame_conf_gate: float = 0.55  # einzel-frame gate

    # Cooldown: kurze Pause nach Commit
    cooldown_frames: int = 12

    # Prediction throttling
    pred_min_interval_s: float = 0.075


STATE_COLORS = {
    "IDLE": (180, 180, 180),
    "ARMED": (0, 255, 0),
    "RECORDING": (0, 150, 255),
    "COOLDOWN": (0, 0, 255),
}


def _draw_phase_overlay(frame_bgr, state: str, seconds_left: float, live_label: str, live_conf: float):
    h, w = frame_bgr.shape[:2]
    color = STATE_COLORS.get(state, (200, 200, 200))
    cv2.rectangle(frame_bgr, (10, 10), (360, 110), color, -1)

    cv2.putText(frame_bgr, f"STATE: {state}", (22, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(
        frame_bgr, f"sec_left: {seconds_left:.2f}", (22, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA
    )

    # unten links live prediction
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
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    return [(p.x, p.y, p.z) for p in hand.landmark]


def _resample_to_T(seq: np.ndarray, T: int) -> np.ndarray:
    """seq: (N,63) -> (T,63)"""
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
    on_prediction: Optional[Callable[[str, float, np.ndarray, str, float], None]] = None,
    on_render: Optional[Callable[[np.ndarray, str, float], None]] = None,
    cfg: LiveConfig = LiveConfig(),
):
    """
    on_prediction: wird NUR bei COMMIT ausgelöst (also eine echte Geste)
    on_render: wird jeden Frame aufgerufen (für Web/Tetris embedding etc.)
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

    # Sliding window für 12 Frames (Training)
    window: List[np.ndarray] = []

    # FSM states
    state = "IDLE"
    neutral_counter = 0
    end_counter = 0
    cooldown = 0

    # Recording buffer
    rec_frames: List[np.ndarray] = []

    # Live pred throttle
    pred_agg = PredictionAggregator(min_interval_s=cfg.pred_min_interval_s)

    # Early-commit aggregator (nur RECORDING)
    commit_agg = PredictionAggregator(min_interval_s=cfg.pred_min_interval_s)

    live_label, live_conf = "", 0.0
    last_pred_t = 0.0

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

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Hand zeichnen
            if results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            lm_list = _extract_lm_list(results)
            x63 = None
            if lm_list is not None:
                x63 = normalize_landmarks(lm_list)  # (63,)

                # window füllen
                window.append(x63)
                if len(window) > cfg.window_size:
                    window.pop(0)

            # LIVE prediction (für State-Logik + Overlay)
            now = time.time()
            if x63 is not None and len(window) == cfg.window_size:
                win12 = np.asarray(window, dtype=np.float32)
                # throttle über PredictionAggregator:
                # wir nutzen label/conf trotzdem fürs overlay -> nur aktualisieren wenn feed True war
                tmp_label, tmp_conf = classify_from_window(win12)
                if pred_agg.feed(tmp_label, tmp_conf, now):
                    live_label, live_conf = tmp_label, tmp_conf

            # --- FSM LOGIK ---
            seconds_left = 0.0

            # 1) IDLE: warten bis neutral_palm stabil
            if state == "IDLE":
                if live_label == cfg.neutral_label and live_conf >= cfg.neutral_min_conf:
                    neutral_counter += 1
                else:
                    neutral_counter = 0

                seconds_left = max(0.0, (cfg.neutral_hold_frames - neutral_counter) / 30.0)

                if neutral_counter >= cfg.neutral_hold_frames:
                    state = "ARMED"
                    end_counter = 0
                    commit_agg.reset()
                    rec_frames = []

            # 2) ARMED: sobald wir von neutral weggehen → RECORDING starten
            elif state == "ARMED":
                # bleibt neutral, dann armed bleibt armed
                if live_label == cfg.neutral_label and live_conf >= cfg.neutral_min_conf:
                    pass
                else:
                    # Start recording
                    state = "RECORDING"
                    rec_frames = []
                    commit_agg.reset()
                    end_counter = 0

                seconds_left = 0.0

            # 3) RECORDING: Frames sammeln, Early-Commit + Ende bei neutral
            elif state == "RECORDING":
                if x63 is not None:
                    rec_frames.append(x63)

                # Early-Commit nur auf NICHT-garbage und NICHT-neutral
                if (
                    live_label
                    and live_label not in (cfg.garbage_label, cfg.neutral_label)
                    and live_conf >= cfg.commit_frame_conf_gate
                ):
                    commit_agg.feed(live_label, live_conf, now)

                maj_label, maj_conf, maj_n = commit_agg.result()
                if maj_label is not None and maj_n >= cfg.commit_min_samples and maj_conf >= cfg.commit_min_conf:
                    # COMMIT: sofort auslösen
                    if on_prediction is not None:
                        on_prediction(maj_label, float(maj_conf), frame, state, seconds_left)
                    state = "COOLDOWN"
                    cooldown = cfg.cooldown_frames
                    neutral_counter = 0
                    end_counter = 0
                    rec_frames = []
                    commit_agg.reset()
                else:
                    # Ende-Erkennung: zurück zu neutral stabil ODER max frames
                    if live_label == cfg.neutral_label and live_conf >= cfg.neutral_min_conf:
                        end_counter += 1
                    else:
                        end_counter = 0

                    if len(rec_frames) >= cfg.record_max_frames or end_counter >= cfg.end_hold_frames:
                        # Fallback: Segment einmal klassifizieren (nur wenn genug frames)
                        if len(rec_frames) >= cfg.record_min_frames:
                            seg = np.asarray(rec_frames, dtype=np.float32)  # (N,63)
                            seg12 = _resample_to_T(seg, cfg.window_size)
                            seg_label, seg_conf = classify_from_window(seg12)

                            # garbage/neutral blocken
                            if (
                                seg_label not in (cfg.garbage_label, cfg.neutral_label)
                                and seg_conf >= cfg.commit_min_conf
                            ):
                                if on_prediction is not None:
                                    on_prediction(seg_label, float(seg_conf), frame, state, seconds_left)

                        # egal ob erkannt oder nicht: cooldown
                        state = "COOLDOWN"
                        cooldown = cfg.cooldown_frames
                        neutral_counter = 0
                        end_counter = 0
                        rec_frames = []
                        commit_agg.reset()

            # 4) COOLDOWN: kurze Sperre, dann zurück zu IDLE
            elif state == "COOLDOWN":
                cooldown -= 1
                seconds_left = max(0.0, cooldown / 30.0)
                if cooldown <= 0:
                    state = "IDLE"
                    neutral_counter = 0
                    end_counter = 0
                    commit_agg.reset()

            # Render callback (für Tetris embedding usw.)
            if on_render is not None:
                on_render(frame, state, seconds_left)

            # Overlay im normalen Live-Fenster
            if draw_phase_overlay:
                _draw_phase_overlay(frame, state, seconds_left, live_label, live_conf)

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
