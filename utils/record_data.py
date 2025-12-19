# utils/record_data.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

from utils.feature_extractor import normalize_landmarks


@dataclass
class RecordConfig:
    gestures: Tuple[str, ...] = (
        # Startpose / Neutral
        "neutral_palm",
        # Eure Actions:
        "swipe_left",
        "swipe_right",
        "rotate",
        "fist",
        # Negativklasse:
        "garbage",
    )

    # Wie oft pro Geste aufnehmen?
    sequences_per_gesture: int = 25

    # Phasen-Timing
    pre_red_s: float = 1.5  # "komm in neutral_palm / bereit machen"
    green_s: float = 1.0  # "jetzt Geste ausführen"
    post_red_s: float = 0.8  # "zurück zu neutral_palm / relax"

    # Frames, die wir zusätzlich nach Green noch einsammeln (damit das Ende der Bewegung drin ist)
    tail_capture_s: float = 0.35

    # Handfilter
    prefer_hand: str = "Right"  # "Right" oder "Left" (MediaPipe Sprachgebrauch)

    # MediaPipe
    min_det_conf: float = 0.6
    min_track_conf: float = 0.6


def _pick_hand_index(results, prefer_hand: str) -> int:
    """
    Wählt die Hand (Index) passend zur gewünschten Händigkeit.
    Fallback: 0.
    """
    if not results.multi_hand_landmarks:
        return -1
    if not results.multi_handedness:
        return 0

    prefer = prefer_hand.lower()
    best_idx = 0
    best_score = -1.0
    for i, h in enumerate(results.multi_handedness):
        label = h.classification[0].label.lower()  # "left"/"right"
        score = h.classification[0].score
        if label == prefer and score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _draw_ui(frame, gesture: str, phase: str, secs_left: float, seq_i: int, seq_total: int):
    h, w = frame.shape[:2]
    # top-left square
    phase_up = phase.upper()
    if phase_up.startswith("GREEN"):
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    cv2.rectangle(frame, (10, 10), (110, 110), color, -1)

    cv2.putText(frame, f"{gesture}", (130, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Phase: {phase}  ({secs_left:.2f}s)",
        (130, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(frame, (10, h - 60), (520, h - 10), (30, 30, 30), -1)
    cv2.putText(
        frame,
        f"Seq {seq_i}/{seq_total}  |  q=quit  |  s=skip gesture",
        (20, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )


def run_record(gesture_arg: str, name: str, camera_index: int = 0, hand: str = "Right", cfg: RecordConfig = RecordConfig()):
    """
    gesture_arg: 'all' oder eine konkrete Geste (z.B. 'swipe_left')
    hand: 'Right'/'Left' (optional)
    name: Person/ID Ordner
    """

    gesture_arg_l = str(gesture_arg).lower()

    # --- Welche Gesten aufnehmen? ---
    cfg_gestures_lower = [g.lower() for g in cfg.gestures]

    if gesture_arg_l in ("all", "*"):
        gestures_to_record = cfg.gestures
    elif gesture_arg_l in cfg_gestures_lower:
        # nur diese eine Geste aufnehmen
        gestures_to_record = tuple(g for g in cfg.gestures if g.lower() == gesture_arg_l)
    elif gesture_arg_l in ("l", "left", "r", "right"):
        # Backwards-compat: falls jemand doch nur Hand übergibt
        hand = "Right" if gesture_arg_l.startswith("r") else "Left"
        gestures_to_record = cfg.gestures
    else:
        raise ValueError(
            f"Unknown gesture_arg='{gesture_arg}'. Use 'all' or one of: {cfg.gestures}"
        )

    prefer_hand = "Right" if str(hand).lower().startswith("r") else "Left"

    # cfg neu setzen (mit ausgewählten Gesten + Hand)
    cfg = RecordConfig(
        gestures=gestures_to_record,
        sequences_per_gesture=cfg.sequences_per_gesture,
        pre_red_s=cfg.pre_red_s,
        green_s=cfg.green_s,
        post_red_s=cfg.post_red_s,
        tail_capture_s=cfg.tail_capture_s,
        prefer_hand=prefer_hand,
        min_det_conf=cfg.min_det_conf,
        min_track_conf=cfg.min_track_conf,
    )

    data_root = Path("./data_raw")
    data_root.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {camera_index} konnte nicht geöffnet werden.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=cfg.min_det_conf,
        min_tracking_confidence=cfg.min_track_conf,
    )

    print(f"[INFO] Recording for hand={cfg.prefer_hand}, name={name}, cam={camera_index}")
    print(f"[INFO] Gestures: {cfg.gestures}")
    print(f"[INFO] Sequences per gesture: {cfg.sequences_per_gesture}")
    print("[INFO] Controls: q=quit, s=skip current gesture")

    try:
        for gesture in cfg.gestures:
            out_dir = data_root / gesture / name
            out_dir.mkdir(parents=True, exist_ok=True)

            seq_total = cfg.sequences_per_gesture
            seq_i = 1

            while seq_i <= seq_total:
                # --- Pre phase (RED) ---
                t0 = time.time()
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)

                    secs_left = max(0.0, cfg.pre_red_s - (time.time() - t0))
                    _draw_ui(frame, gesture, "RED (prepare)", secs_left, seq_i, seq_total)

                    cv2.imshow("Record Data", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        return
                    if key == ord("s"):
                        print(f"[SKIP] Gesture {gesture}")
                        seq_i = seq_total + 1
                        break

                    if secs_left <= 0:
                        break
                if seq_i > seq_total:
                    break

                # --- GREEN capture ---
                seq_frames: List[np.ndarray] = []
                t1 = time.time()
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        idx = _pick_hand_index(results, cfg.prefer_hand)
                        if idx >= 0:
                            mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[idx], mp_hands.HAND_CONNECTIONS)
                            lm = results.multi_hand_landmarks[idx]
                            lm_list = [(p.x, p.y, p.z) for p in lm.landmark]
                            x63 = normalize_landmarks(lm_list)
                            seq_frames.append(x63)

                    secs_left = max(0.0, cfg.green_s - (time.time() - t1))
                    _draw_ui(frame, gesture, "GREEN (do gesture)", secs_left, seq_i, seq_total)

                    cv2.imshow("Record Data", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        return
                    if secs_left <= 0:
                        break

                # --- Tail capture ---
                t_tail = time.time()
                while time.time() - t_tail < cfg.tail_capture_s:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    if results.multi_hand_landmarks:
                        idx = _pick_hand_index(results, cfg.prefer_hand)
                        if idx >= 0:
                            lm = results.multi_hand_landmarks[idx]
                            lm_list = [(p.x, p.y, p.z) for p in lm.landmark]
                            x63 = normalize_landmarks(lm_list)
                            seq_frames.append(x63)

                    _draw_ui(
                        frame,
                        gesture,
                        "RED (tail)",
                        max(0.0, cfg.tail_capture_s - (time.time() - t_tail)),
                        seq_i,
                        seq_total,
                    )
                    cv2.imshow("Record Data", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        return

                # --- Post phase (RED) ---
                t2 = time.time()
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    secs_left = max(0.0, cfg.post_red_s - (time.time() - t2))
                    _draw_ui(frame, gesture, "RED (reset)", secs_left, seq_i, seq_total)

                    cv2.imshow("Record Data", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        return
                    if secs_left <= 0:
                        break

                # --- Save sequence ---
                if len(seq_frames) < 4:
                    print(f"[WARN] Too few frames for {gesture} seq {seq_i}, discarded ({len(seq_frames)} frames).")
                else:
                    arr = np.asarray(seq_frames, dtype=np.float32)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_path = out_dir / f"{ts}_seq{seq_i:03d}.npy"
                    np.save(out_path, arr)
                    print(f"[OK] Saved {out_path.as_posix()}  shape={arr.shape}")

                seq_i += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            hands.close()
        except Exception:
            pass

