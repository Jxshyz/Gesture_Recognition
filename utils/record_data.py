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
    fps: float = 30.0
    record_s: float = 2.0
    cooldown_s: float = 2.0

    window_size: int = 12
    samples_per_gesture: int = 10

    min_det_conf: float = 0.6
    min_track_conf: float = 0.6

    out_dir: Path = Path("./data/recordings")


# ❌ Entfernt: swipe_up, rotate_right, neutral_peace, garbage
GESTURE_ORDER: List[str] = [
    "swipe_left",
    "swipe_right",
    "rotate_left",
    "close_fist",
    "neutral_palm",
    "finger_pistol",
    "pinch",
]


def _resample_to_T(seq: np.ndarray, T: int) -> np.ndarray:
    N, D = seq.shape
    if N <= 0:
        return np.zeros((T, D), dtype=np.float32)
    if N == T:
        return seq.astype(np.float32, copy=False)

    xs = np.linspace(0, 1, N)
    xt = np.linspace(0, 1, T)
    out = np.zeros((T, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(xt, xs, seq[:, d])
    return out


def _pick_hand(results, desired_hand: str) -> Optional[List[Tuple[float, float, float]]]:
    if not results.multi_hand_landmarks:
        return None

    if results.multi_handedness and len(results.multi_handedness) == len(results.multi_hand_landmarks):
        for idx, hd in enumerate(results.multi_handedness):
            label = hd.classification[0].label
            if label.lower() == desired_hand.lower():
                hand = results.multi_hand_landmarks[idx]
                return [(p.x, p.y, p.z) for p in hand.landmark]

    hand = results.multi_hand_landmarks[0]
    return [(p.x, p.y, p.z) for p in hand.landmark]


def _draw_ui_box(
    frame_bgr: np.ndarray,
    border_bgr: Tuple[int, int, int],
    title: str,
    subtitle: str,
    timer_text: str,
):
    x0, y0 = 20, 20
    w, h = 360, 170

    cv2.rectangle(frame_bgr, (x0, y0), (x0 + w, y0 + h), (245, 245, 245), -1)
    cv2.rectangle(frame_bgr, (x0, y0), (x0 + w, y0 + h), border_bgr, 6)

    cv2.putText(frame_bgr, title, (x0 + 14, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, subtitle, (x0 + 14, y0 + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(
        frame_bgr, timer_text, (x0 + 14, y0 + 144), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2, cv2.LINE_AA
    )

    cv2.putText(
        frame_bgr,
        "q=quit | s=skip sample | n=skip gesture",
        (x0 + 14, y0 + h + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_record(
    gesture_arg: str,
    name: str,
    camera_index: int = 0,
    hand: str = "Right",
    cfg: RecordConfig = RecordConfig(),
):
    gesture_arg = gesture_arg.lower().strip()
    desired_hand = "Right" if str(hand).lower().startswith("r") else "Left"

    if gesture_arg == "all":
        plan: List[Tuple[str, int]] = [(g, cfg.samples_per_gesture) for g in GESTURE_ORDER]
    else:
        plan = [(gesture_arg, cfg.samples_per_gesture)]

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {camera_index} konnte nicht geöffnet werden.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands_model = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=cfg.min_det_conf,
        min_tracking_confidence=cfg.min_track_conf,
    )

    out_base = cfg.out_dir / name / desired_hand
    _ensure_dir(out_base)

    print("\n============================================================")
    print("RECORD MODE")
    print(f"  user/name:     {name}")
    print(f"  hand:          {desired_hand}")
    print(f"  camera_index:  {camera_index}")
    print(f"  fps target:    {cfg.fps}")
    print(f"  record_s:      {cfg.record_s}")
    print(f"  cooldown_s:    {cfg.cooldown_s}")
    print(f"  out_dir:       {out_base.resolve()}")
    print("============================================================\n")

    frame_interval = 1.0 / float(cfg.fps)
    quit_all = False

    try:
        for gi, (gesture, n_samples) in enumerate(plan, start=1):
            if quit_all:
                break

            print(f"\n--- Gesture [{gi}/{len(plan)}]: {gesture} | samples={n_samples} ---")
            gesture_dir = out_base / gesture
            _ensure_dir(gesture_dir)

            sample_idx = 0
            while sample_idx < n_samples and not quit_all:
                # COOLDOWN
                t0 = time.time()
                while True:
                    now = time.time()
                    remaining = cfg.cooldown_s - (now - t0)
                    if remaining <= 0:
                        break

                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands_model.process(rgb)
                    if results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                    _draw_ui_box(
                        frame, (0, 0, 255), "COOLDOWN", f"{gesture}  ({sample_idx+1}/{n_samples})", f"{remaining:0.2f}s"
                    )
                    cv2.imshow("Record Data", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        quit_all = True
                        break
                    if key == ord("n"):
                        sample_idx = n_samples
                        break
                    if key == ord("s"):
                        remaining = 0
                        break

                if quit_all or sample_idx >= n_samples:
                    break

                # RECORDING
                seq: List[np.ndarray] = []
                t_start = time.time()
                next_sample_t = t_start

                while True:
                    now = time.time()
                    elapsed = now - t_start
                    remaining = cfg.record_s - elapsed
                    if remaining <= 0:
                        break

                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands_model.process(rgb)
                    if results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                    if now >= next_sample_t:
                        next_sample_t += frame_interval
                        lm = _pick_hand(results, desired_hand=desired_hand)
                        if lm is not None:
                            x63 = normalize_landmarks(lm).astype(np.float32, copy=False)
                            seq.append(x63)

                    _draw_ui_box(
                        frame,
                        (0, 255, 0),
                        "RECORDING",
                        f"{gesture}  ({sample_idx+1}/{n_samples})",
                        f"{remaining:0.2f}s | frames:{len(seq)}",
                    )
                    cv2.imshow("Record Data", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        quit_all = True
                        break
                    if key == ord("n"):
                        sample_idx = n_samples
                        break
                    if key == ord("s"):
                        seq = []
                        break

                if quit_all or sample_idx >= n_samples:
                    break

                if len(seq) < 4:
                    print(f"  [WARN] too few frames ({len(seq)}). Not saved. Repeat sample.")
                    continue

                seq_arr = np.asarray(seq, dtype=np.float32)
                seq12 = _resample_to_T(seq_arr, cfg.window_size)

                ts = int(time.time())
                out_path = gesture_dir / f"{gesture}_{name}_{desired_hand}_{ts}_{sample_idx:03d}.npz"
                np.savez_compressed(
                    out_path,
                    seq=seq_arr,
                    seq12=seq12,
                    label=gesture,
                    name=name,
                    hand=desired_hand,
                    fps=float(cfg.fps),
                    record_s=float(cfg.record_s),
                    cooldown_s=float(cfg.cooldown_s),
                    ts=float(time.time()),
                )
                print(f"  saved: {out_path.name} | raw_frames={seq_arr.shape[0]}")
                sample_idx += 1

        print("\nDONE. Recording finished.")
        print(f"Output root: {out_base.resolve()}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            hands_model.close()
        except Exception:
            pass
