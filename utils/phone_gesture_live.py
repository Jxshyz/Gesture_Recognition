# utils/phone_gesture_live.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

from utils.feature_extractor import normalize_landmarks, window_features
from utils.model_io import load_model_and_encoder, predict_feat189, decode_label
from utils.prediction_utils import PredictionAggregator
from utils.phone_controller import AndroidDevice


KEYCODE_DPAD_LEFT = 21
KEYCODE_DPAD_RIGHT = 22
KEYCODE_DPAD_UP = 19
KEYCODE_DPAD_DOWN = 20
KEYCODE_ENTER = 66


@dataclass
class PhoneLiveConfig:
    camera_index: int = 0
    window_size: int = 12

    neutral_label: str = "neutral_palm"
    pistol_label: str = "finger_pistol"
    pinch_label: str = "pinch"

    palm_hold_s: float = 0.5
    pistol_hold_s: float = 0.5
    start_min_conf: float = 0.60

    pred_min_interval_s: float = 0.06

    commit_min_samples: int = 3
    commit_min_conf: float = 0.60
    commit_frame_conf_gate: float = 0.55
    cooldown_s: float = 0.35

    # Ohne garbage: reject, wenn Conf zu klein (damit no-hand nicht random triggert)
    reject_below_conf: float = 0.55

    # Drag movement (kohärent)
    drag_send_hz: float = 8.0
    drag_move_min_px: int = 14
    drag_duration_ms: int = 110

    pinch_hold_s: float = 0.30
    pinch_min_conf: float = 0.62
    pinch_release_s: float = 0.12

    track_smooth_alpha: float = 0.25

    display_scale: float = 2.0

    adb_path: Optional[str] = None
    serial: Optional[str] = None


STATE_COLORS = {
    "IDLE": (180, 180, 180),
    "GESTURE_ARMED": (0, 255, 0),
    "GESTURE_RECORDING": (0, 150, 255),
    "TRACKING": (255, 130, 130),
    "DRAGGING": (255, 220, 120),
    "COOLDOWN": (0, 0, 255),
}


def _extract_lm(results) -> Optional[np.ndarray]:
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    return np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)


def _resample_to_T(seq: np.ndarray, T: int) -> np.ndarray:
    N, D = seq.shape
    if N == T:
        return seq
    xs = np.linspace(0, 1, N)
    xt = np.linspace(0, 1, T)
    out = np.zeros((T, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(xt, xs, seq[:, d])
    return out


def _map_label_to_keyevent(label: str) -> Optional[int]:
    # swipe_up / rotate_right removed (training removed)
    if label == "swipe_left":
        return KEYCODE_DPAD_LEFT
    if label == "swipe_right":
        return KEYCODE_DPAD_RIGHT
    if label == "swipe_down":
        return KEYCODE_DPAD_DOWN
    if label == "rotate_left":
        return KEYCODE_DPAD_UP
    if label in ("close_fist", "pinch"):
        return KEYCODE_ENTER
    return None


def _draw_phone_inset(frame: np.ndarray, phone_w: int, phone_h: int, cursor_xy: Tuple[int, int], active: bool):
    h, w = frame.shape[:2]

    inset_w = 220
    inset_h = int(inset_w * (phone_h / max(1, phone_w)))
    inset_h = max(260, min(inset_h, 380))

    x0 = w - inset_w - 12
    y0 = h - inset_h - 12
    x1 = x0 + inset_w
    y1 = y0 + inset_h

    cv2.rectangle(frame, (x0, y0), (x1, y1), (10, 10, 10), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 2)

    title = "PHONE POINTER" + (" (ACTIVE)" if active else "")
    cv2.putText(frame, title, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

    cx, cy = cursor_xy
    ix = int(x0 + (cx / max(1, phone_w)) * inset_w)
    iy = int(y0 + (cy / max(1, phone_h)) * inset_h)

    cv2.circle(frame, (ix, iy), 8, (0, 0, 255), -1)
    cv2.circle(frame, (ix, iy), 12, (255, 255, 255), 2)


def _draw_ui(frame, mode: str,
             palm_prog: float, pistol_prog: float, pinch_prog: float,
             live_label: str, live_conf: float,
             last_commit: str,
             cursor_xy: Tuple[int, int],
             phone_size: Tuple[int, int]):
    h, _ = frame.shape[:2]
    col = STATE_COLORS.get(mode, (200, 200, 200))

    cv2.rectangle(frame, (10, 10), (760, 210), col, -1)
    cv2.putText(frame, f"MODE: {mode}", (22, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"LIVE: {live_label} ({live_conf:.2f})", (22, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"LAST: {last_commit}", (22, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    bar_w = 520
    bar_h = 16
    x0 = 10
    y0 = h - 122

    def bar(y, p, title):
        cv2.putText(frame, title, (x0, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (40, 40, 40), -1)
        fill = int(bar_w * max(0.0, min(1.0, p)))
        cv2.rectangle(frame, (x0, y), (x0 + fill, y + bar_h), (230, 230, 230), -1)

    bar(y0, palm_prog, "Palm hold (exit tracking / enter gesture)")
    bar(y0 + 38, pistol_prog, "Pistol hold (enter tracking)")
    bar(y0 + 76, pinch_prog, "Pinch hold (start drag)")

    sw, sh = phone_size
    cx, cy = cursor_xy
    cv2.putText(frame, f"CURSOR(phone): {cx},{cy}  screen:{sw}x{sh}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    _draw_phone_inset(frame, sw, sh, (cx, cy), active=(mode in ("TRACKING", "DRAGGING")))


def run_phone_gesture_live(camera_index: int = 0, cfg: PhoneLiveConfig = PhoneLiveConfig()):
    cfg.camera_index = camera_index

    dev = AndroidDevice.connect(adb_path=cfg.adb_path, serial=cfg.serial)
    print(f"[OK] Phone connected: {dev.serial}  size={dev.screen_w}x{dev.screen_h}")

    model, le = load_model_and_encoder()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {cfg.camera_index} konnte nicht geöffnet werden.")

    window: List[np.ndarray] = []
    pred_agg = PredictionAggregator(min_interval_s=cfg.pred_min_interval_s)
    commit_agg = PredictionAggregator(min_interval_s=cfg.pred_min_interval_s)

    mode = "IDLE"
    last_commit = "-"

    raw_label, raw_conf = "-", 0.0
    live_label, live_conf = "-", 0.0

    palm_hold_t = 0.0
    pistol_hold_t = 0.0
    pinch_hold_t = 0.0
    pinch_missing_t = 0.0

    cooldown_until = 0.0
    rec_frames: List[np.ndarray] = []
    end_counter = 0

    cursor_x = dev.screen_w // 2
    cursor_y = dev.screen_h // 2

    last_drag_x = cursor_x
    last_drag_y = cursor_y
    last_drag_send_t = 0.0

    last_loop_t = time.time()

    def classify(win12: np.ndarray) -> Tuple[str, float]:
        feat189 = window_features(win12)
        y_enc, conf = predict_feat189(model, feat189)
        lab = decode_label(y_enc, le)
        return lab, float(conf)

    tracking_allowed = {cfg.neutral_label, cfg.pistol_label, cfg.pinch_label}

    cv2.namedWindow("PHONE LIVE", cv2.WINDOW_NORMAL)

    try:
        while True:
            now = time.time()
            dt = now - last_loop_t
            last_loop_t = now

            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            lm = _extract_lm(results)
            if results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            x63 = None
            if lm is not None:
                x63 = normalize_landmarks([(float(x), float(y), float(z)) for x, y, z in lm])
                window.append(x63)
                if len(window) > cfg.window_size:
                    window.pop(0)

                idx = lm[8, :2]
                nx, ny = float(idx[0]), float(idx[1])
                tx = int(nx * dev.screen_w)
                ty = int(ny * dev.screen_h)

                a = cfg.track_smooth_alpha
                cursor_x = int((1 - a) * cursor_x + a * tx)
                cursor_y = int((1 - a) * cursor_y + a * ty)

            if x63 is not None and len(window) == cfg.window_size:
                win12 = np.asarray(window, dtype=np.float32)
                tmp_label, tmp_conf = classify(win12)
                if pred_agg.feed(tmp_label, tmp_conf, now):
                    raw_label, raw_conf = tmp_label, tmp_conf

            # Reject low confidence globally (no garbage class)
            if raw_conf < cfg.reject_below_conf:
                raw_label = "-"
                raw_conf = 0.0

            # filter while tracking/dragging
            if mode in ("TRACKING", "DRAGGING"):
                if raw_label in tracking_allowed:
                    live_label, live_conf = raw_label, raw_conf
                else:
                    live_label, live_conf = "-", 0.0
            else:
                live_label, live_conf = raw_label, raw_conf

            palm_prog = 0.0
            pistol_prog = 0.0
            pinch_prog = 0.0

            if mode == "IDLE":
                if live_label == cfg.neutral_label and live_conf >= cfg.start_min_conf:
                    palm_hold_t += dt
                else:
                    palm_hold_t = 0.0

                if live_label == cfg.pistol_label and live_conf >= cfg.start_min_conf:
                    pistol_hold_t += dt
                else:
                    pistol_hold_t = 0.0

                palm_prog = min(1.0, palm_hold_t / cfg.palm_hold_s)
                pistol_prog = min(1.0, pistol_hold_t / cfg.pistol_hold_s)

                if palm_hold_t >= cfg.palm_hold_s:
                    mode = "GESTURE_ARMED"
                    commit_agg.reset()
                    rec_frames = []
                    end_counter = 0
                    last_commit = "ARM -> GESTURE"
                    palm_hold_t = cfg.palm_hold_s
                    pistol_hold_t = 0.0
                    pinch_hold_t = 0.0
                    pinch_missing_t = 0.0

                elif pistol_hold_t >= cfg.pistol_hold_s:
                    mode = "TRACKING"
                    last_commit = "ARM -> TRACKING"
                    palm_hold_t = 0.0
                    pistol_hold_t = cfg.pistol_hold_s
                    pinch_hold_t = 0.0
                    pinch_missing_t = 0.0
                    last_drag_x, last_drag_y = cursor_x, cursor_y
                    last_drag_send_t = 0.0

            elif mode == "GESTURE_ARMED":
                palm_prog = 1.0
                if not (live_label == cfg.neutral_label and live_conf >= cfg.start_min_conf):
                    mode = "GESTURE_RECORDING"
                    rec_frames = []
                    commit_agg.reset()
                    end_counter = 0
                    last_commit = "GESTURE -> RECORDING"

            elif mode == "GESTURE_RECORDING":
                if x63 is not None:
                    rec_frames.append(x63)

                # commit only real gestures (exclude palm + pistol + "-" )
                if (
                    live_label
                    and live_label not in (cfg.neutral_label, cfg.pistol_label, "-")
                    and live_conf >= cfg.commit_frame_conf_gate
                ):
                    commit_agg.feed(live_label, live_conf, now)

                maj_label, maj_conf, maj_n = commit_agg.result()
                if maj_label is not None and maj_n >= cfg.commit_min_samples and maj_conf >= cfg.commit_min_conf:
                    key = _map_label_to_keyevent(maj_label)
                    if key is not None:
                        dev.keyevent(key)
                        last_commit = f"{maj_label} -> KEY({key})"
                    else:
                        last_commit = f"{maj_label} (no mapping)"

                    mode = "COOLDOWN"
                    cooldown_until = now + cfg.cooldown_s
                    rec_frames = []
                    commit_agg.reset()
                    palm_hold_t = 0.0
                    pistol_hold_t = 0.0
                    pinch_hold_t = 0.0
                    pinch_missing_t = 0.0

                else:
                    if live_label == cfg.neutral_label and live_conf >= cfg.start_min_conf:
                        end_counter += 1
                    else:
                        end_counter = 0

                    if len(rec_frames) >= 80 or end_counter >= 6:
                        if len(rec_frames) >= 8:
                            seg = np.asarray(rec_frames, dtype=np.float32)
                            seg12 = _resample_to_T(seg, cfg.window_size)
                            seg_label, seg_conf = classify(seg12)

                            if seg_conf < cfg.reject_below_conf:
                                seg_label = "-"

                            if seg_label not in (cfg.neutral_label, cfg.pistol_label, "-") and seg_conf >= cfg.commit_min_conf:
                                key = _map_label_to_keyevent(seg_label)
                                if key is not None:
                                    dev.keyevent(key)
                                    last_commit = f"{seg_label} -> KEY({key})"
                                else:
                                    last_commit = f"{seg_label} (no mapping)"

                        mode = "COOLDOWN"
                        cooldown_until = now + cfg.cooldown_s
                        rec_frames = []
                        commit_agg.reset()
                        palm_hold_t = 0.0
                        pistol_hold_t = 0.0
                        pinch_hold_t = 0.0
                        pinch_missing_t = 0.0

            elif mode == "TRACKING":
                pistol_prog = 1.0

                if live_label == cfg.neutral_label and live_conf >= cfg.start_min_conf:
                    palm_hold_t += dt
                else:
                    palm_hold_t = 0.0
                palm_prog = min(1.0, palm_hold_t / cfg.palm_hold_s)

                if live_label == cfg.pinch_label and live_conf >= cfg.pinch_min_conf:
                    pinch_hold_t += dt
                else:
                    pinch_hold_t = 0.0
                pinch_prog = min(1.0, pinch_hold_t / max(1e-6, cfg.pinch_hold_s))

                if pinch_hold_t >= cfg.pinch_hold_s:
                    mode = "DRAGGING"
                    last_commit = "PINCH -> DRAG START"
                    last_drag_x, last_drag_y = cursor_x, cursor_y
                    last_drag_send_t = 0.0
                    pinch_hold_t = 0.0
                    pinch_missing_t = 0.0

                if palm_hold_t >= cfg.palm_hold_s:
                    mode = "GESTURE_ARMED"
                    commit_agg.reset()
                    rec_frames = []
                    end_counter = 0
                    last_commit = "TRACKING -> GESTURE"
                    palm_hold_t = cfg.palm_hold_s
                    pinch_hold_t = 0.0
                    pinch_missing_t = 0.0

            elif mode == "DRAGGING":
                # EASY EXIT: pistol => end drag immediately
                if live_label == cfg.pistol_label and live_conf >= cfg.start_min_conf:
                    mode = "TRACKING"
                    last_commit = "PISTOL -> DRAG END -> TRACKING"
                    pinch_missing_t = 0.0
                    pinch_hold_t = 0.0
                    last_drag_x, last_drag_y = cursor_x, cursor_y
                    last_drag_send_t = 0.0
                else:
                    # pinch missing => end drag quickly
                    if not (live_label == cfg.pinch_label and live_conf >= cfg.pinch_min_conf):
                        pinch_missing_t += dt
                    else:
                        pinch_missing_t = 0.0

                    if pinch_missing_t >= cfg.pinch_release_s:
                        mode = "TRACKING"
                        last_commit = "PINCH LOST -> DRAG END -> TRACKING"
                        pinch_missing_t = 0.0
                        pinch_hold_t = 0.0
                        last_drag_x, last_drag_y = cursor_x, cursor_y
                        last_drag_send_t = 0.0

                    # drag move segments (coherent)
                    if lm is not None and mode == "DRAGGING":
                        interval = 1.0 / max(1.0, cfg.drag_send_hz)
                        if (now - last_drag_send_t) >= interval:
                            dx = abs(cursor_x - last_drag_x)
                            dy = abs(cursor_y - last_drag_y)
                            if (dx + dy) >= cfg.drag_move_min_px:
                                dev.swipe(last_drag_x, last_drag_y, cursor_x, cursor_y, duration_ms=cfg.drag_duration_ms)
                                last_drag_x, last_drag_y = cursor_x, cursor_y
                                last_drag_send_t = now

            elif mode == "COOLDOWN":
                if now >= cooldown_until:
                    mode = "IDLE"
                    palm_hold_t = 0.0
                    pistol_hold_t = 0.0
                    pinch_hold_t = 0.0
                    pinch_missing_t = 0.0
                    commit_agg.reset()
                    last_commit = "COOLDOWN -> IDLE"

            _draw_ui(
                frame=frame,
                mode=mode,
                palm_prog=min(1.0, palm_hold_t / cfg.palm_hold_s) if cfg.palm_hold_s > 0 else 0.0,
                pistol_prog=min(1.0, pistol_hold_t / cfg.pistol_hold_s) if cfg.pistol_hold_s > 0 else 0.0,
                pinch_prog=pinch_prog if mode == "TRACKING" else 0.0,
                live_label=live_label,
                live_conf=float(live_conf),
                last_commit=last_commit,
                cursor_xy=(cursor_x, cursor_y),
                phone_size=(dev.screen_w, dev.screen_h),
            )

            if lm is not None and mode in ("TRACKING", "DRAGGING"):
                idx = lm[8, :2]
                cx = int(idx[0] * frame.shape[1])
                cy = int(idx[1] * frame.shape[0])
                cv2.circle(frame, (cx, cy), 12, (0, 0, 255), -1)

            if cfg.display_scale != 1.0:
                frame_show = cv2.resize(frame, None, fx=cfg.display_scale, fy=cfg.display_scale, interpolation=cv2.INTER_LINEAR)
            else:
                frame_show = frame

            cv2.imshow("PHONE LIVE", frame_show)
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
