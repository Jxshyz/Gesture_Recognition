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


# Android keycodes (DPAD + ENTER)
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
    garbage_label: str = "garbage"

    palm_hold_s: float = 0.5
    pistol_hold_s: float = 0.5
    start_min_conf: float = 0.60

    pred_min_interval_s: float = 0.06

    commit_min_samples: int = 3
    commit_min_conf: float = 0.60
    commit_frame_conf_gate: float = 0.55

    cooldown_s: float = 0.35

    # dragging (pinch hold -> touch DOWN, release -> UP)
    drag_send_hz: float = 18.0
    drag_move_min_px: int = 4
    pinch_hold_s: float = 0.18
    pinch_min_conf: float = 0.60
    pinch_release_s: float = 0.12

    track_smooth_alpha: float = 0.25

    adb_path: Optional[str] = None
    serial: Optional[str] = None

    window_scale: float = 3.0
    phone_inset_scale: float = 0.5

    # ✅ NEW: swipe gesture as real drag across screen
    swipe_y_ratio: float = 0.50
    swipe_x_left_ratio: float = 0.18
    swipe_x_right_ratio: float = 0.82
    swipe_duration_s: float = 0.22
    swipe_steps: int = 14


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
    return np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)  # (21,3)


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
    # NOTE: swipe_left / swipe_right machen wir bewusst NICHT mehr als keyevent!
    if label == "swipe_up":
        return KEYCODE_DPAD_UP
    if label == "swipe_down":
        return KEYCODE_DPAD_DOWN
    if label in ("rotate_left", "rotate_right"):
        return KEYCODE_DPAD_UP
    if label in ("close_fist",):
        return KEYCODE_ENTER
    return None


def _draw_phone_inset(frame: np.ndarray, phone_w: int, phone_h: int, cursor_xy: Tuple[int, int], active: bool, inset_scale: float = 1.0):
    h, w = frame.shape[:2]
    base_inset_w = 170
    inset_w = max(60, int(base_inset_w * inset_scale))
    inset_h = int(inset_w * (phone_h / max(1, phone_w)))
    inset_h = max(120, min(inset_h, 260))

    x0 = w - inset_w - 12
    y0 = h - inset_h - 12
    x1 = x0 + inset_w
    y1 = y0 + inset_h

    cv2.rectangle(frame, (x0, y0), (x1, y1), (10, 10, 10), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 2)

    title = "PHONE POINTER" + (" (ACTIVE)" if active else "")
    cv2.putText(frame, title, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    cx, cy = cursor_xy
    ix = int(x0 + (cx / max(1, phone_w)) * inset_w)
    iy = int(y0 + (cy / max(1, phone_h)) * inset_h)

    cv2.circle(frame, (ix, iy), 7, (0, 0, 255), -1)
    cv2.circle(frame, (ix, iy), 10, (255, 255, 255), 2)


def _draw_ui(frame, mode: str, palm_prog: float, pistol_prog: float, pinch_prog: float,
             live_label: str, live_conf: float, last_commit: str,
             cursor_xy: Tuple[int, int], phone_size: Tuple[int, int], phone_inset_scale: float):
    h, w = frame.shape[:2]
    col = STATE_COLORS.get(mode, (200, 200, 200))

    cv2.rectangle(frame, (10, 10), (640, 195), col, -1)
    cv2.putText(frame, f"MODE: {mode}", (22, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"LIVE: {live_label} ({live_conf:.2f})", (22, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"LAST: {last_commit}", (22, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)

    bar_w = 430
    bar_h = 14
    x0 = 10
    y0 = h - 98

    def bar(y, p, title):
        cv2.putText(frame, title, (x0, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (40, 40, 40), -1)
        fill = int(bar_w * max(0.0, min(1.0, p)))
        cv2.rectangle(frame, (x0, y), (x0 + fill, y + bar_h), (230, 230, 230), -1)

    bar(y0, palm_prog, "Palm hold (exit tracking / enter gesture)")
    bar(y0 + 30, pistol_prog, "Pistol hold (enter tracking)")
    bar(y0 + 60, pinch_prog, "Pinch hold (touch DOWN), release pinch -> UP")

    sw, sh = phone_size
    cx, cy = cursor_xy
    _draw_phone_inset(frame, sw, sh, (cx, cy), active=(mode in ("TRACKING", "DRAGGING")), inset_scale=phone_inset_scale)


def run_phone_gesture_live(camera_index: int = 0, cfg: PhoneLiveConfig = PhoneLiveConfig()):
    cfg.camera_index = camera_index

    dev = AndroidDevice.connect(adb_path=cfg.adb_path, serial=cfg.serial, enable_u2=True)

    try:
        dev.set_show_touches(True)
        dev.set_pointer_location(True)
    except Exception:
        pass

    dev.refresh_display_info(force=True)
    screen_w, screen_h = dev.input_w, dev.input_h
    print(f"[OK] Phone connected: {dev.serial} input={screen_w}x{screen_h} rot={dev.rotation}")

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
    pinch_release_t = 0.0

    cooldown_until = 0.0
    rec_frames: List[np.ndarray] = []
    end_counter = 0

    cursor_x = screen_w // 2
    cursor_y = screen_h // 2

    last_drag_x = cursor_x
    last_drag_y = cursor_y
    last_drag_send_t = 0.0

    touch_is_down = False

    last_loop_t = time.time()

    def clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    def classify(win12: np.ndarray) -> Tuple[str, float]:
        feat189 = window_features(win12)
        y_enc, conf = predict_feat189(model, feat189)
        lab = decode_label(y_enc, le)
        return lab, float(conf)

    # ✅ NEW: swipe across phone screen using real drag
    def do_screen_swipe(direction: str) -> str:
        dev.refresh_display_info(force=True)
        sw, sh = dev.input_w, dev.input_h
        y = int(sh * cfg.swipe_y_ratio)
        xL = int(sw * cfg.swipe_x_left_ratio)
        xR = int(sw * cfg.swipe_x_right_ratio)

        y = clamp(y, 0, max(0, sh - 1))
        xL = clamp(xL, 0, max(0, sw - 1))
        xR = clamp(xR, 0, max(0, sw - 1))

        if direction == "left":
            # finger from RIGHT to LEFT
            dev.drag(xR, y, xL, y, duration_s=cfg.swipe_duration_s, steps=cfg.swipe_steps)
            return f"swipe_left -> DRAG({xR},{y})->({xL},{y})"
        else:
            # finger from LEFT to RIGHT
            dev.drag(xL, y, xR, y, duration_s=cfg.swipe_duration_s, steps=cfg.swipe_steps)
            return f"swipe_right -> DRAG({xL},{y})->({xR},{y})"

    tracking_allowed = {cfg.neutral_label, cfg.pistol_label, cfg.pinch_label}

    win_name = "PHONE LIVE (cohesive touch + swipe drag)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    resized_once = False

    try:
        while True:
            now = time.time()
            dt = now - last_loop_t
            last_loop_t = now

            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)

            if not resized_once:
                resized_once = True
                fw = int(frame.shape[1] * cfg.window_scale)
                fh = int(frame.shape[0] * cfg.window_scale)
                cv2.resizeWindow(win_name, fw, fh)

            dev.refresh_display_info()
            screen_w, screen_h = dev.input_w, dev.input_h

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
                tx = int(nx * screen_w)
                ty = int(ny * screen_h)

                a = cfg.track_smooth_alpha
                cursor_x = int((1 - a) * cursor_x + a * tx)
                cursor_y = int((1 - a) * cursor_y + a * ty)

                cursor_x = clamp(cursor_x, 0, max(0, screen_w - 1))
                cursor_y = clamp(cursor_y, 0, max(0, screen_h - 1))
            else:
                raw_label, raw_conf = "-", 0.0

            if x63 is not None and len(window) == cfg.window_size:
                win12 = np.asarray(window, dtype=np.float32)
                tmp_label, tmp_conf = classify(win12)
                if pred_agg.feed(tmp_label, tmp_conf, now):
                    raw_label, raw_conf = tmp_label, tmp_conf

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

            pinch_on = (live_label == cfg.pinch_label and live_conf >= cfg.pinch_min_conf)

            if mode == "IDLE":
                if live_label == cfg.neutral_label and live_conf >= cfg.start_min_conf:
                    palm_hold_t += dt
                else:
                    palm_hold_t = 0.0

                if live_label == cfg.pistol_label and live_conf >= cfg.start_min_conf:
                    pistol_hold_t += dt
                else:
                    pistol_hold_t = 0.0

                palm_prog = min(1.0, palm_hold_t / cfg.palm_hold_s) if cfg.palm_hold_s > 0 else 0.0
                pistol_prog = min(1.0, pistol_hold_t / cfg.pistol_hold_s) if cfg.pistol_hold_s > 0 else 0.0

                if palm_hold_t >= cfg.palm_hold_s:
                    mode = "GESTURE_ARMED"
                    commit_agg.reset()
                    rec_frames = []
                    end_counter = 0
                    last_commit = "ARM -> GESTURE"
                    palm_hold_t = cfg.palm_hold_s
                    pistol_hold_t = 0.0
                    pinch_hold_t = 0.0
                    pinch_release_t = 0.0

                elif pistol_hold_t >= cfg.pistol_hold_s:
                    mode = "TRACKING"
                    last_commit = "ARM -> TRACKING"
                    palm_hold_t = 0.0
                    pistol_hold_t = cfg.pistol_hold_s
                    pinch_hold_t = 0.0
                    pinch_release_t = 0.0
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

                if (
                    live_label
                    and live_label not in (cfg.neutral_label, cfg.pistol_label, cfg.garbage_label)
                    and live_conf >= cfg.commit_frame_conf_gate
                ):
                    commit_agg.feed(live_label, live_conf, now)

                maj_label, maj_conf, maj_n = commit_agg.result()
                if maj_label is not None and maj_n >= cfg.commit_min_samples and maj_conf >= cfg.commit_min_conf:
                    # ✅ HERE: swipe_left/right = drag across screen
                    if maj_label == "swipe_left":
                        last_commit = do_screen_swipe("left")
                    elif maj_label == "swipe_right":
                        last_commit = do_screen_swipe("right")
                    else:
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
                    pinch_release_t = 0.0

                else:
                    if live_label == cfg.neutral_label and live_conf >= cfg.start_min_conf:
                        end_counter += 1
                    else:
                        end_counter = 0

                    if len(rec_frames) >= 80 or end_counter >= 6:
                        # fallback classify
                        if len(rec_frames) >= 8:
                            seg = np.asarray(rec_frames, dtype=np.float32)
                            seg12 = _resample_to_T(seg, cfg.window_size)
                            seg_label, seg_conf = classify(seg12)

                            if seg_label not in (cfg.neutral_label, cfg.pistol_label, cfg.garbage_label) and seg_conf >= cfg.commit_min_conf:
                                if seg_label == "swipe_left":
                                    last_commit = do_screen_swipe("left")
                                elif seg_label == "swipe_right":
                                    last_commit = do_screen_swipe("right")
                                else:
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
                        pinch_release_t = 0.0

            elif mode == "TRACKING":
                pistol_prog = 1.0

                if live_label == cfg.neutral_label and live_conf >= cfg.start_min_conf:
                    palm_hold_t += dt
                else:
                    palm_hold_t = 0.0
                palm_prog = min(1.0, palm_hold_t / cfg.palm_hold_s) if cfg.palm_hold_s > 0 else 0.0

                if pinch_on:
                    pinch_hold_t += dt
                else:
                    pinch_hold_t = 0.0
                pinch_prog = min(1.0, pinch_hold_t / max(1e-6, cfg.pinch_hold_s))

                if pinch_hold_t >= cfg.pinch_hold_s:
                    mode = "DRAGGING"
                    last_commit = "PINCH -> TOUCH DOWN"
                    last_drag_x, last_drag_y = cursor_x, cursor_y
                    last_drag_send_t = 0.0
                    pinch_release_t = 0.0

                    dev.touch_down(cursor_x, cursor_y)
                    touch_is_down = True
                    pinch_hold_t = 0.0

                if palm_hold_t >= cfg.palm_hold_s:
                    mode = "GESTURE_ARMED"
                    commit_agg.reset()
                    rec_frames = []
                    end_counter = 0
                    last_commit = "TRACKING -> GESTURE"
                    palm_hold_t = cfg.palm_hold_s
                    pinch_hold_t = 0.0
                    pinch_release_t = 0.0

            elif mode == "DRAGGING":
                pistol_prog = 1.0

                if pinch_on:
                    pinch_release_t = 0.0
                else:
                    pinch_release_t += dt

                pinch_prog = 1.0 if pinch_on else max(0.0, 1.0 - (pinch_release_t / max(1e-6, cfg.pinch_release_s)))

                if touch_is_down and lm is not None:
                    interval = 1.0 / max(1.0, cfg.drag_send_hz)
                    if (now - last_drag_send_t) >= interval:
                        dx = abs(cursor_x - last_drag_x)
                        dy = abs(cursor_y - last_drag_y)
                        if (dx + dy) >= cfg.drag_move_min_px:
                            dev.touch_move(cursor_x, cursor_y)
                            last_drag_x, last_drag_y = cursor_x, cursor_y
                            last_drag_send_t = now

                if pinch_release_t >= cfg.pinch_release_s:
                    if touch_is_down:
                        dev.touch_up(cursor_x, cursor_y)
                        touch_is_down = False
                    mode = "TRACKING"
                    last_commit = "PINCH RELEASE -> TOUCH UP -> TRACKING"
                    pinch_release_t = 0.0
                    pinch_hold_t = 0.0
                    last_drag_x, last_drag_y = cursor_x, cursor_y
                    last_drag_send_t = 0.0

                if live_label == cfg.neutral_label and live_conf >= cfg.start_min_conf:
                    palm_hold_t += dt
                else:
                    palm_hold_t = 0.0
                palm_prog = min(1.0, palm_hold_t / cfg.palm_hold_s) if cfg.palm_hold_s > 0 else 0.0

                if palm_hold_t >= cfg.palm_hold_s:
                    if touch_is_down:
                        dev.touch_up(cursor_x, cursor_y)
                        touch_is_down = False
                    mode = "GESTURE_ARMED"
                    commit_agg.reset()
                    rec_frames = []
                    end_counter = 0
                    last_commit = "DRAGGING -> (UP) -> GESTURE"
                    palm_hold_t = cfg.palm_hold_s
                    pinch_release_t = 0.0
                    pinch_hold_t = 0.0

            elif mode == "COOLDOWN":
                if now >= cooldown_until:
                    mode = "IDLE"
                    palm_hold_t = 0.0
                    pistol_hold_t = 0.0
                    pinch_hold_t = 0.0
                    pinch_release_t = 0.0
                    commit_agg.reset()
                    last_commit = "COOLDOWN -> IDLE"

            _draw_ui(
                frame=frame,
                mode=mode,
                palm_prog=min(1.0, palm_hold_t / cfg.palm_hold_s) if cfg.palm_hold_s > 0 else 0.0,
                pistol_prog=min(1.0, pistol_hold_t / cfg.pistol_hold_s) if cfg.pistol_hold_s > 0 else 0.0,
                pinch_prog=pinch_prog if mode in ("TRACKING", "DRAGGING") else 0.0,
                live_label=live_label,
                live_conf=float(live_conf),
                last_commit=last_commit,
                cursor_xy=(cursor_x, cursor_y),
                phone_size=(screen_w, screen_h),
                phone_inset_scale=cfg.phone_inset_scale,
            )

            if lm is not None and mode in ("TRACKING", "DRAGGING"):
                idx = lm[8, :2]
                cx = int(idx[0] * frame.shape[1])
                cy = int(idx[1] * frame.shape[0])
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        try:
            if touch_is_down:
                dev.touch_up(cursor_x, cursor_y)
        except Exception:
            pass

        cap.release()
        cv2.destroyAllWindows()
        try:
            hands.close()
        except Exception:
            pass
