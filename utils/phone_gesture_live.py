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
KEYCODE_DPAD_UP = 19
KEYCODE_DPAD_DOWN = 20
KEYCODE_ENTER = 66


@dataclass
class PhoneLiveConfig:
    camera_index: int = 0

    # must match training
    window_size: int = 12

    # labels
    neutral_label: str = "neutral_palm"
    pistol_label: str = "finger_pistol"
    pinch_label: str = "pinch"
    garbage_label: str = "garbage"  # exists, but we ignore it (no trigger)

    # start holds (arming)
    palm_hold_s: float = 0.5
    pistol_hold_s: float = 0.5
    start_min_conf: float = 0.60

    # prediction rate
    pred_min_interval_s: float = 0.06  # ~16.6/s

    # gesture mode commit stability
    commit_min_samples: int = 3
    commit_min_conf: float = 0.60
    commit_frame_conf_gate: float = 0.55

    # cooldown after gesture commit
    cooldown_s: float = 0.35

    # dragging (pinch)
    drag_send_hz: float = 18.0
    drag_move_min_px: int = 4
    drag_duration_ms: int = 90

    pinch_hold_s: float = 0.18
    pinch_min_conf: float = 0.60
    pinch_release_s: float = 0.12

    # smoothing cursor on phone coords
    track_smooth_alpha: float = 0.25

    # adb
    adb_path: Optional[str] = None
    serial: Optional[str] = None

    # UI sizing
    window_scale: float = 1.5        # Fenster kleiner
    phone_inset_scale: float = 0.5   # Handy-Inlay halb so groß


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
    """(N,63)->(T,63)"""
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
    # swipe_left/right machen wir absichtlich NICHT als keyevent (drag/screen swipe)
    if label == "swipe_up":
        return KEYCODE_DPAD_UP
    if label == "swipe_down":
        return KEYCODE_DPAD_DOWN
    if label in ("rotate_left", "rotate_right"):
        return KEYCODE_DPAD_UP
    if label in ("close_fist",):
        return KEYCODE_ENTER
    return None


def _draw_phone_inset(
    frame: np.ndarray,
    phone_w: int,
    phone_h: int,
    cursor_xy: Tuple[int, int],
    active: bool,
    inset_scale: float = 1.0,
):
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


def _draw_arming_bar(frame: np.ndarray, x: int, y: int, w: int, h: int, prog: float, ready: bool):
    prog = max(0.0, min(1.0, float(prog)))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (35, 35, 35), -1)

    fill_w = int(w * prog)
    # ✅ wird grün wenn READY, sonst hellgrau
    fill_col = (0, 255, 0) if ready else (220, 220, 220)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), fill_col, -1)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)


def _draw_ui(
    frame: np.ndarray,
    status: str,                 # IDLE / ARMED / READY
    mode: str,                   # intern (TRACKING/DRAGGING/...)
    palm_prog: float,            # 0..1
    last_sent_gesture: str,
    cursor_xy: Tuple[int, int],
    phone_size: Tuple[int, int],
    phone_inset_scale: float,
):
    h, w = frame.shape[:2]

    overlay = frame.copy()

    # Top Panel Hintergrund auf Overlay
    col = STATE_COLORS.get(mode, (200, 200, 200))
    cv2.rectangle(overlay, (10, 10), (370, 120), col, -1)

    # Bottom-left Panel Hintergrund auf Overlay
    cv2.rectangle(overlay, (10, h - 55), (250, h - 10), (200, 200, 200), -1)

    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    
    # STATUS Text
    cv2.putText(frame, f"STATUS: {status}", (22, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # MODE Text
    cv2.putText(frame, f"MODE: {mode}", (22, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)

    # Arming bar (palm)
    ready = (status == "READY")
    bar_x, bar_y = 22, 100
    bar_w, bar_h = 200, 8
    
    _draw_arming_bar(frame, bar_x, bar_y, bar_w, bar_h, palm_prog, ready)
    
    cv2.putText(frame, f"ARM (palm): {int(palm_prog*100)}%", (bar_x + bar_w + 12, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)

    # Bottom-left: LAST Gesture Text
    cv2.putText(frame, f"LAST: {last_sent_gesture}", (22, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)

    # Phone inset rechts unten
    #sw, sh = phone_size
    #cx, cy = cursor_xy
    #_draw_phone_inset(frame, sw, sh, (cx, cy), active=(mode in ("TRACKING", "DRAGGING")), inset_scale=phone_inset_scale)


def run_phone_gesture_live(camera_index: int = 0, cfg: PhoneLiveConfig = PhoneLiveConfig()):
    cfg.camera_index = camera_index

    # robust connect: falls AndroidDevice.connect KEIN enable_u2 kennt, fallback
    try:
        dev = AndroidDevice.connect(adb_path=cfg.adb_path, serial=cfg.serial, enable_u2=True)
    except TypeError:
        dev = AndroidDevice.connect(adb_path=cfg.adb_path, serial=cfg.serial)

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
    last_sent_gesture = "-"

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

    # --- Touch helpers (kohärent wenn dev touch_* kann) ---
    def touch_down(x: int, y: int) -> bool:
        if hasattr(dev, "touch_down"):
            dev.touch_down(x, y)
            return True
        if hasattr(dev, "swipe"):
            dev.swipe(x, y, x, y, duration_ms=80)
        return False

    def touch_move(x: int, y: int) -> None:
        if hasattr(dev, "touch_move"):
            dev.touch_move(x, y)

    def touch_up(x: int, y: int) -> None:
        if hasattr(dev, "touch_up"):
            dev.touch_up(x, y)

    # ✅ Swipe über Bildschirm als Drag (rechts<->links)
    def do_screen_swipe(direction: str) -> None:
        dev.refresh_display_info(force=True)
        sw, sh = dev.input_w, dev.input_h
        y = int(sh * 0.50)
        xL = int(sw * 0.18)
        xR = int(sw * 0.82)

        y = clamp(y, 0, max(0, sh - 1))
        xL = clamp(xL, 0, max(0, sw - 1))
        xR = clamp(xR, 0, max(0, sw - 1))

        if hasattr(dev, "drag"):
            if direction == "left":
                dev.drag(xR, y, xL, y, duration_s=0.18, steps=10)
            else:
                dev.drag(xL, y, xR, y, duration_s=0.18, steps=10)
            return

        if hasattr(dev, "swipe"):
            dur = 180
            if direction == "left":
                dev.swipe(xR, y, xL, y, duration_ms=dur)
            else:
                dev.swipe(xL, y, xR, y, duration_ms=dur)

    tracking_allowed = {cfg.neutral_label, cfg.pistol_label, cfg.pinch_label}

    win_name = "PHONE LIVE"
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

            # Fenster skalieren
            if not resized_once:
                resized_once = True
                fw = int(frame.shape[1] * cfg.window_scale)
                fh = int(frame.shape[0] * cfg.window_scale)
                cv2.resizeWindow(win_name, fw, fh)

            dev.refresh_display_info(force=False, min_interval_s=2.0)
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

                # cursor from index tip
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

            # LIVE prediction
            if x63 is not None and len(window) == cfg.window_size:
                win12 = np.asarray(window, dtype=np.float32)
                tmp_label, tmp_conf = classify(win12)
                if pred_agg.feed(tmp_label, tmp_conf, now):
                    raw_label, raw_conf = tmp_label, tmp_conf

            # FILTER while tracking/dragging
            if mode in ("TRACKING", "DRAGGING"):
                if raw_label in tracking_allowed:
                    live_label, live_conf = raw_label, raw_conf
                else:
                    live_label, live_conf = "-", 0.0
            else:
                live_label, live_conf = raw_label, raw_conf

            pinch_on = (live_label == cfg.pinch_label and live_conf >= cfg.pinch_min_conf)

            # -----------------------------
            # FSM
            # -----------------------------
            if mode == "IDLE":
                # palm arming
                if live_label == cfg.neutral_label and live_conf >= cfg.start_min_conf:
                    palm_hold_t += dt
                else:
                    palm_hold_t = 0.0

                # optional pistol arming (tracking)
                if live_label == cfg.pistol_label and live_conf >= cfg.start_min_conf:
                    pistol_hold_t += dt
                else:
                    pistol_hold_t = 0.0

                if palm_hold_t >= cfg.palm_hold_s:
                    mode = "GESTURE_ARMED"
                    commit_agg.reset()
                    rec_frames = []
                    end_counter = 0
                    palm_hold_t = cfg.palm_hold_s
                    pistol_hold_t = 0.0
                    pinch_hold_t = 0.0
                    pinch_release_t = 0.0

                elif pistol_hold_t >= cfg.pistol_hold_s:
                    mode = "TRACKING"
                    palm_hold_t = 0.0
                    pistol_hold_t = cfg.pistol_hold_s
                    pinch_hold_t = 0.0
                    pinch_release_t = 0.0
                    last_drag_x, last_drag_y = cursor_x, cursor_y
                    last_drag_send_t = 0.0

            elif mode == "GESTURE_ARMED":
                # READY state, wartet auf "neutral verlassen"
                palm_hold_t = cfg.palm_hold_s
                if not (live_label == cfg.neutral_label and live_conf >= cfg.start_min_conf):
                    mode = "GESTURE_RECORDING"
                    rec_frames = []
                    commit_agg.reset()
                    end_counter = 0
                    palm_hold_t = 0.0

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
                    if maj_label == "swipe_left":
                        do_screen_swipe("left")
                        last_sent_gesture = "swipe_left"
                    elif maj_label == "swipe_right":
                        do_screen_swipe("right")
                        last_sent_gesture = "swipe_right"
                    else:
                        key = _map_label_to_keyevent(maj_label)
                        if key is not None:
                            dev.keyevent(key)
                            last_sent_gesture = maj_label
                        else:
                            last_sent_gesture = f"ignored:{maj_label}"

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
                        if len(rec_frames) >= 8:
                            seg = np.asarray(rec_frames, dtype=np.float32)
                            seg12 = _resample_to_T(seg, cfg.window_size)
                            seg_label, seg_conf = classify(seg12)

                            if seg_label not in (cfg.neutral_label, cfg.pistol_label, cfg.garbage_label) and seg_conf >= cfg.commit_min_conf:
                                if seg_label == "swipe_left":
                                    do_screen_swipe("left")
                                    last_sent_gesture = "swipe_left"
                                elif seg_label == "swipe_right":
                                    do_screen_swipe("right")
                                    last_sent_gesture = "swipe_right"
                                else:
                                    key = _map_label_to_keyevent(seg_label)
                                    if key is not None:
                                        dev.keyevent(key)
                                        last_sent_gesture = seg_label
                                    else:
                                        last_sent_gesture = f"ignored:{seg_label}"

                        mode = "COOLDOWN"
                        cooldown_until = now + cfg.cooldown_s
                        rec_frames = []
                        commit_agg.reset()
                        palm_hold_t = 0.0
                        pistol_hold_t = 0.0
                        pinch_hold_t = 0.0
                        pinch_release_t = 0.0

            elif mode == "TRACKING":
                # pinch latch into dragging
                if pinch_on:
                    pinch_hold_t += dt
                else:
                    pinch_hold_t = 0.0

                if pinch_hold_t >= cfg.pinch_hold_s:
                    mode = "DRAGGING"
                    last_drag_x, last_drag_y = cursor_x, cursor_y
                    last_drag_send_t = 0.0
                    pinch_release_t = 0.0

                    down_ok = touch_down(cursor_x, cursor_y)
                    touch_is_down = down_ok
                    last_sent_gesture = "pinch_down" if down_ok else "pinch_down(fallback)"
                    pinch_hold_t = 0.0

                # palm arming while tracking -> back to gesture
                if live_label == cfg.neutral_label and live_conf >= cfg.start_min_conf:
                    palm_hold_t += dt
                else:
                    palm_hold_t = 0.0

                if palm_hold_t >= cfg.palm_hold_s:
                    mode = "GESTURE_ARMED"
                    commit_agg.reset()
                    rec_frames = []
                    end_counter = 0
                    palm_hold_t = cfg.palm_hold_s
                    pinch_hold_t = 0.0
                    pinch_release_t = 0.0

            elif mode == "DRAGGING":
                # release if pinch off for pinch_release_s
                if pinch_on:
                    pinch_release_t = 0.0
                else:
                    pinch_release_t += dt

                if touch_is_down and lm is not None:
                    interval = 1.0 / max(1.0, cfg.drag_send_hz)
                    if (now - last_drag_send_t) >= interval:
                        dx = abs(cursor_x - last_drag_x)
                        dy = abs(cursor_y - last_drag_y)
                        if (dx + dy) >= cfg.drag_move_min_px:
                            touch_move(cursor_x, cursor_y)
                            last_drag_x, last_drag_y = cursor_x, cursor_y
                            last_drag_send_t = now

                if pinch_release_t >= cfg.pinch_release_s:
                    if touch_is_down:
                        touch_up(cursor_x, cursor_y)
                        touch_is_down = False
                        last_sent_gesture = "pinch_up"
                    else:
                        last_sent_gesture = "pinch_up(fallback)"

                    mode = "TRACKING"
                    pinch_release_t = 0.0
                    pinch_hold_t = 0.0
                    last_drag_x, last_drag_y = cursor_x, cursor_y
                    last_drag_send_t = 0.0

                # palm arming while dragging -> gesture mode (and release touch)
                if live_label == cfg.neutral_label and live_conf >= cfg.start_min_conf:
                    palm_hold_t += dt
                else:
                    palm_hold_t = 0.0

                if palm_hold_t >= cfg.palm_hold_s:
                    if touch_is_down:
                        touch_up(cursor_x, cursor_y)
                        touch_is_down = False
                        last_sent_gesture = "pinch_up"
                    mode = "GESTURE_ARMED"
                    commit_agg.reset()
                    rec_frames = []
                    end_counter = 0
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

            # -----------------------------
            # STATUS + Palm progress (IDLE/ARMED/READY)
            # -----------------------------
            if cfg.palm_hold_s > 0:
                palm_prog = max(0.0, min(1.0, palm_hold_t / cfg.palm_hold_s))
            else:
                palm_prog = 0.0

            if mode == "GESTURE_ARMED":
                status = "READY"
                palm_prog = 1.0
            elif palm_prog > 0.0:
                status = "ARMED"
            else:
                status = "IDLE"

            # -----------------------------
            # UI
            # -----------------------------
            _draw_ui(
                frame=frame,
                status=status,
                mode=mode,
                palm_prog=palm_prog,
                last_sent_gesture=last_sent_gesture,
                cursor_xy=(cursor_x, cursor_y),
                phone_size=(screen_w, screen_h),
                phone_inset_scale=cfg.phone_inset_scale,
            )

            # red dot on camera frame while tracking/dragging
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
                touch_up(cursor_x, cursor_y)
        except Exception:
            pass

        cap.release()
        cv2.destroyAllWindows()
        try:
            hands.close()
        except Exception:
            pass
