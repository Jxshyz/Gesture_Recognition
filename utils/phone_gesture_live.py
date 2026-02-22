# utils/phone_gesture_live.py
from __future__ import annotations

# ------------------------------------------------------------
# QUIET LOGS (MUST be before importing mediapipe / tf stuff)
# ------------------------------------------------------------
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF
os.environ.setdefault("GLOG_minloglevel", "2")  # quiet glog (mediapipe)
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")  # quiet absl

import time
import traceback
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import socket

# filter noisy python warnings (protobuf)
warnings.filterwarnings("ignore", message=".*GetPrototype\\(\\) is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import mediapipe as mp

from utils.feature_extractor import normalize_landmarks, window_features
from utils.model_io import load_model_and_encoder, predict_feat189, decode_label
from utils.prediction_utils import PredictionAggregator
from utils.phone_controller import AndroidDevice


# Android keycodes
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

    drag_send_hz: float = 18.0
    drag_move_min_px: int = 4
    drag_duration_ms: int = 90

    pinch_hold_s: float = 0.5
    pinch_min_conf: float = 0.60
    pinch_release_s: float = 0.12

    track_smooth_alpha: float = 0.25

    # which mediapipe landmark to use for cursor tracking
    # 8 = index fingertip (old)
    # 5 = index MCP "knuckle" at the hand (recommended)
    # 6 = index PIP (middle joint)
    track_landmark_idx: int = 5

    adb_path: Optional[str] = None
    serial: Optional[str] = None

    window_scale: float = 1.5

    overlay_udp_host: str = "127.0.0.1"
    overlay_udp_port: int = 5005
    overlay_send_hz: float = 30.0


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
    if label == "swipe_up":
        return KEYCODE_DPAD_UP
    if label == "swipe_down":
        return KEYCODE_DPAD_DOWN
    if label in ("rotate_left", "rotate_right"):
        return KEYCODE_DPAD_UP
    if label in ("close_fist",):
        return KEYCODE_ENTER
    return None


def _draw_arming_bar(
    frame: np.ndarray, x: int, y: int, w: int, h: int, prog: float, ready: bool
):
    prog = max(0.0, min(1.0, float(prog)))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (35, 35, 35), -1)
    fill_w = int(w * prog)
    fill_col = (0, 255, 0) if ready else (220, 220, 220)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), fill_col, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)


def _draw_ui(
    frame: np.ndarray, status: str, mode: str, palm_prog: float, last_sent_gesture: str
):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    col = STATE_COLORS.get(mode, (200, 200, 200))
    cv2.rectangle(overlay, (10, 10), (620, 150), col, -1)
    cv2.rectangle(overlay, (10, h - 55), (620, h - 10), (200, 200, 200), -1)

    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (22, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"MODE: {mode}",
        (22, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    ready = status == "READY"
    bar_x, bar_y = 22, 115
    bar_w, bar_h = 260, 12
    _draw_arming_bar(frame, bar_x, bar_y, bar_w, bar_h, palm_prog, ready)
    cv2.putText(
        frame,
        f"ARM(palm): {int(palm_prog*100)}%",
        (bar_x + bar_w + 12, bar_y + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"LAST: {last_sent_gesture}",
        (22, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def _safe_lm_xy(lm: np.ndarray, idx: int) -> Tuple[float, float]:
    if lm is None or lm.ndim != 2 or lm.shape[0] == 0:
        return 0.0, 0.0
    i = int(np.clip(int(idx), 0, lm.shape[0] - 1))
    return float(lm[i, 0]), float(lm[i, 1])


def run_phone_gesture_live(
    camera_index: int = 0, cfg: PhoneLiveConfig = PhoneLiveConfig()
):
    cfg.camera_index = camera_index

    # Crash log file (so you always see the reason even if window closes)
    crash_log_path = os.path.join(os.getcwd(), "phone_live_crash.log")

    def log(msg: str):
        print(msg, flush=True)

    def log_exc(prefix: str):
        tb = traceback.format_exc()
        log(prefix)
        log(tb)
        try:
            with open(crash_log_path, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(prefix + "\n")
                f.write(tb + "\n")
        except Exception:
            pass

    # connect
    try:
        dev = AndroidDevice.connect(
            adb_path=cfg.adb_path, serial=cfg.serial, enable_u2=True
        )
    except TypeError:
        dev = AndroidDevice.connect(adb_path=cfg.adb_path, serial=cfg.serial)

    # initial display info (guarded)
    try:
        dev.refresh_display_info(force=True)
    except Exception:
        log_exc("[ADB] refresh_display_info(force=True) failed at startup!")
        # fallback to sane defaults
        if (
            hasattr(dev, "input_w")
            and hasattr(dev, "input_h")
            and dev.input_w > 0
            and dev.input_h > 0
        ):
            pass

    screen_w, screen_h = getattr(dev, "input_w", 1080), getattr(dev, "input_h", 1920)
    log(
        f"[OK] Phone connected: {dev.serial} input={screen_w}x{screen_h} rot={getattr(dev, 'rotation', 0)}"
    )
    log(f"[LOG] Crash log: {crash_log_path}")
    log(
        f"[CFG] tracking landmark idx={cfg.track_landmark_idx} (5=index MCP knuckle, 8=index tip)"
    )

    model, le = load_model_and_encoder()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {cfg.camera_index} konnte nicht geöffnet werden.")

    # UDP sender (overlay)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_addr = (cfg.overlay_udp_host, int(cfg.overlay_udp_port))
    last_udp_send_t = 0.0

    window: List[np.ndarray] = []
    pred_agg = PredictionAggregator(min_interval_s=cfg.pred_min_interval_s)
    commit_agg = PredictionAggregator(min_interval_s=cfg.pred_min_interval_s)

    mode = "IDLE"
    last_mode = mode
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

    # debug throttle
    last_debug_t = 0.0

    def clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    def classify(win12: np.ndarray) -> Tuple[str, float]:
        feat189 = window_features(win12)
        y_enc, conf = predict_feat189(model, feat189)
        lab = decode_label(y_enc, le)
        return lab, float(conf)

    # touch helpers
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

    # swipe across screen (drag)
    def do_screen_swipe(direction: str) -> None:
        # guarded refresh
        try:
            dev.refresh_display_info(force=True)
        except Exception:
            log_exc(
                "[ADB] refresh_display_info(force=True) failed inside do_screen_swipe()"
            )
            return

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
            try:
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

                # ADB refresh guarded (THIS is a common crash source)
                try:
                    dev.refresh_display_info(force=False, min_interval_s=2.0)
                    screen_w, screen_h = dev.input_w, dev.input_h
                except Exception:
                    log_exc("[ADB] refresh_display_info(force=False) failed in loop!")
                    # keep last known sizes instead of crashing

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                lm = _extract_lm(results)
                if results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                    )

                x63 = None
                track_nx = None
                track_ny = None

                if lm is not None:
                    x63 = normalize_landmarks(
                        [(float(x), float(y), float(z)) for x, y, z in lm]
                    )
                    window.append(x63)
                    if len(window) > cfg.window_size:
                        window.pop(0)

                    # Tracking point: use knuckle (cfg.track_landmark_idx, default 5)
                    nx, ny = _safe_lm_xy(lm, cfg.track_landmark_idx)
                    track_nx, track_ny = nx, ny

                    tx = int(nx * screen_w)
                    ty = int(ny * screen_h)

                    a = cfg.track_smooth_alpha
                    cursor_x = int((1 - a) * cursor_x + a * tx)
                    cursor_y = int((1 - a) * cursor_y + a * ty)
                    cursor_x = clamp(cursor_x, 0, max(0, screen_w - 1))
                    cursor_y = clamp(cursor_y, 0, max(0, screen_h - 1))
                else:
                    raw_label, raw_conf = "-", 0.0

                # prediction
                if x63 is not None and len(window) == cfg.window_size:
                    win12 = np.asarray(window, dtype=np.float32)
                    tmp_label, tmp_conf = classify(win12)
                    if pred_agg.feed(tmp_label, tmp_conf, now):
                        raw_label, raw_conf = tmp_label, tmp_conf

                # filter in tracking/dragging
                if mode in ("TRACKING", "DRAGGING"):
                    if raw_label in tracking_allowed:
                        live_label, live_conf = raw_label, raw_conf
                    else:
                        live_label, live_conf = "-", 0.0
                else:
                    live_label, live_conf = raw_label, raw_conf

                pinch_on = (
                    live_label == cfg.pinch_label and live_conf >= cfg.pinch_min_conf
                )

                # DEBUG: mode changes
                if mode != last_mode:
                    log(
                        f"[FSM] {last_mode} -> {mode} (live={live_label} conf={live_conf:.2f})"
                    )
                    last_mode = mode

                # UDP send to overlay (in tracking/dragging)
                if mode in ("TRACKING", "DRAGGING"):
                    hz = max(1.0, float(cfg.overlay_send_hz))
                    if (now - last_udp_send_t) >= (1.0 / hz):
                        last_udp_send_t = now
                        try:
                            px = cursor_x / max(1.0, float(screen_w))
                            py = cursor_y / max(1.0, float(screen_h))
                            udp_sock.sendto(
                                f"{px:.6f} {py:.6f}".encode("utf-8"), udp_addr
                            )
                        except Exception:
                            log_exc("[UDP] send failed")

                # periodic debug line (once/sec)
                if now - last_debug_t >= 1.0:
                    last_debug_t = now
                    log(
                        f"[DBG] mode={mode} live={live_label} {live_conf:.2f} cursor={cursor_x},{cursor_y} screen={screen_w}x{screen_h}"
                    )

                # ---------------- FSM ----------------
                if mode == "IDLE":
                    if (
                        live_label == cfg.neutral_label
                        and live_conf >= cfg.start_min_conf
                    ):
                        palm_hold_t += dt
                    else:
                        palm_hold_t = 0.0

                    if (
                        live_label == cfg.pistol_label
                        and live_conf >= cfg.start_min_conf
                    ):
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
                    palm_hold_t = cfg.palm_hold_s
                    if not (
                        live_label == cfg.neutral_label
                        and live_conf >= cfg.start_min_conf
                    ):
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
                        and live_label
                        not in (cfg.neutral_label, cfg.pistol_label, cfg.garbage_label)
                        and live_conf >= cfg.commit_frame_conf_gate
                    ):
                        commit_agg.feed(live_label, live_conf, now)

                    maj_label, maj_conf, maj_n = commit_agg.result()
                    if (
                        maj_label is not None
                        and maj_n >= cfg.commit_min_samples
                        and maj_conf >= cfg.commit_min_conf
                    ):
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
                        if (
                            live_label == cfg.neutral_label
                            and live_conf >= cfg.start_min_conf
                        ):
                            end_counter += 1
                        else:
                            end_counter = 0

                        if len(rec_frames) >= 80 or end_counter >= 6:
                            if len(rec_frames) >= 8:
                                seg = np.asarray(rec_frames, dtype=np.float32)
                                seg12 = _resample_to_T(seg, cfg.window_size)
                                seg_label, seg_conf = classify(seg12)

                                if (
                                    seg_label
                                    not in (
                                        cfg.neutral_label,
                                        cfg.pistol_label,
                                        cfg.garbage_label,
                                    )
                                    and seg_conf >= cfg.commit_min_conf
                                ):
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
                    # -----------------------------
                    # PINCH = CLICK (HOLD 0.5s)
                    # -----------------------------
                    if pinch_on:
                        pinch_hold_t += dt
                    else:
                        pinch_hold_t = 0.0

                    # CLICK DOWN after hold
                    if pinch_hold_t >= cfg.pinch_hold_s and not touch_is_down:
                        down_ok = touch_down(cursor_x, cursor_y)
                        touch_is_down = down_ok
                        last_sent_gesture = "click_down"
                        pinch_hold_t = cfg.pinch_hold_s  # clamp

                    # CLICK UP after release
                    if touch_is_down and not pinch_on:
                        touch_up(cursor_x, cursor_y)
                        touch_is_down = False
                        pinch_hold_t = 0.0
                        last_sent_gesture = "click_up"

                    # Palm → back to Gesture Mode
                    if (
                        live_label == cfg.neutral_label
                        and live_conf >= cfg.start_min_conf
                    ):
                        palm_hold_t += dt
                    else:
                        palm_hold_t = 0.0

                    if palm_hold_t >= cfg.palm_hold_s:
                        if touch_is_down:
                            touch_up(cursor_x, cursor_y)
                            touch_is_down = False
                        mode = "GESTURE_ARMED"
                        commit_agg.reset()
                        rec_frames = []
                        palm_hold_t = cfg.palm_hold_s
                        pinch_hold_t = 0.0

                    if (
                        live_label == cfg.neutral_label
                        and live_conf >= cfg.start_min_conf
                    ):
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

                    if (
                        live_label == cfg.neutral_label
                        and live_conf >= cfg.start_min_conf
                    ):
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

                # status
                palm_prog = (
                    (palm_hold_t / cfg.palm_hold_s) if cfg.palm_hold_s > 0 else 0.0
                )
                palm_prog = max(0.0, min(1.0, palm_prog))
                if mode == "GESTURE_ARMED":
                    status = "READY"
                    palm_prog = 1.0
                elif palm_prog > 0.0:
                    status = "ARMED"
                else:
                    status = "IDLE"

                _draw_ui(frame, status, mode, palm_prog, last_sent_gesture)

                # red dot on camera frame while tracking/dragging (same landmark as tracking)
                if lm is not None and mode in ("TRACKING", "DRAGGING"):
                    nx, ny = _safe_lm_xy(lm, cfg.track_landmark_idx)
                    cx = int(nx * frame.shape[1])
                    cy = int(ny * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

                cv2.imshow(win_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            except Exception:
                log_exc(
                    "[PHONE_LIVE] LOOP EXCEPTION (this is why your window disappears):"
                )
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
