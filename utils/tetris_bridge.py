# utils/tetris_bridge.py
from __future__ import annotations

import requests

TETRIS_GESTURE_URL = "http://127.0.0.1:8000/gesture"
TETRIS_TELEMETRY_URL = "http://127.0.0.1:8000/api/telemetry"

# Labels aus deinem Modell -> Events für die Webapp
LABEL_TO_GESTURE = {
    # Gewünschte Steuerung
    "swipe_left": "swipe_left",
    "swipe_right": "swipe_right",
    "rotate_left": "rotate_left",
    "close_fist": "close_fist",
    # Rückwärts-Kompatibilität
    "rotate": "rotate_left",
    "fist": "close_fist",
    # NICHT ans Spiel senden:
    "neutral_palm": None,
    "neutral_peace": None,
    "garbage": None,
    "pinch": None,
    "finger_pistol": None,
    "swipe_up": None,
    "swipe_down": None,
    "rotate_right": None,
}


def _phase_color_from_state(state_str: str) -> str:
    """
    Sehr robust:
    - wenn state_str irgendwie nach rot klingt -> "red"
    - sonst -> "green"
    """
    s = (state_str or "").strip().lower()
    if "red" in s or "pause" in s or "wait" in s or "idle" in s:
        return "red"
    return "green"


def send_telemetry_only(
    state: str,
    label: str,
    conf: float,
    seconds_left: float,
    armed_progress: float,
    armed_ready: bool,
    push_history: bool = False,
):
    payload = {
        "state": str(state),
        "label": str(label),
        "conf": float(conf),
        "seconds_left": float(seconds_left),
        "armed_progress": float(armed_progress),
        "armed_ready": bool(armed_ready),
        "push_history": bool(push_history),
    }

    try:
        requests.post(TETRIS_TELEMETRY_URL, json=payload, timeout=0.2)
    except Exception:
        pass


def send_gesture_to_tetris(label: str, conf: float, state_str: str, seconds_left: float):
    """
    main.py ruft genau diese Signatur auf:
      send_gesture_to_tetris(label, conf, state_str, seconds_left)

    Wir mappen label -> gesture und geben phase_color aus state_str abgeleitet mit.
    """
    gesture = LABEL_TO_GESTURE.get((label or "").lower())
    if gesture is None:
        return

    phase_color = _phase_color_from_state(state_str)

    payload = {
        "gesture": gesture,
        "confidence": float(conf),
        "params": {
            "phase_color": str(phase_color),
            "seconds_left": float(seconds_left),
            "raw_label": str(label),
            "state_str": str(state_str),
        },
    }

    try:
        requests.post(TETRIS_GESTURE_URL, json=payload, timeout=0.2)
    except Exception:
        pass
