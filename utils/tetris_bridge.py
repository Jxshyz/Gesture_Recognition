# utils/tetris_bridge.py
from __future__ import annotations

import requests

TETRIS_GESTURE_URL = "http://127.0.0.1:8000/gesture"
TETRIS_TELEMETRY_URL = "http://127.0.0.1:8000/api/telemetry"

# Labels aus deinem Modell -> Events für die Webapp
LABEL_TO_GESTURE = {
    "swipe_left": "swipe_left",
    "swipe_right": "swipe_right",
    "rotate": "rotate",
    "fist": "fist",
    # NICHT ans Spiel senden:
    "neutral_palm": None,
    "neutral_peace": None,
    "garbage": None,
}


def send_telemetry_only(
    state: str,
    label: str,
    conf: float,
    seconds_left: float,
    armed_progress: float,
    armed_ready: bool,
    push_history: bool = False,
):
    """
    UI/Telemetry Updates (Progressbar, Status, Live Label).
    Default: push_history=False, damit History nicht vollgemüllt wird.
    """
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


def send_gesture_to_tetris(label: str, conf: float, phase_color: str, seconds_left: float):
    """
    Wird bei COMMIT aufgerufen -> steuert das Spiel.
    """
    gesture = LABEL_TO_GESTURE.get(label)
    if gesture is None:
        return

    payload = {
        "gesture": gesture,
        "confidence": float(conf),
        "params": {
            "phase_color": str(phase_color),
            "seconds_left": float(seconds_left),
            "raw_label": str(label),
        },
    }

    try:
        requests.post(TETRIS_GESTURE_URL, json=payload, timeout=0.2)
    except Exception:
        pass
