import requests

TETRIS_GESTURE_URL = "http://127.0.0.1:8000/gesture"
TETRIS_TELEMETRY_URL = "http://127.0.0.1:8000/api/telemetry"

LABEL_TO_GESTURE = {
    "Links wischen": "swipe_left",
    "Rechts wischen": "swipe_right",
    "hand links drehen": "rotate",
    "hand rechts drehen": "rotate",
    "faust schließen": "fist",
    "nach oben wischen": None,
    "nach unten wischen": None,
    "NO_GESTURE": None,
    "garbage": None,
    "neutral_palm": None,
    "neutral_peace": None,
}


def _push_telemetry(state_str: str, label: str, conf: float, seconds_left: float, push_history: bool):
    try:
        requests.post(
            TETRIS_TELEMETRY_URL,
            json={
                "state": state_str,
                "label": label,
                "conf": float(conf),
                "seconds_left": float(seconds_left),
                "push_history": bool(push_history),
            },
            timeout=0.15,
        )
    except Exception:
        pass


def send_gesture_to_tetris(label: str, conf: float, phase_color: str, seconds_left: float):
    """
    Wird in main.py/run_live (Tetris mode) bei on_prediction aufgerufen.
    Wir verwenden:
      - gesture event (für Spielsteuerung)
      - telemetry event (für UI: mode + history)
    """
    # Telemetry immer aktualisieren (History: ja, weil committed)
    _push_telemetry(phase_color, label, conf, seconds_left, push_history=True)

    gesture = LABEL_TO_GESTURE.get(label)
    if gesture is None:
        return

    payload = {
        "gesture": gesture,
        "confidence": float(conf),
        "params": {
            "phase_color": phase_color,
            "seconds_left": float(seconds_left),
            "raw_label": label,
        },
    }

    try:
        requests.post(TETRIS_GESTURE_URL, json=payload, timeout=0.15)
    except Exception:
        pass
