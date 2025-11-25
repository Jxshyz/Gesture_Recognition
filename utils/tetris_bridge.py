import requests

TETRIS_GESTURE_URL = "http://127.0.0.1:8000/gesture"

LABEL_TO_GESTURE = {
    "Links wischen": "swipe_left",
    "Rechts wischen": "swipe_right",
    "hand links drehen": "rotate",
    "hand rechts drehen": "rotate",
    "faust schließen": "fist",
    "nach oben wischen": None,
    "nach unten wischen": None,
    "NO_GESTURE": None,
}


def send_gesture_to_tetris(label: str, conf: float, phase_color: str, seconds_left: float):
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
