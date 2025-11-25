# tools/mock_sender.py
import time
import random
import requests

URL = "http://127.0.0.1:8000/gesture"

GESTURES = [
    {"gesture": "swipe_left", "confidence": 0.95, "params": {"speed": 1.0}},
    {"gesture": "swipe_right", "confidence": 0.93, "params": {"speed": 1.0}},
    {"gesture": "rotate", "confidence": 0.90, "params": {}},
    {"gesture": "push", "confidence": 0.88, "params": {"strength": 0.6}},
    {"gesture": "fist", "confidence": 0.97, "params": {}},
    {"gesture": "pause", "confidence": 0.99, "params": {}},
]


def main():
    print("Sending random gestures to Tetris. Ctrl+C to stop.")
    while True:
        ev = random.choice(GESTURES)
        try:
            r = requests.post(URL, json=ev, timeout=2)
            print("=>", ev, "| resp:", r.status_code, r.json())
        except Exception as e:
            print("POST failed:", e)
        time.sleep(1.5)


if __name__ == "__main__":
    main()
