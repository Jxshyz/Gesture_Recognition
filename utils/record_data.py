# utils/record_data.py

import cv2
import time
from pathlib import Path
import mediapipe as mp
import numpy as np
from datetime import datetime

from .feature_extractor import normalize_landmarks

# ---------------------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------------------

GESTURE_SEQUENCE = [
    ("neutral_palm", 10),
    ("neutral_peace", 10),
    ("swipe_left", 10),
    ("swipe_right", 10),
    ("swipe_up", 10),
    ("swipe_down", 10),
    ("rotate_left", 10),
    ("rotate_right", 10),
    ("close_fist", 10),
    ("garbage", 20),
]

RECORD_SECONDS = 2.0
COOLDOWN_SECONDS = 2.0
FPS_TARGET = 30

DATA_ROOT = Path("./data_raw")
DATA_ROOT.mkdir(exist_ok=True)


# ---------------------------------------------------------------
# UI BOX
# ---------------------------------------------------------------
def draw_status_box(frame, text, timer, is_recording):
    h, w = frame.shape[:2]

    box_w = 350
    box_h = 120
    x0, y0 = 10, 10
    x1, y1 = x0 + box_w, y0 + box_h

    # Hintergrund
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), -1)

    # Randfarbe abhängig von Aufnahme/Cooldown
    color = (0, 255, 0) if is_recording else (0, 0, 255)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 6)

    # Geste
    cv2.putText(frame, text, (x0 + 15, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Timer
    cv2.putText(frame, f"{timer:.1f}s", (x0 + 15, y0 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)


# ---------------------------------------------------------------
# EIN SAMPLEBLOCK EINER GESTE
# ---------------------------------------------------------------
def record_gesture_block(cap, gesture_name, person_name, samples):
    print(f"\n🔥 Aufnahme gestartet für: {gesture_name} — {samples} Samples")

    out_dir = DATA_ROOT / gesture_name / person_name
    out_dir.mkdir(parents=True, exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    drawing = mp.solutions.drawing_utils

    for sample_idx in range(samples):

        # ---------------------------------------------------
        # 1) COOLDOWN
        # ---------------------------------------------------
        cooldown_start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = time.time() - cooldown_start
            remaining = COOLDOWN_SECONDS - elapsed

            draw_status_box(frame, f"{gesture_name} (Cooldown)", remaining, is_recording=False)

            cv2.imshow("RECORDING", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                return False  # Abbruch

            if elapsed >= COOLDOWN_SECONDS:
                break

        # ---------------------------------------------------
        # 2) AUFNAHMEPHASE (2 Sekunden)
        # ---------------------------------------------------
        rec_start = time.time()
        save_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = time.time() - rec_start
            remaining = RECORD_SECONDS - elapsed

            draw_status_box(frame, f"{gesture_name}  [{sample_idx + 1}/{samples}]", remaining, is_recording=True)

            # Mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                lm = [(p.x, p.y, p.z) for p in result.multi_hand_landmarks[0].landmark]
                features = normalize_landmarks(lm)

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                np.save(out_dir / f"{gesture_name}_{person_name}_{ts}.npy", features)
                save_counter += 1

                drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            cv2.imshow("RECORDING", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                return False

            if elapsed >= RECORD_SECONDS:
                break

        print(f"[+] Sample {sample_idx + 1}/{samples} gespeichert ({save_counter} Frames).")

    print(f"✔ Fertig mit Geste: {gesture_name}")
    return True


# ---------------------------------------------------------------
# HAUPT-FUNKTION
# ---------------------------------------------------------------
def run_record(hand_arg: str, name: str, camera_index: int = 0):
    """
    Neu:
    - Ignoriert 'hand_arg'
    - Führt AUTOMATISCH ALLE GESTEN AUS
    """

    print("\n🚀 Starte automatischen Aufnahme-Modus für ALLE Gesten")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Kamera konnte nicht geöffnet werden.")
        return

    for gesture_name, samples in GESTURE_SEQUENCE:
        success = record_gesture_block(cap, gesture_name, name, samples)
        if not success:
            break  # ESC gedrückt → komplett abbrechen

    cap.release()
    cv2.destroyAllWindows()
    print("\n🎉 Aufnahme vollständig abgeschlossen!")
