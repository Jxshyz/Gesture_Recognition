# main.py (erweitert um "run_live_test": Bildlogik liegt hier in main, nicht mehr in infer_runtime)
import sys
from pathlib import Path

from utils.cam_test import run_test_cam
from utils.record_data import run_record

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command>\nCommands: test_cam | record_data | train_model | run_live | run_live_test")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "test_cam":
        cam_idx = 0
        if len(sys.argv) >= 3:
            try:
                cam_idx = int(sys.argv[2])
            except ValueError:
                pass
        run_test_cam(camera_index=cam_idx)

    elif cmd == "record_data":
        if len(sys.argv) < 4:
            print("Usage: python main.py record_data <l|r> <Name> [camera_index]")
            sys.exit(2)
        hand_arg = sys.argv[2]
        name = sys.argv[3]
        cam_idx = 0
        if len(sys.argv) >= 5:
            try:
                cam_idx = int(sys.argv[4])
            except ValueError:
                pass
        run_record(hand_arg=hand_arg, name=name, camera_index=cam_idx)

    elif cmd == "train_model":
        from utils.train_gesture_model import train_and_save
        train_and_save()

    elif cmd == "run_live":
        cam_idx = 0
        if len(sys.argv) >= 3:
            try:
                cam_idx = int(sys.argv[2])
            except ValueError:
                pass
        from utils.infer_runtime import run_live
        # Run ohne spezielle Ausgabe-Logik (nur HUD/Hand-Landmarks)
        run_live(camera_index=cam_idx, green_dur=1.0, red_dur=1.0, conf_threshold=0.0)

    elif cmd == "run_live_test":
        # Hier liegt die Bild-Logik (Mapping Geste -> Bild) in main.py:
        cam_idx = 0
        if len(sys.argv) >= 3:
            try:
                cam_idx = int(sys.argv[2])
            except ValueError:
                pass

        from utils.infer_runtime import run_live
        from utils.image_utils import load_pictures, draw_picture_with_border, draw_label

        PICTURES_DIR = Path("./pictures")
        GESTURE_TO_FILE = {
            "Links wischen":        "links_wischen.png",
            "Rechts wischen":       "rechts_wischen.png",
            "nach oben wischen":    "nach_oben_wischen.png",
            "nach unten wischen":   "nach_unten_wischen.png",
            "faust schließen":      "faust_schliessen.png",
            "hand links drehen":    "hand_links_drehen.png",
            "hand rechts drehen":   "hand_rechts_drehen.png",
        }
        cache = load_pictures(PICTURES_DIR, GESTURE_TO_FILE)
        current_label = "Links wischen"

        def on_prediction(label: str, conf: float, frame_bgr, phase_color: str, seconds_left: float):
            # Setze aktuelles Bild/Label bei jeder gültigen Vorhersage (in Grün-Phase)
            nonlocal current_label
            current_label = label
            img = cache.get(label)
            draw_picture_with_border(frame_bgr, img, phase_color)
            draw_label(frame_bgr, f"{label}  (conf={conf:.2f})")

        # run_live kümmert sich um Kamera, Tracking, Puffer & Modell;
        # wir geben nur die Overlay-Logik via Callback.
        run_live(camera_index=cam_idx, green_dur=1.0, red_dur=1.0, conf_threshold=0.0, on_prediction=on_prediction)

    else:
        print(f"Unbekannter Befehl: {cmd}\nVerfügbare Befehle: test_cam | record_data | train_model | run_live | run_live_test")
        sys.exit(2)

if __name__ == "__main__":
    main()
