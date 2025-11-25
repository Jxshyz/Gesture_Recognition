import sys
import webbrowser
from pathlib import Path

from utils.cam_test import run_test_cam
from utils.record_data import run_record


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python main.py <command>\n"
            "Commands: test_cam | record_data | train_model | run_live | run_live_test | -Tetris"
        )
        sys.exit(1)

    cmd = sys.argv[1].lower()

    # ---------------------------------------------------
    # 1) Kamera testen
    # ---------------------------------------------------
    if cmd == "test_cam":
        cam_idx = int(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[2].isdigit() else 0
        run_test_cam(camera_index=cam_idx)

    # ---------------------------------------------------
    # 2) Daten aufnehmen
    # ---------------------------------------------------
    elif cmd == "record_data":
        if len(sys.argv) < 4:
            print("Usage: python main.py record_data <l|r> <Name> [camera_index]")
            sys.exit(2)
        hand_arg, name = sys.argv[2], sys.argv[3]
        cam_idx = int(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[4].isdigit() else 0
        run_record(hand_arg=hand_arg, name=name, camera_index=cam_idx)

    # ---------------------------------------------------
    # 3) Modell trainieren
    # ---------------------------------------------------
    elif cmd == "train_model":
        from utils.train_gesture_model import train_and_save

        train_and_save()

    # ---------------------------------------------------
    # 4) Live-Erkennung (mit oder ohne Tetris)
    # ---------------------------------------------------
    elif cmd == "run_live":
        from utils.infer_runtime import run_live
        from utils.gesture_stream import set_latest_frame
        import cv2

        cam_idx = 0
        tetris_mode = False

        extra = sys.argv[2:]
        if extra:
            if extra[0].lower() == "-tetris":
                tetris_mode = True
                if len(extra) >= 2 and extra[1].isdigit():
                    cam_idx = int(extra[1])
            else:
                if extra[0].isdigit():
                    cam_idx = int(extra[0])
                if len(extra) >= 2 and extra[1].lower() == "-tetris":
                    tetris_mode = True

        if not tetris_mode:
            run_live(camera_index=cam_idx, green_dur=2.0, red_dur=2.0)
            return

        # -------------------------------------------------------
        # TETRIS + Kamera einbetten
        # -------------------------------------------------------
        from utils.tetris_app import start_tetris_server_background
        from utils.tetris_bridge import send_gesture_to_tetris

        start_tetris_server_background()
        webbrowser.open("http://127.0.0.1:8000")

        def on_prediction(label, conf, frame_bgr, phase_color, seconds_left):
            send_gesture_to_tetris(label, conf, phase_color, seconds_left)

        def on_render(frame_bgr, phase_color, seconds_left):
            ok, buf = cv2.imencode(".jpg", frame_bgr)
            if ok:
                set_latest_frame(buf.tobytes())

        run_live(
            camera_index=cam_idx,
            green_dur=2.0,
            red_dur=2.0,
            conf_threshold=0.0,
            final_min_votes=3,
            final_min_conf=0.55,
            downsample_interval_s=0.075,
            enable_no_gesture=True,
            on_prediction=on_prediction,
            on_render=on_render,
        )

    # ---------------------------------------------------
    # TESTMODUS (mit Bild-Einblendungen)
    # ---------------------------------------------------
    elif cmd == "run_live_test":
        cam_idx = int(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[2].isdigit() else 0

        from utils.infer_runtime import run_live
        from utils.image_utils import load_pictures, draw_picture_with_border, draw_label

        PICTURES_DIR = Path("./pictures")
        GESTURE_TO_FILE = {
            "Links wischen": "links_wischen.png",
            "Rechts wischen": "rechts_wischen.png",
            "nach oben wischen": "nach_oben_wischen.png",
            "nach unten wischen": "nach_unten_wischen.png",
            "faust schließen": "faust_schliessen.png",
            "hand links drehen": "hand_links_drehen.png",
            "hand rechts drehen": "hand_rechts_drehen.png",
            "NO_GESTURE": "no_gesture.png",
        }

        cache = load_pictures(PICTURES_DIR, GESTURE_TO_FILE)
        current_label = "NO_GESTURE"
        current_img = cache.get(current_label)

        def on_prediction(label, conf, frame_bgr, phase_color, seconds_left):
            nonlocal current_label, current_img
            current_label = label
            current_img = cache.get(label)

        def on_render(frame_bgr, phase_color, seconds_left):
            draw_picture_with_border(frame_bgr, current_img, phase_color)
            draw_label(frame_bgr, current_label)

        run_live(
            camera_index=cam_idx,
            green_dur=2.0,
            red_dur=2.0,
            conf_threshold=0.0,
            final_min_votes=3,
            final_min_conf=0.55,
            downsample_interval_s=0.075,
            enable_no_gesture=True,
            on_prediction=on_prediction,
            on_render=on_render,
        )

    # ---------------------------------------------------
    # EIGENSTÄNDIGES TETRIS
    # ---------------------------------------------------
    elif cmd == "-tetris":
        from utils.tetris_app import run_tetris_server

        run_tetris_server()

    else:
        print("Unbekannter Befehl:", cmd)
        sys.exit(2)


if __name__ == "__main__":
    main()
