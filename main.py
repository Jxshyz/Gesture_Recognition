import sys
import webbrowser
from pathlib import Path

from utils.cam_test import run_test_cam
from utils.record_data import run_record


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python main.py <command>\n"
            "Commands:\n"
            "  test_cam\n"
            "  record_data\n"
            "  train_model\n"
            "  run_live\n"
            "  run_live_test\n"
            "  -tetris\n"
        )
        sys.exit(1)

    cmd = sys.argv[1].lower()

    # -------------------------------------------------------------------------
    # 1) Kamera testen
    # -------------------------------------------------------------------------
    if cmd == "test_cam":
        cam_idx = int(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[2].isdigit() else 0
        run_test_cam(camera_index=cam_idx)
        return

    # -------------------------------------------------------------------------
    # 2) Trainingsdaten aufnehmen
    # -------------------------------------------------------------------------
    elif cmd == "record_data":
        if len(sys.argv) < 4:
            print("Usage: python main.py record_data <gesture> <Name> [camera_index]")
            sys.exit(2)

        gesture = sys.argv[2]
        name = sys.argv[3]
        cam_idx = int(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[4].isdigit() else 0

        run_record(hand_arg=gesture, name=name, camera_index=cam_idx)
        return

    # -------------------------------------------------------------------------
    # 3) Modell trainieren
    # -------------------------------------------------------------------------
    elif cmd == "train_model":
        from utils.train_gesture_model import train_and_save

        train_and_save()
        return

    # -------------------------------------------------------------------------
    # 4) LIVE-FSM (mit/ohne Tetris)
    # -------------------------------------------------------------------------
    elif cmd == "run_live":
        from utils.infer_runtime import run_live
        from utils.gesture_stream import set_latest_frame
        import cv2

        cam_idx = 0
        tetris_mode = False

        extra = sys.argv[2:]
        if extra:
            # run_live -tetris 1
            if extra[0].lower() == "-tetris":
                tetris_mode = True
                if len(extra) >= 2 and extra[1].isdigit():
                    cam_idx = int(extra[1])
            # run_live 1 -tetris
            else:
                if extra[0].isdigit():
                    cam_idx = int(extra[0])
                if len(extra) >= 2 and extra[1].lower() == "-tetris":
                    tetris_mode = True

        # ---------------------------------------------------------------------
        # NORMALER LIVE-MODUS (OpenCV-Fenster)
        # ---------------------------------------------------------------------
        if not tetris_mode:
            run_live(
                camera_index=cam_idx,
                show_window=True,
                draw_phase_overlay=True,
                on_prediction=None,
                on_render=None,
            )
            return

        # ---------------------------------------------------------------------
        # TETRIS-MODUS (Web-App + FSM + eingebettete Kamera)
        # ---------------------------------------------------------------------
        from utils.tetris_app import start_tetris_server_background
        from utils.tetris_bridge import send_gesture_to_tetris

        start_tetris_server_background()
        webbrowser.open("http://127.0.0.1:8000")

        # Geste an Tetris senden
        def on_prediction(label, conf, frame_bgr, state_str, seconds_left):
            send_gesture_to_tetris(label, conf, state_str, seconds_left)

        # Kamera in Web-App einbetten
        def on_render(frame_bgr, state_str, seconds_left):
            ok, buf = cv2.imencode(".jpg", frame_bgr)
            if ok:
                set_latest_frame(buf.tobytes())

        # FSM starten (kein OpenCV)
        run_live(
            camera_index=cam_idx,
            show_window=False,
            draw_phase_overlay=False,
            on_prediction=on_prediction,
            on_render=on_render,
        )
        return

    # -------------------------------------------------------------------------
    # 5) Debug-Testmodus
    # -------------------------------------------------------------------------
    elif cmd == "run_live_test":
        from utils.infer_runtime import run_live
        from utils.image_utils import load_pictures, draw_picture_with_border, draw_label

        cam_idx = int(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[2].isdigit() else 0

        PICTURES_DIR = Path("./pictures")
        G_MAP = {
            "swipe_left": "links_wischen.png",
            "swipe_right": "rechts_wischen.png",
            "rotate": "hand_links_drehen.png",
            "fist": "faust_schliessen.png",
            "neutral_palm": "hand_rechts_drehen.png",
            "neutral_peace": "hand_links_drehen.png",
            "garbage": "no_gesture.png",
        }

        cache = load_pictures(PICTURES_DIR, G_MAP)
        current_label = "garbage"
        current_img = cache.get(current_label)

        def on_prediction(label, conf, frame_bgr, state_str, seconds_left):
            nonlocal current_label, current_img
            current_label = label
            current_img = cache.get(label, None)

        def on_render(frame_bgr, state_str, seconds_left):
            draw_picture_with_border(frame_bgr, current_img, state_str)
            draw_label(frame_bgr, current_label)

        run_live(
            camera_index=cam_idx,
            show_window=True,
            draw_phase_overlay=True,
            on_prediction=on_prediction,
            on_render=on_render,
        )
        return

    # -------------------------------------------------------------------------
    # 6) Standalone-Tetris (kein Kamera-Feed)
    # -------------------------------------------------------------------------
    elif cmd == "-tetris":
        from utils.tetris_app import run_tetris_server

        run_tetris_server()
        return

    else:
        print("Unbekannter Befehl:", cmd)
        sys.exit(2)


if __name__ == "__main__":
    main()
