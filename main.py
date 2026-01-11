# main.py (komplette Datei)
import sys
import webbrowser
from pathlib import Path
import time
from urllib.parse import quote

from utils.cam_test import run_test_cam
from utils.record_data import run_record
from utils.game_runner import run_runner, RunnerConfig


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
            "  debug\n"
            "  -tetris\n"
            "  phone_live\n"
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
            print("Usage: python main.py record_data <gesture|all> <Name> [camera_index] [hand]")
            print("Examples:")
            print("  python main.py record_data all josh 1 right")
            print("  python main.py record_data swipe_left josh 0 left")
            sys.exit(2)

        gesture_or_all = sys.argv[2].lower()
        name = sys.argv[3]

        cam_idx = 0
        hand = "Right"

        for arg in sys.argv[4:]:
            if arg.isdigit():
                cam_idx = int(arg)
            elif arg.lower() in ("l", "left", "r", "right"):
                hand = "Right" if arg.lower().startswith("r") else "Left"

        run_record(gesture_arg=gesture_or_all, name=name, camera_index=cam_idx, hand=hand)
        return

    # -------------------------------------------------------------------------
    # 3) Modell trainieren
    # -------------------------------------------------------------------------
    elif cmd == "train_model":
        from utils.train_gesture_model import train_and_save

        train_and_save()
        return

    # -------------------------------------------------------------------------
    # 4) DEBUG
    # -------------------------------------------------------------------------
    elif cmd == "debug":
        from utils.debug_runner import run_debug

        cam_idx = int(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[2].isdigit() else 0
        run_debug(camera_index=cam_idx)
        return

    # -------------------------------------------------------------------------
    # 5) LIVE-FSM (normal / tetris / subway)
    # -------------------------------------------------------------------------
    elif cmd == "run_live":
        extra = sys.argv[2:]

        subway_mode = any(a.lower() == "-subway" for a in extra)
        tetris_mode = any(a.lower() == "-tetris" for a in extra)

        cam_idx = 0
        for a in extra:
            if a.isdigit():
                cam_idx = int(a)
                break

        if subway_mode:
            run_runner(
                RunnerConfig(
                    camera_index=cam_idx,
                    port=8010,
                    pred_min_interval_s=0.06,
                    arm_hold_s=0.10,
                    cooldown_s=0.01,
                )
            )
            return

        from utils.infer_runtime import run_live
        from utils.gesture_stream import set_latest_frame
        import cv2

        if not tetris_mode:
            run_live(
                camera_index=cam_idx,
                show_window=True,
                draw_phase_overlay=True,
                on_prediction=None,
                on_render=None,
                on_telemetry=None,
            )
            return

        # ----------------------------
        # TETRIS MODE: ask username
        # ----------------------------
        user_name = ""
        while not user_name:
            user_name = input("Choose your name: ").strip()

        from utils.tetris_app import start_tetris_server_background
        from utils.tetris_bridge import send_gesture_to_tetris, send_telemetry_only

        start_tetris_server_background()

        user_q = quote(user_name)
        webbrowser.open(f"http://127.0.0.1:8000/?user={user_q}&v={int(time.time())}")

        def on_prediction(label, conf, frame_bgr, state_str, seconds_left):
            send_gesture_to_tetris(label, conf, state_str, seconds_left)

        def on_render(frame_bgr, state_str, seconds_left):
            ok, buf = cv2.imencode(".jpg", frame_bgr)
            if ok:
                set_latest_frame(buf.tobytes())

        def on_telemetry(state, live_label, live_conf, seconds_left, armed_progress, armed_ready):
            send_telemetry_only(
                state=state,
                label=live_label,
                conf=live_conf,
                seconds_left=seconds_left,
                armed_progress=armed_progress,
                armed_ready=armed_ready,
                push_history=False,
            )

        run_live(
            camera_index=cam_idx,
            show_window=False,
            draw_phase_overlay=False,
            on_prediction=on_prediction,
            on_render=on_render,
            on_telemetry=on_telemetry,
        )
        return

    # -------------------------------------------------------------------------
    # 6) Debug-Testmodus
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
            on_telemetry=None,
        )
        return

    # -------------------------------------------------------------------------
    # PHONE LIVE + OVERLAY
    # -------------------------------------------------------------------------
    elif cmd == "phone_live":
        import subprocess
        import atexit
        from utils.phone_gesture_live import run_phone_gesture_live

        cam_idx = int(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[2].isdigit() else 0

        overlay_proc = subprocess.Popen(
            [sys.executable, "-u", "utils/overlay.py", "--udp", "5005"],
        )

        def _cleanup():
            try:
                overlay_proc.terminate()
            except Exception:
                pass

        atexit.register(_cleanup)

        run_phone_gesture_live(camera_index=cam_idx)
        return

    # -------------------------------------------------------------------------
    # Standalone-Tetris
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
