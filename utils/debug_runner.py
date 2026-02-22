from __future__ import annotations

import time
import cv2

from utils.infer_runtime import run_live


def run_debug(camera_index: int = 0):
    """
    Run the live inference pipeline in debug mode with commit visualization.

    This function starts the real-time gesture inference loop using `run_live`
    and overlays additional debug information onto the video stream.

    A commit box is rendered in the bottom-right corner of the frame showing:
        - The last committed gesture label
        - The associated confidence score
        - The elapsed time since the last commit

    Internally:
        - `on_prediction` is triggered whenever a new prediction is committed.
        - `on_render` draws the debug overlay onto each frame before display.

    Parameters:
        camera_index (int): Index of the camera device to use.

    Returns:
        None
    """
    last_commit = {"label": "", "conf": 0.0, "t": 0.0}

    def on_prediction(label, conf, frame_bgr, state_str, seconds_left):
        last_commit["label"] = str(label)
        last_commit["conf"] = float(conf)
        last_commit["t"] = time.time()

    def on_render(frame_bgr, state_str, seconds_left):
        h, w = frame_bgr.shape[:2]
        age = (time.time() - last_commit["t"]) if last_commit["t"] else 999.0

        # bottom-right commit box
        x1, y1 = w - 420, h - 60
        x2, y2 = w - 10, h - 10
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.putText(
            frame_bgr,
            f"COMMIT: {last_commit['label'] or '-'} ({last_commit['conf']:.2f})",
            (x1 + 12, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"{age:.2f}s ago",
            (x1 + 12, y1 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    run_live(
        camera_index=camera_index,
        show_window=True,
        draw_phase_overlay=True,  # shows state + LIVE label/conf
        on_prediction=on_prediction,
        on_render=on_render,
    )
