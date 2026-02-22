import time
from typing import Optional

import cv2

from utils.hand_tracking import HandTracker, put_hud


def run_test_cam(
    camera_index: int = 0, width: Optional[int] = 1280, height: Optional[int] = 720
) -> None:
    """
    Run a live webcam test for hand landmark tracking.

    The function opens a video capture device, initializes a HandTracker,
    and processes frames in real time. For each frame:

        - The image is horizontally flipped (mirror view).
        - Hand landmarks are detected and optionally drawn.
        - FPS is estimated using exponential smoothing.
        - A HUD overlay displays tracking information.

    Displayed information includes:
        - Number of detected hands
        - Current FPS
        - Handedness and confidence score
        - Example landmark pixel coordinates

    Press 'q' to exit the application.

    Parameters:
        camera_index (int): Index of the camera device to open.
        width (Optional[int]): Desired frame width in pixels.
        height (Optional[int]): Desired frame height in pixels.

    Returns:
        None
    """
    cap = cv2.VideoCapture(camera_index)

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"[ERROR] Konnte Kamera {camera_index} nicht öffnen.")
        return

    tracker = HandTracker(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        draw_style=True,
    )

    last_t = time.time()
    fps = 0.0

    print("[INFO] Kamera läuft. Drücke 'q', um zu beenden.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Kein Frame gelesen – beende.")
                break

            frame = cv2.flip(frame, 1)
            frame, hands = tracker.process_frame(frame, draw_landmarks=True)

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            n_hands = len(hands)
            lines = [
                f"Mode: test_cam | Hands: {n_hands} | FPS: {fps:5.1f}",
                "Taste 'q' zum Beenden",
            ]
            if n_hands:
                h0 = hands[0]
                (x0, y0, z0) = h0.coords_px[0]
                (x9, y9, z9) = h0.coords_px[9]
                lines.append(f"Handedness: {h0.handedness} ({h0.score:.2f})")
                lines.append(f"LM0(px)=({x0},{y0}) z={z0:.3f} | LM9(px)=({x9},{y9})")

            put_hud(frame, lines)

            cv2.imshow("MediaPipe – Hand Landmarks (test_cam)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
