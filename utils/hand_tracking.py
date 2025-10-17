from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp


@dataclass
class HandLandmarks:
    """
    Container für Hand-Landmarks je Hand.
    coords_norm: Liste aus (x,y,z) in normalisierten MediaPipe-Koordinaten [0..1].
    coords_px:   Liste aus (x_px,y_px,z) in Pixel (z unverändert), passend zur Framegröße.
    handedness:  "Left" oder "Right" laut MediaPipe-Klassifikation.
    score:       Klassifikations-Confidence der Händigkeit.
    """

    coords_norm: List[Tuple[float, float, float]]
    coords_px: List[Tuple[int, int, float]]
    handedness: str
    score: float


class HandTracker:
    """Wrapper um MediaPipe Hands für Live-Tracking von 21 Landmarks."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        draw_style: bool = True,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._draw_style = draw_style

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()

    def process_frame(
        self, frame_bgr: np.ndarray, draw_landmarks: bool = True
    ) -> Tuple[np.ndarray, List[HandLandmarks]]:
        """Verarbeitet ein BGR-Frame, gibt optional gezeichnetes Frame und erkannte Hände zurück."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)

        hands_out: List[HandLandmarks] = []

        if result.multi_hand_landmarks:
            handed = result.multi_handedness or []

            for i, hand_lms in enumerate(result.multi_hand_landmarks):
                coords_norm = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
                coords_px = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in hand_lms.landmark]

                label = "Unknown"
                score = 0.0
                if i < len(handed) and handed[i].classification:
                    proto = handed[i].classification[0]
                    label = proto.label
                    score = float(proto.score)

                hands_out.append(
                    HandLandmarks(
                        coords_norm=coords_norm,
                        coords_px=coords_px,
                        handedness=label,
                        score=score,
                    )
                )

                if draw_landmarks:
                    if self._draw_style:
                        self._mp_drawing.draw_landmarks(
                            frame_bgr,
                            hand_lms,
                            self._mp_hands.HAND_CONNECTIONS,
                            self._mp_styles.get_default_hand_landmarks_style(),
                            self._mp_styles.get_default_hand_connections_style(),
                        )
                    else:
                        self._mp_drawing.draw_landmarks(frame_bgr, hand_lms, self._mp_hands.HAND_CONNECTIONS)

        return frame_bgr, hands_out


def put_hud(
    frame_bgr: np.ndarray,
    text_lines: List[str],
    origin: Tuple[int, int] = (10, 24),
    line_height: int = 22,
) -> None:
    """Einfaches Overlay mehrerer Textzeilen links oben."""
    x0, y0 = origin
    for i, line in enumerate(text_lines):
        y = y0 + i * line_height
        cv2.putText(
            frame_bgr,
            line,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
