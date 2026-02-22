"""
Hand tracking utilities built on top of MediaPipe Hands.

This module provides:

- A structured container for detected hand landmarks
- A wrapper class for real-time hand tracking
- A simple HUD text overlay utility

Designed for live gesture recognition and debugging workflows.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp


@dataclass
class HandLandmarks:
    """
    Structured container for a single detected hand.

    Attributes:
        coords_norm (List[Tuple[float, float, float]]):
            List of 21 landmarks in normalized MediaPipe coordinates (0..1).

        coords_px (List[Tuple[int, int, float]]):
            Pixel coordinates adapted to the current frame size
            (z remains unchanged).

        handedness (str):
            Predicted handedness label ("Left", "Right", or "Unknown").

        score (float):
            Confidence score of the handedness classification.
    """

    coords_norm: List[Tuple[float, float, float]]
    coords_px: List[Tuple[int, int, float]]
    handedness: str
    score: float


class HandTracker:
    """
    Real-time wrapper around MediaPipe Hands.

    Provides:
        - Frame processing (BGR input)
        - Landmark extraction (normalized + pixel coordinates)
        - Optional landmark drawing
        - Handedness classification

    Intended for live video streams (e.g., webcam input).
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        draw_style: bool = True,
    ) -> None:
        """
        Initialize the MediaPipe Hands tracker.

        Parameters:
            static_image_mode (bool):
                Whether to treat each frame independently.

            max_num_hands (int):
                Maximum number of hands to detect.

            model_complexity (int):
                Model complexity level (0 or 1).

            min_detection_confidence (float):
                Minimum confidence required for hand detection.

            min_tracking_confidence (float):
                Minimum confidence required for landmark tracking.

            draw_style (bool):
                If True, use MediaPipe default drawing styles.
        """
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
        """
        Release MediaPipe resources.

        Should be called when the tracker is no longer needed.
        """
        if self._hands is not None:
            self._hands.close()

    def process_frame(
        self, frame_bgr: np.ndarray, draw_landmarks: bool = True
    ) -> Tuple[np.ndarray, List[HandLandmarks]]:
        """
        Process a BGR frame and detect hand landmarks.

        Steps:
            - Convert BGR → RGB
            - Run MediaPipe inference
            - Extract normalized and pixel coordinates
            - Retrieve handedness classification
            - Optionally draw landmarks on the frame

        Parameters:
            frame_bgr (np.ndarray):
                Input frame in BGR format.

            draw_landmarks (bool):
                Whether to draw landmarks onto the frame.

        Returns:
            Tuple[np.ndarray, List[HandLandmarks]]:
                - The (optionally modified) BGR frame
                - A list of detected hands with landmark data
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)

        hands_out: List[HandLandmarks] = []

        if result.multi_hand_landmarks:
            handed = result.multi_handedness or []

            for i, hand_lms in enumerate(result.multi_hand_landmarks):
                coords_norm = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
                coords_px = [
                    (int(lm.x * w), int(lm.y * h), lm.z) for lm in hand_lms.landmark
                ]

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
                        self._mp_drawing.draw_landmarks(
                            frame_bgr, hand_lms, self._mp_hands.HAND_CONNECTIONS
                        )

        return frame_bgr, hands_out


def put_hud(
    frame_bgr: np.ndarray,
    text_lines: List[str],
    origin: Tuple[int, int] = (10, 24),
    line_height: int = 22,
) -> None:
    """
    Draw multiple text lines onto a frame (HUD overlay).

    Parameters:
        frame_bgr (np.ndarray):
            Target frame (modified in place).

        text_lines (List[str]):
            List of strings to render line by line.

        origin (Tuple[int, int]):
            Top-left starting position (x, y).

        line_height (int):
            Vertical spacing between lines in pixels.

    Returns:
        None
    """
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
