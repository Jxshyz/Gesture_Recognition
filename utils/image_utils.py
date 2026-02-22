"""
Utility functions for image display in live testing modes.

This module provides helper functions for:

- Loading label-associated images from disk
- Rendering preview images inside a fixed UI box
- Drawing labels onto video frames

Used primarily in run_live_test (main.py).
"""
# utils/image_utils.py
# Hilfsfunktionen für Bildanzeige im run_live_test (in main.py verwendet)
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import cv2
import numpy as np


def load_pictures(
    pictures_dir: Path, mapping: Dict[str, str]
) -> Dict[str, Optional[np.ndarray]]:
    """
    Load images from disk into a label-based cache.

    Each label is mapped to a filename. If the file exists,
    it is loaded using OpenCV (including alpha channel if present).
    Missing files are stored as None.

    Parameters:
        pictures_dir (Path):
            Directory containing the image files.

        mapping (Dict[str, str]):
            Mapping from label → filename.

    Returns:
        Dict[str, Optional[np.ndarray]]:
            Dictionary mapping label → loaded image (BGR or BGRA),
            or None if the file does not exist.
    """
    cache: Dict[str, Optional[np.ndarray]] = {}
    for label, fname in mapping.items():
        fp = pictures_dir / fname
        if fp.exists():
            img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            cache[label] = img
        else:
            cache[label] = None
    return cache


def draw_picture_with_border(
    frame: np.ndarray, picture: Optional[np.ndarray], phase_color: str
) -> None:
    """
    Draw a preview picture inside a fixed box with a colored border.

    The picture is resized to fit into a 160x160 box in the top-left
    corner of the frame while preserving aspect ratio.

    If no picture is provided (None), a placeholder rectangle is drawn.

    The border color depends on the phase:
        - "green" → green border
        - otherwise → red border

    Parameters:
        frame (np.ndarray):
            Target frame (modified in place).

        picture (Optional[np.ndarray]):
            Image to render (BGR or BGRA), or None.

        phase_color (str):
            Phase indicator controlling border color.

    Returns:
        None
    """
    x0, y0 = 10, 10
    box_w, box_h = 160, 160

    if picture is None:
        cv2.rectangle(
            frame, (x0, y0), (x0 + box_w, y0 + box_h), (60, 60, 60), thickness=-1
        )
    else:
        ph, pw = picture.shape[:2]
        scale = min(box_w / pw, box_h / ph)
        new_w, new_h = int(pw * scale), int(ph * scale)
        resized = cv2.resize(picture, (new_w, new_h), interpolation=cv2.INTER_AREA)
        roi = (
            resized
            if resized.ndim == 3 and resized.shape[2] == 3
            else cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR)
        )
        frame[y0 : y0 + new_h, x0 : x0 + new_w] = roi

    col = (0, 200, 0) if phase_color == "green" else (0, 0, 200)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), col, thickness=3)


def draw_label(frame: np.ndarray, label: str) -> None:
    """
    Draw a text label below the preview picture.

    Parameters:
        frame (np.ndarray):
            Target frame (modified in place).

        label (str):
            Text label to render.

    Returns:
        None
    """
    cv2.putText(
        frame,
        label,
        (12, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
