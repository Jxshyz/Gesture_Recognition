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
