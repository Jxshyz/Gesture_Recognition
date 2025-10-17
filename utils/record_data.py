# utils/record_data.py
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pandas as pd

from utils.hand_tracking import HandTracker, put_hud


@dataclass
class Phase:
    color: str  # "red" oder "green"
    duration_s: float


def _build_label_for_green_index(i: int) -> str:
    # 0..69 in 7 Blöcken à 10:
    blocks = [
        "Links wischen",  # 0-9
        "Rechts wischen",  # 10-19
        "nach oben wischen",  # 20-29
        "nach unten wischen",  # 30-39
        "faust schließen",  # 40-49
        "hand links drehen",  # 50-59
        "hand rechts drehen",  # 60-69
    ]
    block = i // 10
    return blocks[min(block, len(blocks) - 1)]


def _current_label(greens_done: int) -> str:
    # Zeige immer die Zielgeste des aktuellen/kommenden grünen Fensters
    return _build_label_for_green_index(min(greens_done, 69))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _next_out_path(out_dir: Path, name: str) -> Path:
    """
    Liefert einen nicht existierenden Pfad der Form:
    ./data/Gestures_<Name>_<N>.pkl, wobei N bei 1 beginnt und hochzählt.
    """
    n = 1
    while True:
        candidate = out_dir / f"Gestures_{name}_{n}.pkl"
        if not candidate.exists():
            return candidate
        n += 1


def draw_square_with_timer(frame_bgr: np.ndarray, color: str, seconds_left: float) -> None:
    # Viereck oben links
    x0, y0, w, h = 10, 10, 140, 140
    col = (0, 180, 0) if color == "green" else (0, 0, 200)
    cv2.rectangle(frame_bgr, (x0, y0), (x0 + w, y0 + h), col, thickness=-1)

    # Timer-Zahl zentriert ins Viereck
    txt = f"{seconds_left:0.1f}s"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    tx = x0 + (w - tw) // 2
    ty = y0 + (h + th) // 2 - 6
    cv2.putText(frame_bgr, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def draw_label_text(frame_bgr: np.ndarray, label: str) -> None:
    # Text unter dem Viereck
    x, y = 10, 10 + 140 + 28
    cv2.putText(frame_bgr, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def run_record(
    hand_arg: str, name: str, camera_index: int = 0, width: Optional[int] = 1280, height: Optional[int] = 720
) -> None:
    # Hand ableiten
    hand_arg = hand_arg.lower()
    if hand_arg not in ("l", "r"):
        print("[ERROR] Hand-Argument muss 'l' oder 'r' sein.")
        return
    hand_text = "links" if hand_arg == "l" else "rechts"

    # Pfad vorbereiten (inkrementierender Dateiname)
    out_dir = Path("./data")
    _ensure_dir(out_dir)
    out_path = _next_out_path(out_dir, name)  # z. B. Gestures_Joschua_1.pkl, Gestures_Joschua_2.pkl, ...

    # Kamera
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

    # Aufnahmesteuerung:
    # Start: 5s rot, dann (1s grün, 2s rot) bis 70 grüne Fenster erreicht
    greens_target = 70
    greens_done = 0
    phase = Phase(color="red", duration_s=5.0)
    phase_start = time.time()
    phase_end = phase_start + phase.duration_s

    # Datenspeicher
    rows: List[Dict] = []
    idx = 0

    print(f"[INFO] Aufnahme gestartet. Speichere nach Abschluss unter: {out_path.name}")
    print("[INFO] 'q' zum Abbrechen.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Kein Frame gelesen – beende.")
                break

            frame = cv2.flip(frame, 1)

            now = time.time()
            seconds_left = max(0.0, phase_end - now)

            # Hand-Landmarks extrahieren
            frame, hands = tracker.process_frame(frame, draw_landmarks=True)

            # Aktuelles Label (bezieht sich auf den nächsten/aktuellen grünen Slot)
            label_text = _current_label(greens_done)

            # HUD
            hud_lines = [
                f"record_data | greens: {greens_done}/{greens_target}",
                "Taste 'q' zum Beenden",
            ]
            put_hud(frame, hud_lines)
            draw_square_with_timer(frame, phase.color, seconds_left)
            draw_label_text(frame, label_text)

            # Für Speicherung: verwende erste erkannte Hand (falls vorhanden)
            if hands:
                h0 = hands[0]
                # 21 Landmark-Tuples (x,y,z) – normierte Koordinaten
                lm_tuples = h0.coords_norm
            else:
                lm_tuples = [(np.nan, np.nan, np.nan)] * 21

            # Zeile bauen
            row = {
                "idx": idx,
                "timestamp": now,
                "square_color": phase.color,
                "label_text": label_text,
                "hand": hand_text,
            }
            # 21 Spalten lm_0 .. lm_20
            for i_lm in range(21):
                row[f"lm_{i_lm}"] = lm_tuples[i_lm]

            rows.append(row)
            idx += 1

            # Phase-Wechsel prüfen
            if now >= phase_end:
                if phase.color == "red":
                    # Wechsel zu grün (falls noch nicht 70 erreicht)
                    if greens_done < greens_target:
                        phase = Phase(color="green", duration_s=1.0)
                        phase_start = now
                        phase_end = phase_start + phase.duration_s
                    else:
                        # Fertig – wir haben alle grünen Fenster
                        break
                else:
                    # grüne Phase war aktiv → Zähler hoch
                    greens_done += 1
                    # danach 2s rot, außer wenn schon fertig
                    if greens_done >= greens_target:
                        break
                    phase = Phase(color="red", duration_s=2.0)
                    phase_start = now
                    phase_end = phase_start + phase.duration_s

            # Anzeige
            cv2.imshow("Recording – MediaPipe Hand Landmarks", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Aufnahme abgebrochen.")
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    # DataFrame erstellen und speichern
    if rows:
        df = pd.DataFrame(rows)
        # Index-Spalte auch als tatsächlichen Index setzen
        df.set_index("idx", inplace=True)
        df.to_pickle(out_path)
        print(f"[OK] Gespeichert: {out_path.resolve()}")
    else:
        print("[WARN] Keine Daten erfasst – nichts gespeichert.")
