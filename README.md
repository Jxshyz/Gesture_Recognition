# Gesture Recognition (MediaPipe) – Uni-Projekt

Einfaches Projekt zur **Erkennung von Handgesten** mit **MediaPipe Hands**.  
Wir erfassen zunächst Daten (21 Hand-Landmarks + Metadaten) in einem strukturierten Format.  
Später trainieren wir damit ein **neuronales Netz**, um Gesten in Echtzeit zu klassifizieren und als **Inputs** für eine Anwendung (z. B. Tetris) zu nutzen.

---

## Features (Phase 1–2)
- Live-Handtracking (21 Landmark-Punkte) mit MediaPipe
- Kameratest zum schnellen Check (`test_cam`)
- Datenerfassung mit visuellem Takt:
  - Viereck oben links: **Rot** → keine Geste, **Grün** → Geste ausführen
  - Start: **5 s Rot**, danach **(1 s Grün, 2 s Rot)** im Wechsel
  - **70** grüne Phasen → Aufnahme endet automatisch (~215 s)
  - Unter dem Viereck steht die aktuell gewünschte Geste (Label)

---

## Voraussetzungen
- Python 3.9–3.11 (empfohlen: 3.10+)
- Pakete:
  ```bash
  pip install mediapipe opencv-python numpy pandas


## Nutzung
1) Kamera testen

Zeigt das Kamerabild und zeichnet erkannte Hand-Landmarks.

Windows PowerShell aus dem Projekt-Root
  ```bash
  python .\main.py test_cam

_optional mit Kameraindex (z. B. externe Webcam)_
  ```bash
  python .\main.py test_cam 1

2) Daten aufnehmen

Startet den Rot/Grün-Takt, zeigt Timer & Gesten-Text, speichert Daten in ./data/Gestures_<Name>.pkl.

__Syntax: python .\main.py record_data <l|r> <Name> [camera_index]__
Beispiel (rechte Hand):
  ```bash
  python .\main.py record_data r Joschua
  
Beispiel (linke Hand, Kameraindex 1):
  ```bash
  python .\main.py record_data r Meric


Während der Aufnahme:

__Erste Phase__: 5 s Rot (keine Geste ausführen)

__Danach 70 Zyklen__: 1 s Grün (Geste ausführen) + 2 s Rot

__Unter dem Viereck steht die anzuzeigende Geste__ in Blöcken à 10:
1–10: Links wischen → 11–20: Rechts wischen → 21–30: nach oben wischen → 31–40: nach unten wischen → 41–50: faust schließen → 51–60: hand links drehen → 61–70: hand rechts drehen

q beendet jederzeit manuell

## Datensatz / Format

Datei: ./data/Gestures_<Name>.pkl (Pandas-DataFrame)

__Spalten__:
- idx (als Index gesetzt)
- timestamp (Sekunden, float)
- square_color ("red" oder "green")
- label_text (z. B. "Links wischen")
- hand ("links" oder "rechts", aus dem Aufruf l|r)
- lm_0 … lm_20 → jeweils ein (x, y, z)-Tuple in normalisierten Koordinaten (MediaPipe 0..1, z relativ)

Hinweise:
Frames ohne erkannte Hand enthalten NaN-Tuples, damit die Zeitreihe konsistent bleibt.