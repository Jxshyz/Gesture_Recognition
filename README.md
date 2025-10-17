# Gesture Recognition (MediaPipe)

Ein Projekt zur **Erkennung von Handgesten** mit **MediaPipe Hands**.  
Phase 1: Erfassung von Hand-Landmarks + Metadaten.  
Phase 2: Training eines **NN** (Fallback: **HMM**) zur **Echtzeit-Klassifikation** und **Steuerung** einer Anwendung (z. B. Tetris).

---

## Inhaltsverzeichnis

- [Features](#features)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Nutzung](#nutzung)
  - [1) Kamera testen](#1-kamera-testen)
  - [2) Daten aufnehmen](#2-daten-aufnehmen)
- [Datensatz / Format](#datensatz--format)
- [Tipps für Labels](#tipps-für-labels)
- [Changelog / Versionierung](#changelog--versionierung)
- [Lizenz / Datenschutz](#lizenz--datenschutz)

---

## Features

- Live-Handtracking (21 Landmark-Punkte) via **MediaPipe**
- **Kameratest** (`test_cam`)
- **Datenerfassung** mit visuellem Takt:
  - Viereck oben links: **Rot** → keine Geste, **Grün** → Geste ausführen
  - Start: **5 s Rot**, danach **1 s Grün / 2 s Rot** im Wechsel
  - **70** grüne Phasen → Aufnahme endet automatisch (≈ **215 s**)
  - Unter dem Viereck steht die aktuell gewünschte Geste (Label)

---

## Voraussetzungen

- **Python** 3.9–3.11 (empfohlen: 3.10+)
- Betriebssystem: Windows / macOS / Linux
- Kamera/Webcam

---

## Installation

```bash
# (optional) in ein virtuelles Environment wechseln
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Pakete installieren
pip install --upgrade pip
pip install mediapipe opencv-python numpy pandas
```

> Hinweis: Falls `opencv-python` auf Linux Probleme macht, verwende ggf. `opencv-python-headless`.

---

## Nutzung

### 1) Kamera testen

Zeigt das Kamerabild und zeichnet erkannte Hand-Landmarks.

**Windows (PowerShell)**
```bash
python .\main.py test_cam
```

**macOS/Linux (Bash/Zsh)**
```bash
python ./main.py test_cam
```

Optional mit Kameraindex (z. B. externe Webcam):
```bash
python ./main.py test_cam 1
```

---

### 2) Daten aufnehmen

Startet den Rot/Grün-Takt, zeigt Timer & Gesten-Text, speichert Daten in `./data/Gestures_<Name>.pkl`.

**Syntax**
```bash
python ./main.py record_data <l|r> <Name> [camera_index]
```

**Beispiele**
```bash
# Rechte Hand, Standardkamera
python ./main.py record_data r Joschua

# Linke Hand, Kameraindex 1
python ./main.py record_data l Meric 1
```

**Ablauf während der Aufnahme**
- **Erste Phase:** 5 s **Rot** (keine Geste ausführen)  
- **Danach 70 Zyklen:** 1 s **Grün** (Geste ausführen) + 2 s **Rot**

**Gesten-Reihenfolge** (Blöcke à 10), angezeigt unter dem Viereck:

| Zyklen | Anzeige-Label         |
|------:|------------------------|
| 1–10  | Links wischen          |
| 11–20 | Rechts wischen         |
| 21–30 | Nach oben wischen      |
| 31–40 | Nach unten wischen     |
| 41–50 | Faust schließen        |
| 51–60 | Hand links drehen      |
| 61–70 | Hand rechts drehen     |

**Abbruch:** `q` beendet jederzeit manuell.

---

## Datensatz / Format

**Datei:** `./data/Gestures_<Name>.pkl` (Pandas-DataFrame)

**Spalten**

| Spalte         | Typ        | Beschreibung                                                                 |
|----------------|------------|------------------------------------------------------------------------------|
| `idx`          | Index/int  | Laufender Index (als DataFrame-Index gesetzt)                                |
| `timestamp`    | float      | Sekunden (Monotonic/Wall-Clock, je nach Implementierung)                     |
| `square_color` | string     | `"red"` oder `"green"`                                                       |
| `label_text`   | string     | Mensch-lesbares Label (z. B. `"Links wischen"`)                              |
| `hand`         | string     | `"links"` oder `"rechts"` (aus CLI-Argument `l|r`)                           |
| `lm_0` … `lm_20` | tuple   | Jeweils `(x, y, z)` in **normalisierten Koordinaten** (MediaPipe 0..1, `z` relativ) |

**Hinweise**
- Frames **ohne erkannte Hand** → **NaN-Tuples** in `lm_*`, damit die Zeitreihe konsistent bleibt.
- FPS & Session-Metadaten (Teilnehmer-ID, Hände-Info, Licht/Ort, Gerät) in separater **JSON-Meta** ablegen.

---

## Tipps für Labels

Für Trainings-Pipelines zusätzlich maschinenlesbare Codes führen (z. B. `label_code` in `snake_case`):

```text
swipe_left, swipe_right, swipe_up, swipe_down, fist, rotate_left, rotate_right
```

Optional: numerische `label_id` für schnellere Verarbeitung.

---

## Changelog / Versionierung

- **Schema/Meta**: `schema_version`, `data_version`, `app_version` in Metadaten pflegen
- Beispiel-Änderungen dokumentieren (z. B. neue Gesten, neues Timing, Filterung)

---

## Lizenz / Datenschutz

- Speicherung nur technischer Handdaten; **kein Video** (wenn möglich)
- Einwilligung der Teilnehmenden, Anonymisierung (IDs), Zweckbindung
- Sichere Ablage (verschlüsselte Datenträger/Repos, Zugriffskontrolle)

---

### Quick-Reference (Cheatsheet)

```text
# Kamera
python ./main.py test_cam [camera_index]

# Aufnahme
python ./main.py record_data <l|r> <Name> [camera_index]

# Ablauf
5s Rot  → (1s Grün + 2s Rot) × 70 → Auto-Stopp
q = Abbruch
```
