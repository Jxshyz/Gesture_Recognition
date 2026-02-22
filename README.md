# Gesture Recognition (MediaPipe)

A project for **hand gesture recognition** using **MediaPipe Hands**.  
Phase 1: Capture hand landmarks + metadata.  
Phase 2: Train a **NN** (fallback: **HMM**) for **real-time classification** and **application control** (e.g., Tetris).

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1) Test Camera](#1-test-camera)
  - [2) Record Data](#2-record-data)
- [Dataset / Format](#dataset--format)
- [License / Privacy](#license--privacy)
- [Quick Reference (Cheatsheet)](#quick-reference-cheatsheet)

---

## Features

- Live hand tracking (21 landmark points) via **MediaPipe**
- **Camera test** (`test_cam`)
- **Data recording** with visual timing:
  - Top-left square: **Red** → no gesture, **Green** → perform gesture
  - Start: **5 s Red**, then **1 s Green / 2 s Red** alternating
  - **70** green phases → recording ends automatically (≈ **215 s**)
  - The currently requested gesture (label) is displayed below the square

---

## Requirements

- **Python** 3.9–3.11 (recommended: 3.10+)
- Operating System: Windows / macOS / Linux
- Camera/Webcam

---

## Installation

```bash
# (optional) create and activate a virtual environment
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# install packages
pip install --upgrade pip
pip install mediapipe opencv-python numpy pandas
```

> Note: If `opencv-python` causes issues on Linux, use `opencv-python-headless` if necessary.

---

## Usage

### 1) Test Camera

Displays the camera feed and draws detected hand landmarks.

**Windows (PowerShell)**
```bash
python .\main.py test_cam
```

**macOS/Linux (Bash/Zsh)**
```bash
python ./main.py test_cam
```

Optional with camera index (e.g., external webcam):
```bash
python ./main.py test_cam 1
```

---

### 2) Record Data

Starts the red/green timing sequence, shows timer & gesture text, saves data to `./data/Gestures_<Name>.pkl`.

**Syntax**
```bash
python ./main.py record_data <l|r> <Name> [camera_index]
```

**Examples**
```bash
# Right hand, default camera
python ./main.py record_data r Joschua

# Left hand, camera index 1
python ./main.py record_data l Meric 1
```

**Recording Procedure**
- **Initial phase:** 5 s **Red** (do not perform gesture)  
- **Then 70 cycles:** 1 s **Green** (perform gesture) + 2 s **Red**

**Gesture Order** (blocks of 10), displayed below the square:

| Cycles | Display Label        |
|-------:|----------------------|
| 1–10   | Swipe left           |
| 11–20  | Swipe right          |
| 21–30  | Swipe up             |
| 31–40  | Swipe down           |
| 41–50  | Close fist           |
| 51–60  | Rotate hand left     |
| 61–70  | Rotate hand right    |

**Abort:** Press `q` to stop manually at any time.

---

## Dataset / Format

**File:** `./data/Gestures_<Name>.pkl` (Pandas DataFrame)

**Columns**

| Column          | Type        | Description                                                                 |
|-----------------|------------|-----------------------------------------------------------------------------|
| `idx`           | Index/int  | Sequential index (set as DataFrame index)                                   |
| `timestamp`     | float      | Seconds (monotonic/wall-clock, depending on implementation)                  |
| `square_color`  | string     | `"red"` or `"green"`                                                         |
| `label_text`    | string     | Human-readable label (e.g., `"Swipe left"`)                                  |
| `hand`          | string     | `"left"` or `"right"` (from CLI argument `l|r`)                              |
| `lm_0` … `lm_20` | tuple     | Each `(x, y, z)` in **normalized coordinates** (MediaPipe 0..1, `z` relative) |

**Notes**
- Frames **without detected hand** → **NaN tuples** in `lm_*` to keep the time series consistent.
- FPS & session metadata (participant ID, hand info, lighting/location, device) should be stored in a separate **JSON meta file**.

---

## License / Privacy

- Store technical hand data only; **no video** (if possible)
- Participant consent, anonymization (IDs), purpose limitation

---

## Quick Reference (Cheatsheet)

```text
# Camera
python ./main.py test_cam [camera_index]

# Recording
python ./main.py record_data <l|r> <Name> [camera_index]

# Procedure
5s Red → (1s Green + 2s Red) × 70 → Auto-stop
q = abort
```
