# utils/phone_touch_control.py
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# pip install pynput
from pynput import keyboard


@dataclass
class TouchConfig:
    serial: Optional[str] = None
    adb_path: Optional[str] = None

    # Wiederholrate (wenn du Taste gedrückt hältst / schnell drückst)
    repeat_min_interval_s: float = 0.08

    # Swipe-Settings (werden dynamisch aus Screen-Size berechnet)
    anchor_x_ratio: float = 0.50   # Mittelpunkt X (0..1)
    anchor_y_ratio: float = 0.75   # Mittelpunkt Y (0..1) -> oft gut: eher unten im Spielfeld
    swipe_dx_ratio: float = 0.18   # Swipe-Weite horizontal
    swipe_dy_ratio: float = 0.18   # Swipe-Weite vertikal
    swipe_duration_ms: int = 80    # 60-120ms ist meist gut

    # Tap (z.B. Rotate)
    tap_x_ratio: float = 0.50
    tap_y_ratio: float = 0.55

    # Feinjustierung während Laufzeit (WASD verschiebt Anchor)
    nudge_px: int = 40


def _find_adb(adb_path: Optional[str] = None) -> str:
    if adb_path and os.path.exists(adb_path):
        return adb_path

    for env in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        root = os.environ.get(env)
        if root:
            cand = os.path.join(root, "platform-tools", "adb.exe" if os.name == "nt" else "adb")
            if os.path.exists(cand):
                return cand

    return "adb"


def _run(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out.strip()


def _list_devices(adb: str) -> List[Tuple[str, str]]:
    rc, out = _run([adb, "devices"])
    if rc != 0:
        raise RuntimeError(f"adb devices failed:\n{out}")

    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
    return rows


def _pick_default_serial(devs: List[Tuple[str, str]]) -> Optional[str]:
    real = [s for s, st in devs if st == "device" and not s.startswith("emulator-")]
    if real:
        return real[0]
    any_dev = [s for s, st in devs if st == "device"]
    if any_dev:
        return any_dev[0]
    return None


def _get_screen_size(adb: str, serial: str) -> Tuple[int, int]:
    # Expect something like:
    # Physical size: 1080x2400
    rc, out = _run([adb, "-s", serial, "shell", "wm", "size"])
    if rc != 0:
        raise RuntimeError(f"adb wm size failed:\n{out}")

    m = re.search(r"Physical size:\s*(\d+)x(\d+)", out)
    if not m:
        # fallback: sometimes only "Override size"
        m = re.search(r"Override size:\s*(\d+)x(\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse screen size from:\n{out}")

    w = int(m.group(1))
    h = int(m.group(2))
    return w, h


def adb_tap(adb: str, serial: str, x: int, y: int) -> None:
    subprocess.run(
        [adb, "-s", serial, "shell", "input", "tap", str(x), str(y)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def adb_swipe(adb: str, serial: str, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> None:
    subprocess.run(
        [adb, "-s", serial, "shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_phone_touch_control(cfg: TouchConfig = TouchConfig()) -> None:
    adb = _find_adb(cfg.adb_path)

    serial = cfg.serial or os.environ.get("ANDROID_SERIAL")
    devs = _list_devices(adb)

    if not serial:
        serial = _pick_default_serial(devs)

    if not serial:
        pretty = "\n".join([f"{s}\t{st}" for s, st in devs]) or "(no devices)"
        raise RuntimeError(
            "Kein nutzbares Android-Device gefunden.\n"
            "Check: USB-Debugging an, RSA-Dialog akzeptiert, adb devices zeigt 'device'.\n\n"
            f"adb devices:\n{pretty}"
        )

    st = dict(devs).get(serial, "unknown")
    if st != "device":
        raise RuntimeError(
            f"Device '{serial}' ist nicht im Status 'device' sondern '{st}'.\n"
            "Wenn 'unauthorized': am Handy RSA-Debugging-Popup bestätigen."
        )

    w, h = _get_screen_size(adb, serial)

    # compute initial coords
    anchor_x = int(w * cfg.anchor_x_ratio)
    anchor_y = int(h * cfg.anchor_y_ratio)
    tap_x = int(w * cfg.tap_x_ratio)
    tap_y = int(h * cfg.tap_y_ratio)

    dx = int(w * cfg.swipe_dx_ratio)
    dy = int(h * cfg.swipe_dy_ratio)

    print("\n=== Phone Touch Control (ADB) ===")
    print(f"ADB: {adb}")
    print(f"Device: {serial}")
    print(f"Screen: {w}x{h}")
    print("")
    print("Controls:")
    print("  Arrow keys  -> Swipe (left/right/up/down)")
    print("  Enter       -> Tap (rotate/click, je nach Game)")
    print("  WASD        -> verschiebt den Swipe-Anker live (damit du 'triffst')")
    print("  T           -> Tap-Test sofort auslösen")
    print("  Esc         -> Quit")
    print("")
    print("Tipp: Aktivier am Handy in Entwickleroptionen 'Berührungen anzeigen' + optional 'Zeigerposition',")
    print("      dann siehst du genau, wo die ADB-Taps/Swipes landen.\n")

    last_sent = {"LEFT": 0.0, "RIGHT": 0.0, "UP": 0.0, "DOWN": 0.0, "ENTER": 0.0}

    def _print_anchor():
        axr = anchor_x / w
        ayr = anchor_y / h
        print(f"[ANCHOR] x={anchor_x} y={anchor_y}  (ratios: {axr:.3f}, {ayr:.3f})")

    def _send_swipe(name: str):
        now = time.time()
        if (now - last_sent[name]) < cfg.repeat_min_interval_s:
            return
        last_sent[name] = now

        if name == "LEFT":
            adb_swipe(adb, serial, anchor_x + dx, anchor_y, anchor_x - dx, anchor_y, cfg.swipe_duration_ms)
        elif name == "RIGHT":
            adb_swipe(adb, serial, anchor_x - dx, anchor_y, anchor_x + dx, anchor_y, cfg.swipe_duration_ms)
        elif name == "UP":
            adb_swipe(adb, serial, anchor_x, anchor_y + dy, anchor_x, anchor_y - dy, cfg.swipe_duration_ms)
        elif name == "DOWN":
            adb_swipe(adb, serial, anchor_x, anchor_y - dy, anchor_x, anchor_y + dy, cfg.swipe_duration_ms)

    def _send_tap():
        now = time.time()
        if (now - last_sent["ENTER"]) < cfg.repeat_min_interval_s:
            return
        last_sent["ENTER"] = now
        adb_tap(adb, serial, tap_x, tap_y)

    def on_press(key):
        nonlocal anchor_x, anchor_y, tap_x, tap_y

        try:
            if key == keyboard.Key.left:
                _send_swipe("LEFT")
            elif key == keyboard.Key.right:
                _send_swipe("RIGHT")
            elif key == keyboard.Key.up:
                _send_swipe("UP")
            elif key == keyboard.Key.down:
                _send_swipe("DOWN")
            elif key == keyboard.Key.enter:
                _send_tap()
            elif key == keyboard.Key.esc:
                return False

            # Live-Nudging (WASD) + Tap-Test (T)
            else:
                if hasattr(key, "char") and key.char:
                    c = key.char.lower()
                    if c == "a":
                        anchor_x = max(0, anchor_x - cfg.nudge_px)
                        _print_anchor()
                    elif c == "d":
                        anchor_x = min(w - 1, anchor_x + cfg.nudge_px)
                        _print_anchor()
                    elif c == "w":
                        anchor_y = max(0, anchor_y - cfg.nudge_px)
                        _print_anchor()
                    elif c == "s":
                        anchor_y = min(h - 1, anchor_y + cfg.nudge_px)
                        _print_anchor()
                    elif c == "t":
                        print("[TEST] tap")
                        adb_tap(adb, serial, tap_x, tap_y)

        except Exception:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
