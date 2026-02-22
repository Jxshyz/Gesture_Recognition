"""
Keyboard-to-Android controller via ADB.

This module allows controlling an Android device using
local keyboard input. It listens for arrow key and enter
events and sends corresponding Android DPAD keycodes
via adb.

Designed for:

    - Manual testing
    - Debugging gesture pipelines
    - Simple remote navigation control

Press ESC to exit.
"""
# utils/phone_keyboard_control.py
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

# pip install pynput
from pynput import keyboard


# Android KeyEvent Codes
KEYCODES = {
    "LEFT": 21,  # KEYCODE_DPAD_LEFT
    "RIGHT": 22,  # KEYCODE_DPAD_RIGHT
    "UP": 19,  # KEYCODE_DPAD_UP
    "DOWN": 20,  # KEYCODE_DPAD_DOWN
    "ENTER": 66,  # KEYCODE_ENTER
    # Alternative for "select": 23 (KEYCODE_DPAD_CENTER)
}


@dataclass
class PhoneKeyConfig:
    """
    Configuration for keyboard-based Android control.

    Attributes:
        serial (Optional[str]):
            Target Android device serial.
            If None, auto-detection is attempted.

        adb_path (Optional[str]):
            Explicit path to adb executable.
            If None, auto-detection is used.

        repeat_min_interval_s (float):
            Minimum time between repeated key events
            to prevent rapid key spamming.
    """

    serial: Optional[str] = None
    adb_path: Optional[str] = None
    repeat_min_interval_s: float = 0.05  # schützt vor Key-Spam


def _find_adb(adb_path: Optional[str] = None) -> str:
    """
    Locate the adb executable.

    Search order:
        1. Explicit adb_path argument
        2. ANDROID_SDK_ROOT / ANDROID_HOME
        3. Fallback: assume 'adb' is available in PATH

    Returns:
        str: Path to adb executable.
    """
    if adb_path and os.path.exists(adb_path):
        return adb_path

    # Try env vars
    for env in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        root = os.environ.get(env)
        if root:
            cand = os.path.join(
                root, "platform-tools", "adb.exe" if os.name == "nt" else "adb"
            )
            if os.path.exists(cand):
                return cand

    # fallback: adb in PATH
    return "adb"


def _run(cmd: List[str]) -> Tuple[int, str]:
    """
    Execute a subprocess command and capture output.

    Parameters:
        cmd (List[str]): Command and arguments.

    Returns:
        Tuple[int, str]:
            (return_code, combined_stdout_stderr_output)
    """
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out.strip()


def _list_devices(adb: str) -> List[Tuple[str, str]]:
    """
    List connected Android devices via adb.

    Parameters:
        adb (str): Path to adb executable.

    Returns:
        List[Tuple[str, str]]:
            List of (serial, status) pairs.
            Status is typically: device, unauthorized, offline.
    """
    rc, out = _run([adb, "devices"])
    if rc != 0:
        raise RuntimeError(f"adb devices failed:\n{out}")

    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    # first line: "List of devices attached"
    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
    return rows


def _pick_default_serial(devs: List[Tuple[str, str]]) -> Optional[str]:
    # prefer real device (not emulator) with status=device
    real = [s for s, st in devs if st == "device" and not s.startswith("emulator-")]
    if real:
        return real[0]
    any_dev = [s for s, st in devs if st == "device"]
    if any_dev:
        return any_dev[0]
    return None


def adb_keyevent(adb: str, serial: str, keycode: int) -> None:
    subprocess.run(
        [adb, "-s", serial, "shell", "input", "keyevent", str(keycode)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_phone_keyboard_control(cfg: PhoneKeyConfig = PhoneKeyConfig()) -> None:
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

    # sanity check
    if serial not in [s for s, _ in devs]:
        pretty = "\n".join([f"{s}\t{st}" for s, st in devs]) or "(no devices)"
        raise RuntimeError(
            f"Serial '{serial}' nicht in adb devices.\n\nadb devices:\n{pretty}"
        )

    st = dict(devs).get(serial, "unknown")
    if st != "device":
        raise RuntimeError(
            f"Device '{serial}' ist nicht im Status 'device' sondern '{st}'.\n"
            "Wenn 'unauthorized': am Handy RSA-Debugging-Popup bestätigen."
        )

    print("\n=== Phone Keyboard Control (ADB) ===")
    print(f"ADB: {adb}")
    print(f"Device: {serial}")
    print("Controls: Arrow keys = DPAD, Enter = ENTER, Esc = Quit\n")

    last_sent = {"LEFT": 0.0, "RIGHT": 0.0, "UP": 0.0, "DOWN": 0.0, "ENTER": 0.0}

    def _send(name: str):
        now = time.time()
        if (now - last_sent[name]) < cfg.repeat_min_interval_s:
            return
        last_sent[name] = now
        adb_keyevent(adb, serial, KEYCODES[name])

    def on_press(key):
        try:
            if key == keyboard.Key.left:
                _send("LEFT")
            elif key == keyboard.Key.right:
                _send("RIGHT")
            elif key == keyboard.Key.up:
                _send("UP")
            elif key == keyboard.Key.down:
                _send("DOWN")
            elif key == keyboard.Key.enter:
                _send("ENTER")
            elif key == keyboard.Key.esc:
                # stop listener
                return False
        except Exception:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
