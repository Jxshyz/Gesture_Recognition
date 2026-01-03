# utils/phone_controller.py
from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


def _run(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out.strip()


def find_adb(explicit_path: Optional[str] = None) -> str:
    """
    Find adb in a cross-platform way:
    - explicit_path if provided and exists
    - ANDROID_SDK_ROOT / ANDROID_HOME
    - common default install paths (Windows/macOS/Linux)
    - fallback: 'adb' (must be in PATH)
    """
    if explicit_path:
        if Path(explicit_path).exists():
            return explicit_path

    env_roots = [os.environ.get("ANDROID_SDK_ROOT"), os.environ.get("ANDROID_HOME")]
    env_roots = [p for p in env_roots if p]

    cand = []
    exe = "adb.exe" if platform.system().lower().startswith("win") else "adb"

    for root in env_roots:
        cand.append(str(Path(root) / "platform-tools" / exe))

    # Common defaults
    home = Path.home()
    if platform.system().lower().startswith("win"):
        cand.append(str(home / "AppData" / "Local" / "Android" / "Sdk" / "platform-tools" / exe))
    elif platform.system().lower() == "darwin":
        cand.append(str(home / "Library" / "Android" / "sdk" / "platform-tools" / exe))
    else:
        cand.append(str(home / "Android" / "Sdk" / "platform-tools" / exe))

    for c in cand:
        if Path(c).exists():
            return c

    return "adb"


def list_devices(adb: str) -> List[Tuple[str, str]]:
    rc, out = _run([adb, "devices"])
    if rc != 0:
        raise RuntimeError(f"adb devices failed:\n{out}")

    devs: List[Tuple[str, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) >= 2:
            serial, state = parts[0], parts[1]
            devs.append((serial, state))
    return devs


def pick_device(adb: str, serial: Optional[str] = None) -> str:
    devs = list_devices(adb)
    if serial:
        for s, st in devs:
            if s == serial:
                if st != "device":
                    raise RuntimeError(f"Device {serial} not ready (state={st}). Authorize it on the phone.")
                return s
        raise RuntimeError(f"Device serial '{serial}' not found. adb devices:\n{devs}")

    ready = [s for s, st in devs if st == "device"]
    if not ready:
        raise RuntimeError(
            f"No authorized device found. adb devices:\n{devs}\n"
            f"-> On phone: accept USB debugging prompt (Allow)."
        )
    return ready[0]


def get_screen_size(adb: str, serial: str) -> Tuple[int, int]:
    rc, out = _run([adb, "-s", serial, "shell", "wm", "size"])
    if rc != 0:
        raise RuntimeError(f"wm size failed:\n{out}")

    # e.g. "Physical size: 1080x2400"
    m = re.search(r"(\d+)\s*x\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse wm size output:\n{out}")

    w = int(m.group(1))
    h = int(m.group(2))
    return w, h


@dataclass
class AndroidDevice:
    adb: str
    serial: str
    screen_w: int
    screen_h: int

    @classmethod
    def connect(cls, adb_path: Optional[str] = None, serial: Optional[str] = None) -> "AndroidDevice":
        adb = find_adb(adb_path)
        chosen = pick_device(adb, serial=serial)
        w, h = get_screen_size(adb, chosen)
        return cls(adb=adb, serial=chosen, screen_w=w, screen_h=h)

    def keyevent(self, keycode: int) -> None:
        _run([self.adb, "-s", self.serial, "shell", "input", "keyevent", str(keycode)])

    def tap(self, x: int, y: int) -> None:
        x = max(0, min(self.screen_w - 1, int(x)))
        y = max(0, min(self.screen_h - 1, int(y)))
        _run([self.adb, "-s", self.serial, "shell", "input", "tap", str(x), str(y)])

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 80) -> None:
        x1 = max(0, min(self.screen_w - 1, int(x1)))
        y1 = max(0, min(self.screen_h - 1, int(y1)))
        x2 = max(0, min(self.screen_w - 1, int(x2)))
        y2 = max(0, min(self.screen_h - 1, int(y2)))
        duration_ms = max(1, int(duration_ms))
        _run([self.adb, "-s", self.serial, "shell", "input", "swipe",
              str(x1), str(y1), str(x2), str(y2), str(duration_ms)])
