# utils/phone_controller.py
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List


def _run(cmd: List[str], timeout: float = 8.0) -> Tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def _find_adb() -> str:
    # 1) explicit env
    for k in ("ADB_PATH", "ADB"):
        v = os.environ.get(k)
        if v and Path(v).exists():
            return v

    # 2) Android SDK env
    for k in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        root = os.environ.get(k)
        if root:
            cand = Path(root) / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
            if cand.exists():
                return str(cand)

    # 3) common Windows default
    if os.name == "nt":
        cand = Path.home() / "AppData" / "Local" / "Android" / "Sdk" / "platform-tools" / "adb.exe"
        if cand.exists():
            return str(cand)

    # 4) fallback: assume in PATH
    return "adb"


@dataclass
class AndroidDevice:
    adb_path: str
    serial: str

    # effective input coordinate system size (changes with rotation)
    input_w: int
    input_h: int
    rotation: int  # 0,1,2,3

    # physical size (optional)
    phys_w: int
    phys_h: int

    @classmethod
    def connect(cls, adb_path: Optional[str] = None, serial: Optional[str] = None) -> "AndroidDevice":
        adb = adb_path or _find_adb()

        # list devices
        rc, out = _run([adb, "devices"])
        if rc != 0:
            raise RuntimeError(f"adb devices failed:\n{out}")

        devs = []
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("List of devices"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] in ("device", "offline", "unauthorized", "authorizing"):
                devs.append((parts[0], parts[1]))

        if not devs:
            raise RuntimeError("Kein Android device gefunden (adb devices ist leer).")

        if serial is None:
            # pick first usable
            for s, st in devs:
                if st == "device":
                    serial = s
                    break
            if serial is None:
                raise RuntimeError(f"Kein 'device' state verfügbar. adb devices:\n{out}")

        dev = cls(adb_path=adb, serial=serial, input_w=0, input_h=0, rotation=0, phys_w=0, phys_h=0)
        dev.refresh_display_info(force=True)
        return dev

    def _base(self) -> List[str]:
        return [self.adb_path, "-s", self.serial]

    def shell(self, *args: str, timeout: float = 8.0) -> str:
        rc, out = _run(self._base() + ["shell", *args], timeout=timeout)
        if rc != 0:
            raise RuntimeError(f"adb shell {' '.join(args)} failed:\n{out}")
        return out

    def keyevent(self, keycode: int) -> None:
        self.shell("input", "keyevent", str(keycode))

    def tap(self, x: int, y: int) -> None:
        self.shell("input", "tap", str(int(x)), str(int(y)))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 220) -> None:
        self.shell("input", "swipe", str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(int(duration_ms)))

    def set_show_touches(self, enabled: bool) -> None:
        self.shell("settings", "put", "system", "show_touches", "1" if enabled else "0")

    def set_pointer_location(self, enabled: bool) -> None:
        self.shell("settings", "put", "system", "pointer_location", "1" if enabled else "0")

    def refresh_display_info(self, force: bool = False, min_interval_s: float = 0.8) -> None:
        now = time.time()
        last = getattr(self, "_last_disp_refresh", 0.0)
        if (not force) and (now - last) < min_interval_s:
            return
        self._last_disp_refresh = now

        # physical size
        phys_w, phys_h = self.phys_w, self.phys_h
        try:
            out = self.shell("wm", "size")
            m = re.search(r"Physical size:\s*(\d+)\s*x\s*(\d+)", out)
            if m:
                phys_w, phys_h = int(m.group(1)), int(m.group(2))
        except Exception:
            pass

        # rotation: try dumpsys input first
        rot = self.rotation
        try:
            out = self.shell("dumpsys", "input")
            m = re.search(r"SurfaceOrientation:\s*(\d+)", out)
            if m:
                rot = int(m.group(1)) % 4
        except Exception:
            pass

        # fallback: dumpsys display
        if rot is None:
            try:
                out = self.shell("dumpsys", "display")
                m = re.search(r"mCurrentRotation\s*=\s*(\d+)", out)
                if m:
                    rot = int(m.group(1)) % 4
            except Exception:
                rot = 0

        if rot is None:
            rot = 0

        # effective input size depends on rotation
        if rot in (1, 3):
            input_w, input_h = phys_h, phys_w
        else:
            input_w, input_h = phys_w, phys_h

        # store
        self.phys_w, self.phys_h = phys_w, phys_h
        self.rotation = rot
        self.input_w, self.input_h = input_w, input_h
