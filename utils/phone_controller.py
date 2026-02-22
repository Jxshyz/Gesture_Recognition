"""
ADB-based Android device controller.

This module provides a lightweight wrapper around:

- adb (Android Debug Bridge)
- optional uiautomator2 (for coherent touch events)

It supports:

- Device discovery and connection
- Key events
- Tap and swipe gestures
- True drag gestures (with uiautomator2)
- Display size and orientation polling

Designed for gesture-driven Android interaction.
"""
# utils/phone_controller.py
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Any


def _run(cmd: List[str], timeout: float = 8.0) -> Tuple[int, str]:
    """
    Execute a subprocess command and capture output.

    Parameters:
        cmd (List[str]): Command and arguments.
        timeout (float): Maximum execution time in seconds.

    Returns:
        Tuple[int, str]:
            (return_code, combined_stdout_stderr)
    """
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def _find_adb() -> str:
    """
    Locate the adb executable.

    Search order:
        1. Environment variables (ADB_PATH, ADB)
        2. ANDROID_SDK_ROOT / ANDROID_HOME
        3. Default Windows SDK location
        4. Fallback: assume 'adb' is in PATH

    Returns:
        str: Path to adb executable.
    """
    for k in ("ADB_PATH", "ADB"):
        v = os.environ.get(k)
        if v and Path(v).exists():
            return v

    for k in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        root = os.environ.get(k)
        if root:
            cand = (
                Path(root)
                / "platform-tools"
                / ("adb.exe" if os.name == "nt" else "adb")
            )
            if cand.exists():
                return str(cand)

    if os.name == "nt":
        cand = (
            Path.home()
            / "AppData"
            / "Local"
            / "Android"
            / "Sdk"
            / "platform-tools"
            / "adb.exe"
        )
        if cand.exists():
            return str(cand)

    return "adb"


@dataclass
class AndroidDevice:
    """
    Android device abstraction via adb and optional uiautomator2.

    Provides:

        - Basic ADB input actions (tap, swipe, keyevent)
        - Optional coherent touch events via uiautomator2
        - Drag gestures
        - Screen size and rotation handling
        - Orientation-aware coordinate space

    Attributes:
        adb_path (str): Path to adb executable.
        serial (str): Device serial identifier.
        input_w / input_h (int): Current logical input resolution.
        rotation (int): Display rotation (0–3).
        phys_w / phys_h (int): Physical display resolution.
        u2 (Any): Optional uiautomator2 device instance.
    """

    adb_path: str
    serial: str

    input_w: int
    input_h: int
    rotation: int  # 0,1,2,3

    phys_w: int
    phys_h: int

    u2: Any = field(default=None, repr=False)

    @property
    def has_u2(self) -> bool:
        return self.u2 is not None

    @classmethod
    def connect(
        cls,
        adb_path: Optional[str] = None,
        serial: Optional[str] = None,
        enable_u2: bool = False,
    ) -> "AndroidDevice":
        """
        Connect to an Android device via adb.

        Steps:
            - Locate adb executable
            - Parse 'adb devices' output
            - Select appropriate device
            - Query display information
            - Optionally connect via uiautomator2

        Parameters:
            adb_path (Optional[str]):
                Explicit adb path (otherwise auto-detected).

            serial (Optional[str]):
                Specific device serial.

            enable_u2 (bool):
                If True, attempt uiautomator2 connection
                for coherent touch gestures.

        Returns:
            AndroidDevice: Connected device instance.

        Raises:
            RuntimeError: If no suitable device is found.
        """
        adb = adb_path or _find_adb()

        rc, out = _run([adb, "devices"])
        if rc != 0:
            raise RuntimeError(f"adb devices failed:\n{out}")

        devs = []
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("List of devices"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] in (
                "device",
                "offline",
                "unauthorized",
                "authorizing",
            ):
                devs.append((parts[0], parts[1]))

        if not devs:
            raise RuntimeError("Kein Android device gefunden (adb devices ist leer).")

        if serial is None:
            for s, st in devs:
                if st == "device":
                    serial = s
                    break
            if serial is None:
                raise RuntimeError(
                    f"Kein 'device' state verfügbar. adb devices:\n{out}"
                )

        dev = cls(
            adb_path=adb,
            serial=serial,
            input_w=0,
            input_h=0,
            rotation=0,
            phys_w=0,
            phys_h=0,
            u2=None,
        )
        dev.refresh_display_info(force=True)

        if enable_u2:
            try:
                import uiautomator2 as u2

                dev.u2 = u2.connect(serial)
                _ = dev.u2.info
            except Exception as e:
                dev.u2 = None
                raise RuntimeError(
                    "uiautomator2 konnte nicht verbinden. Prüfe USB-Debugging + 'adb devices'.\n"
                    f"Original error: {e}"
                ) from e

        return dev

    def _base(self) -> List[str]:
        return [self.adb_path, "-s", self.serial]

    def shell(self, *args: str, timeout: float = 8.0) -> str:
        """
        Execute an adb shell command.

        Parameters:
            *args (str): Shell command arguments.
            timeout (float): Command timeout in seconds.

        Returns:
            str: Command output.

        Raises:
            RuntimeError: If the command fails.
        """
        rc, out = _run(self._base() + ["shell", *args], timeout=timeout)
        if rc != 0:
            raise RuntimeError(f"adb shell {' '.join(args)} failed:\n{out}")
        return out

    # ----------------------------
    # Basic actions (ADB)
    # ----------------------------
    def keyevent(self, keycode: int) -> None:
        self.shell("input", "keyevent", str(keycode))

    def tap(self, x: int, y: int) -> None:
        self.shell("input", "tap", str(int(x)), str(int(y)))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 220) -> None:
        self.shell(
            "input",
            "swipe",
            str(int(x1)),
            str(int(y1)),
            str(int(x2)),
            str(int(y2)),
            str(int(duration_ms)),
        )

    def set_show_touches(self, enabled: bool) -> None:
        self.shell("settings", "put", "system", "show_touches", "1" if enabled else "0")

    def set_pointer_location(self, enabled: bool) -> None:
        self.shell(
            "settings", "put", "system", "pointer_location", "1" if enabled else "0"
        )

    # ----------------------------
    # Cohesive touch (uiautomator2)
    # ----------------------------
    def touch_down(self, x: int, y: int) -> None:
        if not self.u2:
            raise RuntimeError(
                "touch_down benötigt uiautomator2 (connect(enable_u2=True))."
            )
        self.u2.touch.down(int(x), int(y))

    def touch_move(self, x: int, y: int) -> None:
        if not self.u2:
            raise RuntimeError(
                "touch_move benötigt uiautomator2 (connect(enable_u2=True))."
            )
        self.u2.touch.move(int(x), int(y))

    def touch_up(self, x: int, y: int) -> None:
        if not self.u2:
            raise RuntimeError(
                "touch_up benötigt uiautomator2 (connect(enable_u2=True))."
            )
        self.u2.touch.up(int(x), int(y))

    def drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_s: float = 0.22,
        steps: int = 12,
    ) -> None:
        """
        Perform a drag gesture.

        Behavior:
            - If uiautomator2 is available:
                  Simulates coherent DOWN/MOVE/UP sequence.
            - Otherwise:
                  Falls back to adb 'input swipe'.

        Parameters:
            x1, y1 (int): Start coordinates.
            x2, y2 (int): End coordinates.
            duration_s (float): Total drag duration.
            steps (int): Number of interpolation steps.
        """
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        steps = max(2, int(steps))
        duration_s = max(0.02, float(duration_s))

        if self.u2:
            self.touch_down(x1, y1)
            for i in range(1, steps):
                t = i / (steps - 1)
                xi = int(x1 + (x2 - x1) * t)
                yi = int(y1 + (y2 - y1) * t)
                self.touch_move(xi, yi)
                time.sleep(duration_s / steps)
            self.touch_up(x2, y2)
        else:
            self.swipe(x1, y1, x2, y2, duration_ms=int(duration_s * 1000))

    def refresh_display_info(
        self, force: bool = False, min_interval_s: float = 0.8
    ) -> None:
        """
        Refresh device display information.

        Queries:
            - Physical resolution via 'wm size'
            - Display rotation via 'dumpsys input'

        Automatically adjusts input coordinate space for
        landscape rotations.

        Parameters:
            force (bool): Force refresh regardless of timing.
            min_interval_s (float): Minimum interval between refreshes.
        """
        now = time.time()
        last = getattr(self, "_last_disp_refresh", 0.0)
        if (not force) and (now - last) < min_interval_s:
            return
        self._last_disp_refresh = now

        phys_w, phys_h = self.phys_w, self.phys_h
        try:
            out = self.shell("wm", "size")
            m = re.search(r"Physical size:\s*(\d+)\s*x\s*(\d+)", out)
            if m:
                phys_w, phys_h = int(m.group(1)), int(m.group(2))
        except Exception:
            pass

        rot = self.rotation
        try:
            out = self.shell("dumpsys", "input")
            m = re.search(r"SurfaceOrientation:\s*(\d+)", out)
            if m:
                rot = int(m.group(1)) % 4
        except Exception:
            rot = 0

        if rot in (1, 3):
            input_w, input_h = phys_h, phys_w
        else:
            input_w, input_h = phys_w, phys_h

        self.phys_w, self.phys_h = phys_w, phys_h
        self.rotation = rot
        self.input_w, self.input_h = input_w, input_h
