"""
Android device control via scrcpy control protocol.

This module provides a high-level wrapper around:

- adbutils (device access)
- scrcpy (video + control socket)

It enables:

- Touch events (DOWN / MOVE / UP)
- Smooth swipes
- Tap gestures
- Key events
- Display size and rotation polling

Designed for gesture-driven remote phone control.
"""
# utils/phone_controller_scrcpy.py
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

import adbutils
import scrcpy
from scrcpy import const


@dataclass
class ScrcpyDeviceConfig:
    """
    Configuration for connecting to an Android device via scrcpy.

    Attributes:
        serial (Optional[str]):
            Specific device serial to connect to.
            If None, the first available device is used.

        max_width (int):
            Maximum video width for scrcpy stream (0 = default).

        bitrate (int):
            Video bitrate for scrcpy.

        max_fps (int):
            Maximum frames per second (0 = default).

        flip (bool):
            Whether to horizontally flip the video stream.

        block_frame (bool):
            Whether frame processing should block.

        stay_awake (bool):
            Keep device screen awake while connected.

        lock_screen_orientation (int):
            Screen orientation lock (-1 = no lock).
    """

    serial: Optional[str] = None
    # Video is not strictly necessary, but Start must be running for Control to function reliably
    max_width: int = 0
    bitrate: int = 8_000_000
    max_fps: int = 0
    flip: bool = False
    block_frame: bool = False
    stay_awake: bool = True
    lock_screen_orientation: int = -1  # -1 = no lock


class AndroidDeviceScrcpy:
    """
    High-level Android device controller using scrcpy.

    Features:
        - True touch events (DOWN / MOVE / UP)
        - Smooth swipe gestures
        - Tap helper
        - Key event injection
        - Screen size and rotation detection
        - Orientation-aware coordinate handling

    Intended for real-time gesture → phone control pipelines.
    """

    def __init__(self, device: adbutils.AdbDevice, client: scrcpy.Client):
        self._device = device
        self._client = client
        self.serial = device.serial

        self.screen_w = 0
        self.screen_h = 0
        self.rotation = 0  # 0,1,2,3 (optional)
        self._last_refresh_t = 0.0

        self.refresh_display_info(force=True)

    # ----------------------------
    # Connect / lifecycle
    # ----------------------------
    @classmethod
    def connect(cls, cfg: ScrcpyDeviceConfig) -> "AndroidDeviceScrcpy":
        """
        Connect to an Android device and start a scrcpy client.

        Device selection logic:
            - If serial is provided, use it.
            - Otherwise select the first available device.
            - Prefer physical devices over emulators.

        Starts scrcpy in a background thread so the caller
        can continue running its own loop.

        Parameters:
            cfg (ScrcpyDeviceConfig): Connection configuration.

        Returns:
            AndroidDeviceScrcpy: Initialized controller instance.

        Raises:
            RuntimeError: If no device is found.
        """
        # choose device
        if cfg.serial:
            dev = adbutils.adb.device(cfg.serial)
        else:
            devs = adbutils.adb.device_list()
            devs_ok = []
            for d in devs:
                try:
                    # state check
                    _ = d.get_state()
                    devs_ok.append(d)
                except Exception:
                    pass

            if not devs_ok:
                raise RuntimeError("Kein Android device gefunden (adb devices leer).")

            # Prefer physical device over emulator if possible
            phys = [d for d in devs_ok if not str(d.serial).startswith("emulator-")]
            dev = phys[0] if phys else devs_ok[0]

        client = scrcpy.Client(
            device=dev,
            max_width=cfg.max_width,
            bitrate=cfg.bitrate,
            max_fps=cfg.max_fps,
            flip=cfg.flip,
            block_frame=cfg.block_frame,
            stay_awake=cfg.stay_awake,
            lock_screen_orientation=cfg.lock_screen_orientation,
        )

        # Start in background thread so your loop can run
        client.start(threaded=True, daemon_threaded=True)

        # small settle time so control socket is ready
        time.sleep(0.25)

        return cls(device=dev, client=client)

    def close(self):
        try:
            self._client.stop()
        except Exception:
            pass

    def _shell(self, cmd: str) -> str:
        return self._device.shell(cmd)

    def refresh_display_info(self, force: bool = False, min_interval_s: float = 0.75):
        """
        Refresh screen resolution and orientation.

        Uses:
            - 'wm size' to obtain physical resolution
            - 'dumpsys input' to obtain surface rotation

        Automatically swaps width/height for landscape rotations.

        Parameters:
            force (bool):
                If True, refresh immediately.

            min_interval_s (float):
                Minimum time between refresh calls (throttling).
        """
        now = time.time()
        if not force and (now - self._last_refresh_t) < min_interval_s:
            return
        self._last_refresh_t = now

        # Size
        out = self._shell("wm size")
        # examples:
        # Physical size: 1080x2400
        # Override size: 1080x2400
        m = re.search(r"(Physical|Override)\s+size:\s*(\d+)x(\d+)", out)
        if m:
            w = int(m.group(2))
            h = int(m.group(3))
        else:
            # fallback
            w, h = (1080, 2400)

        # Rotation (best-effort)
        rot = 0
        out2 = self._shell("dumpsys input")
        m2 = re.search(r"SurfaceOrientation:\s*(\d)", out2)
        if m2:
            rot = int(m2.group(1))

        # If rotated 90/270 degrees, coordinate space becomes landscape
        if rot in (1, 3):
            self.screen_w, self.screen_h = h, w
        else:
            self.screen_w, self.screen_h = w, h
        self.rotation = rot

    # ----------------------------
    # Controls
    # ----------------------------
    @property
    def control(self):
        """
        Access the underlying scrcpy control interface.
        """
        return self._client.control

    def keyevent(self, keycode: int):
        """
        Send a key event to the device.

        Parameters:
            keycode (int): Android keycode constant.
        """
        # scrcpy control keycode supports DOWN/UP; using DOWN is usually enough
        self.control.keycode(keycode, action=const.ACTION_DOWN)

    def touch_down(self, x: int, y: int, touch_id: int = 1):
        self.control.touch(int(x), int(y), action=const.ACTION_DOWN, touch_id=touch_id)

    def touch_move(self, x: int, y: int, touch_id: int = 1):
        self.control.touch(int(x), int(y), action=const.ACTION_MOVE, touch_id=touch_id)

    def touch_up(self, x: int, y: int, touch_id: int = 1):
        self.control.touch(int(x), int(y), action=const.ACTION_UP, touch_id=touch_id)

    def swipe(
        self, x1: int, y1: int, x2: int, y2: int, step: int = 12, delay: float = 0.003
    ):
        """
        Perform a smooth swipe gesture.

        Parameters:
            x1, y1 (int): Start coordinates.
            x2, y2 (int): End coordinates.
            step (int): Step length for movement interpolation.
            delay (float): Delay between steps (seconds).
        """
        # stepwise swipe (smoother than adb input)
        self.control.swipe(
            int(x1),
            int(y1),
            int(x2),
            int(y2),
            move_step_length=step,
            move_steps_delay=delay,
        )

    def tap(self, x: int, y: int, touch_id: int = 1, hold_s: float = 0.03):
        """
        Perform a tap gesture at (x, y).

        Parameters:
            x, y (int): Coordinates.
            touch_id (int): Touch identifier.
            hold_s (float): Hold duration before release.
        """
        self.touch_down(x, y, touch_id=touch_id)
        time.sleep(max(0.0, hold_s))
        self.touch_up(x, y, touch_id=touch_id)

    # convenience swipes
    def swipe_left(self):
        self.refresh_display_info()
        y = int(self.screen_h * 0.50)
        self.swipe(int(self.screen_w * 0.85), y, int(self.screen_w * 0.15), y)

    def swipe_right(self):
        self.refresh_display_info()
        y = int(self.screen_h * 0.50)
        self.swipe(int(self.screen_w * 0.15), y, int(self.screen_w * 0.85), y)
