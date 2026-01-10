# utils/phone_controller_scrcpy.py
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import adbutils
import scrcpy
from scrcpy import const


@dataclass
class ScrcpyDeviceConfig:
    serial: Optional[str] = None
    # Video ist nicht zwingend nötig, aber Start muss laufen, damit Control zuverlässig geht
    max_width: int = 0
    bitrate: int = 8_000_000
    max_fps: int = 0
    flip: bool = False
    block_frame: bool = False
    stay_awake: bool = True
    lock_screen_orientation: int = -1  # -1 = keine Sperre


class AndroidDeviceScrcpy:
    """
    Android Device Controller über scrcpy control protocol:
    - echte Touch DOWN/MOVE/UP -> kohärentes Drag&Drop
    - keycode events
    - screen size & rotation polling (für Hoch-/Querformat)
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

    # ----------------------------
    # Display info (size/orientation)
    # ----------------------------
    def _shell(self, cmd: str) -> str:
        return self._device.shell(cmd)

    def refresh_display_info(self, force: bool = False, min_interval_s: float = 0.75):
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
        return self._client.control

    def keyevent(self, keycode: int):
        # scrcpy control keycode supports DOWN/UP; using DOWN is usually enough
        self.control.keycode(keycode, action=const.ACTION_DOWN)

    def touch_down(self, x: int, y: int, touch_id: int = 1):
        self.control.touch(int(x), int(y), action=const.ACTION_DOWN, touch_id=touch_id)

    def touch_move(self, x: int, y: int, touch_id: int = 1):
        self.control.touch(int(x), int(y), action=const.ACTION_MOVE, touch_id=touch_id)

    def touch_up(self, x: int, y: int, touch_id: int = 1):
        self.control.touch(int(x), int(y), action=const.ACTION_UP, touch_id=touch_id)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, step: int = 12, delay: float = 0.003):
        # stepwise swipe (smoother than adb input)
        self.control.swipe(int(x1), int(y1), int(x2), int(y2), move_step_length=step, move_steps_delay=delay)

    def tap(self, x: int, y: int, touch_id: int = 1, hold_s: float = 0.03):
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
