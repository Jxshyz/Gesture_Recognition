from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SpotterConfig:
    rest_frames: int = 8
    end_frames: int = 6
    min_gesture_frames: int = 8
    max_gesture_frames: int = 45
    motion_on: float = 0.020
    motion_off: float = 0.010
    cooldown_frames: int = 12
    ema_alpha: float = 0.2


def normalize_landmarks(lm_xyz: np.ndarray) -> np.ndarray:
    """
    lm_xyz: (21,3) mediapipe normalized coords
    -> translation invariance (wrist) + scale invariance
    """
    wrist = lm_xyz[0]
    x = lm_xyz - wrist
    scale = np.linalg.norm(x[5] - x[17]) + 1e-6
    x = x / scale
    return x.reshape(-1)  # (63,)


class GestureSpotterFSM:
    IDLE, ARMED, RECORDING, COOLDOWN = "IDLE", "ARMED", "RECORDING", "COOLDOWN"

    def __init__(self, cfg: SpotterConfig = SpotterConfig()):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.state = self.IDLE
        self.prev = None
        self.motion_ema = 0.0
        self.rest_counter = 0
        self.end_counter = 0
        self.cooldown = 0
        self.buffer = []

    def update(self, lm_xyz: np.ndarray | None):
        """
        Returns: (segment_or_None, state, motion_ema)
        segment is np.ndarray shape (T,63)
        """
        if lm_xyz is None:
            # soft reset, keep state stable
            self.prev = None
            self.motion_ema = 0.0
            self.rest_counter = 0
            if self.state in (self.ARMED, self.RECORDING):
                self.state = self.IDLE
                self.buffer = []
            return None, self.state, self.motion_ema

        cur = normalize_landmarks(lm_xyz)

        if self.prev is None:
            self.prev = cur
            return None, self.state, self.motion_ema

        motion = float(np.mean(np.abs(cur - self.prev)))
        self.prev = cur
        a = self.cfg.ema_alpha
        self.motion_ema = (1 - a) * self.motion_ema + a * motion

        low_m = self.motion_ema < self.cfg.motion_off
        high_m = self.motion_ema > self.cfg.motion_on

        if self.state == self.COOLDOWN:
            self.cooldown -= 1
            if self.cooldown <= 0:
                self.state = self.IDLE
            return None, self.state, self.motion_ema

        if self.state == self.IDLE:
            self.rest_counter = self.rest_counter + 1 if low_m else 0
            if self.rest_counter >= self.cfg.rest_frames:
                self.state = self.ARMED
            return None, self.state, self.motion_ema

        if self.state == self.ARMED:
            if high_m:
                self.state = self.RECORDING
                self.buffer = [cur]
                self.end_counter = 0
            return None, self.state, self.motion_ema

        if self.state == self.RECORDING:
            self.buffer.append(cur)
            if low_m:
                self.end_counter += 1
            else:
                self.end_counter = 0

            T = len(self.buffer)
            should_end = (self.end_counter >= self.cfg.end_frames) or (
                T >= self.cfg.max_gesture_frames
            )

            if should_end:
                seg = np.asarray(self.buffer, dtype=np.float32)
                self.buffer = []
                self.state = self.COOLDOWN
                self.cooldown = self.cfg.cooldown_frames

                if T >= self.cfg.min_gesture_frames:
                    return seg, self.state, self.motion_ema
                return None, self.state, self.motion_ema

            return None, self.state, self.motion_ema

        return None, self.state, self.motion_ema
