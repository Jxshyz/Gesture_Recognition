# utils/gesture_state_machine.py
from __future__ import annotations
from dataclasses import dataclass
from collections import deque, Counter
from time import time
from typing import Deque, Dict, Optional, Tuple


@dataclass
class GSMConfig:
    # Arm gestures (e.g. fist)
    arm_label: str = "fist"
    arm_min_conf: float = 0.60
    arm_hold_s: float = 0.30  # wie lange fist gehalten werden muss, um ARMED zu werden

    # Action gestures
    action_min_conf: float = 0.55
    vote_window: int = 5  # N Predictions for Majority Vote
    vote_min_majority: int = 3  # min counts for winner

    # Timing
    armed_timeout_s: float = 2.0  # if no Action -> back to IDLE
    cooldown_s: float = 0.50  # after Command

    # Map (Model-Labels -> Commands)
    label_to_cmd: Dict[str, str] = None

    def __post_init__(self):
        if self.label_to_cmd is None:
            self.label_to_cmd = {
                "swipe_left": "LEFT",
                "swipe_right": "RIGHT",
                "swipe_up": "JUMP",
                "swipe_down": "DUCK",
            }


class GestureStateMachine:
    """
    IDLE:
      - ignoriert alles außer Arm-Geste (fist)
    ARMED:
      - wartet auf Action-Geste, stabilisiert mit Majority-Vote
    COOLDOWN:
      - kurze Sperre nach Command
    """

    IDLE = "IDLE"
    ARMED = "ARMED"
    COOLDOWN = "COOLDOWN"

    def __init__(self, cfg: GSMConfig = GSMConfig()):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.state = self.IDLE
        self._arm_t = 0.0
        self._last_update_t = time()
        self._armed_since = 0.0
        self._cooldown_until = 0.0
        self._votes: Deque[str] = deque(maxlen=self.cfg.vote_window)

    def update(
        self, label: str, conf: float, ts: Optional[float] = None
    ) -> Tuple[Optional[str], Dict]:
        now = float(time() if ts is None else ts)
        dt = now - self._last_update_t
        if dt < 0:
            dt = 0.0
        self._last_update_t = now

        label = str(label or "-")
        conf = float(conf or 0.0)

        cmd = None

        # COOLDOWN
        if self.state == self.COOLDOWN:
            if now >= self._cooldown_until:
                self.state = self.IDLE
                self._arm_t = 0.0
                self._votes.clear()

        # IDLE -> ARMED (hold fist)
        if self.state == self.IDLE:
            if label == self.cfg.arm_label and conf >= self.cfg.arm_min_conf:
                self._arm_t += dt
            else:
                self._arm_t = 0.0

            progress = min(1.0, self._arm_t / max(1e-6, self.cfg.arm_hold_s))
            if self._arm_t >= self.cfg.arm_hold_s:
                self.state = self.ARMed = self.ARMED
                self._armed_since = now
                self._votes.clear()
                self._arm_t = 0.0

            debug = {
                "state": self.state,
                "arming_progress": progress if self.state == self.IDLE else 1.0,
                "label": label,
                "conf": conf,
            }
            return None, debug

        # ARMED: Majority Vote per Actions
        if self.state == self.ARMED:
            # Timeout
            if (now - self._armed_since) > self.cfg.armed_timeout_s:
                self.state = self.IDLE
                self._votes.clear()
                debug = {
                    "state": self.state,
                    "arming_progress": 0.0,
                    "label": label,
                    "conf": conf,
                }
                return None, debug

            # only vote Action-Labels
            if label in self.cfg.label_to_cmd and conf >= self.cfg.action_min_conf:
                self._votes.append(label)

            # decide, if there are enough votes
            if len(self._votes) == self.cfg.vote_window:
                c = Counter(self._votes)
                winner, n = c.most_common(1)[0]
                if n >= self.cfg.vote_min_majority:
                    cmd = self.cfg.label_to_cmd[winner]
                    # exactly 1 Command, then cooldown
                    self.state = self.COOLDOWN
                    self._cooldown_until = now + self.cfg.cooldown_s
                    self._votes.clear()

            debug = {
                "state": self.state,
                "arming_progress": 1.0,
                "label": label,
                "conf": conf,
                "votes": list(self._votes),
            }
            return cmd, debug

        # Fallback
        debug = {
            "state": self.state,
            "arming_progress": 0.0,
            "label": label,
            "conf": conf,
        }
        return None, debug
