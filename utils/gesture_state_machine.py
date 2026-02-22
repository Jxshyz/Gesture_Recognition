"""
Finite State Machine (FSM) for robust gesture command triggering.

This module stabilizes noisy real-time gesture predictions by:

- Requiring an explicit arm gesture (e.g., fist) before actions
- Applying a majority vote over a sliding prediction window
- Enforcing cooldown periods after committed commands

The FSM ensures that gesture commands are deliberate,
stable, and emitted exactly once per activation cycle.
"""
# utils/gesture_state_machine.py
from __future__ import annotations
from dataclasses import dataclass
from collections import deque, Counter
from time import time
from typing import Deque, Dict, Optional, Tuple


@dataclass
class GSMConfig:
    """
    Configuration parameters for the GestureStateMachine.

    Attributes:
        arm_label (str):
            Label used to arm the system (e.g., "fist").

        arm_min_conf (float):
            Minimum confidence required to count as valid arm gesture.

        arm_hold_s (float):
            Required continuous hold time (seconds) before entering ARMED state.

        action_min_conf (float):
            Minimum confidence for action gestures to be considered.

        vote_window (int):
            Number of recent predictions used for majority voting.

        vote_min_majority (int):
            Minimum count required for a winning label inside the vote window.

        armed_timeout_s (float):
            Maximum duration (seconds) allowed in ARMED state without
            receiving a valid action gesture.

        cooldown_s (float):
            Cooldown duration (seconds) after emitting a command.

        label_to_cmd (Dict[str, str]):
            Mapping from model output labels to command strings.
    """

    # Arm gestures (e.g. fist)
    arm_label: str = "fist"
    arm_min_conf: float = 0.60
    arm_hold_s: float = 0.30  # hold fist time to get ARMED

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
    Gesture-based finite state machine with arming and cooldown logic.

    States:

        IDLE:
            Ignores all predictions except the configured arm gesture.
            Requires a continuous hold to transition to ARMED.

        ARMED:
            Waits for an action gesture.
            Applies majority voting over recent predictions.
            Emits exactly one command upon stable detection.

        COOLDOWN:
            Temporarily blocks new gestures after a committed command.
            Automatically returns to IDLE after cooldown expires.
    """

    IDLE = "IDLE"
    ARMED = "ARMED"
    COOLDOWN = "COOLDOWN"

    def __init__(self, cfg: GSMConfig = GSMConfig()):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        """
        Reset the FSM to its initial IDLE state.

        Clears timing variables, vote buffers, and cooldown state.
        """
        self.state = self.IDLE
        self._arm_t = 0.0
        self._last_update_t = time()
        self._armed_since = 0.0
        self._cooldown_until = 0.0
        self._votes: Deque[str] = deque(maxlen=self.cfg.vote_window)

    def update(
        self, label: str, conf: float, ts: Optional[float] = None
    ) -> Tuple[Optional[str], Dict]:
        """
        Update the FSM with a new prediction.

        This method processes a single model prediction and:

            - Advances state timing
            - Handles arming logic (hold detection)
            - Applies majority voting in ARMED state
            - Emits a command if a stable action is detected
            - Enforces cooldown logic

        Parameters:
            label (str):
                Predicted gesture label from the model.

            conf (float):
                Confidence score associated with the prediction.

            ts (Optional[float]):
                Timestamp of the prediction. If None, current system time is used.

        Returns:
            Tuple[Optional[str], Dict]:

                - cmd (Optional[str]):
                    Emitted command string (e.g., "LEFT", "JUMP"),
                    or None if no command was triggered.

                - debug (Dict):
                    Diagnostic information including:
                        - current state
                        - arming progress (0..1)
                        - current label and confidence
                        - vote buffer (if applicable)
        """
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
