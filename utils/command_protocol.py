# utils/command_protocol.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from time import time
from typing import Dict, Any, Optional


@dataclass
class CommandMsg:
    type: str  # "command"
    cmd: str   # "LEFT"|"RIGHT"|"JUMP"|"DUCK"|"RESTART"...
    ts: float
    meta: Dict[str, Any]


@dataclass
class StateMsg:
    type: str  # "state"
    state: str  # "IDLE"|"ARMED"|"CLASSIFICATION"|"COOLDOWN"
    progress: float  # 0..1 (arming progress)
    ts: float
    info: Dict[str, Any]


def make_command(cmd: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return asdict(
        CommandMsg(
            type="command",
            cmd=str(cmd),
            ts=float(time()),
            meta=dict(meta or {}),
        )
    )


def make_state(state: str, progress: float, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = float(progress)
    if p < 0:
        p = 0.0
    if p > 1:
        p = 1.0
    return asdict(
        StateMsg(
            type="state",
            state=str(state),
            progress=p,
            ts=float(time()),
            info=dict(info or {}),
        )
    )
