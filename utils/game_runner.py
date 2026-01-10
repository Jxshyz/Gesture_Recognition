# utils/game_runner.py
from __future__ import annotations

import webbrowser
from dataclasses import dataclass
from time import time
from typing import Optional

from utils.command_protocol import make_command, make_state
from utils.gesture_state_machine import GestureStateMachine, GSMConfig
from utils.live_predictor import run_live_predictions, PredictorConfig
from utils.ws_server import start_server_background, manager


@dataclass
class RunnerConfig:
    host: str = "127.0.0.1"
    port: int = 8010
    camera_index: int = 0

    # predictor
    pred_min_interval_s: float = 0.06  # ~16-17/s

    # FSM tuning
    arm_hold_s: float = 0.60
    cooldown_s: float = 0.50


def run_runner(cfg: RunnerConfig = RunnerConfig()) -> None:
    # 1) Server starten
    start_server_background(cfg.host, cfg.port)
    webbrowser.open(f"http://{cfg.host}:{cfg.port}/?v={int(time())}")

    # 2) FSM erstellen
    gsm = GestureStateMachine(
        GSMConfig(
            arm_hold_s=cfg.arm_hold_s,
            cooldown_s=cfg.cooldown_s,
        )
    )

    # 3) Prediction callback
    last_state_push = 0.0

    def on_pred(label: str, conf: float, ts: float):
        nonlocal last_state_push

        cmd, dbg = gsm.update(label, conf, ts)

        # state info ~20 Hz (damit Progressbar im Browser wirklich smooth ist)
        if (ts - last_state_push) >= 0.05:
            last_state_push = ts
            manager.broadcast_sync(
                make_state(
                    state=dbg.get("state", "IDLE"),
                    progress=float(dbg.get("arming_progress", 0.0)),
                    info={
                        "label": dbg.get("label", "-"),
                        "conf": float(dbg.get("conf", 0.0)),
                    },
                )
            )

        # command event (genau 1x)
        if cmd is not None:
            manager.broadcast_sync(
                make_command(
                    cmd,
                    meta={
                        "src_label": label,
                        "conf": float(conf),
                    },
                )
            )

    # 4) Live Predictor laufen lassen
    run_live_predictions(
        on_pred=on_pred,
        on_frame=None,
        cfg=PredictorConfig(
            camera_index=cfg.camera_index,
            pred_min_interval_s=cfg.pred_min_interval_s,
            draw_landmarks=False,
            show_window=False,
        ),
    )
