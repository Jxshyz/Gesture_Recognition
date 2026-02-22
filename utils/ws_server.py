# utils/ws_server.py
from __future__ import annotations

import os
import json
import asyncio
import threading
from typing import Dict, Any, Set, Optional

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


def _frontend_dir() -> str:
    # utils/.. -> repo root
    return os.path.join(os.path.dirname(__file__), "..", "frontend_subway")


class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self.lock:
            self.active.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self.lock:
            self.active.discard(ws)

    async def broadcast(self, msg: Dict[str, Any]):
        dead = []
        data = json.dumps(msg)
        async with self.lock:
            for ws in list(self.active):
                try:
                    await ws.send_text(data)
                except Exception:
                    dead.append(ws)
            for d in dead:
                self.active.discard(d)

    def broadcast_sync(self, msg: Dict[str, Any]) -> None:
        """
        Thread-safe broadcast (aus eurer CV/ML Pipeline).
        """
        if self.loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(msg), self.loop)
        except Exception:
            pass


manager = ConnectionManager()

app = FastAPI(title="Gesture Runner")

FRONTEND_DIR = _frontend_dir()
os.makedirs(FRONTEND_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
async def _on_startup():
    manager.set_loop(asyncio.get_running_loop())


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # keep-alive / optional client pings
            await ws.receive_text()
    except Exception:
        await manager.disconnect(ws)


def start_server_background(
    host: str = "127.0.0.1", port: int = 8010
) -> threading.Thread:
    def _run():
        uvicorn.run(app, host=host, port=port, log_level="info")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
