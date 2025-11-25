import os
import json
import asyncio
import threading
from typing import Dict, Any, Set, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from .gesture_stream import get_latest_frame

app = FastAPI(title="Gesture Tetris")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend_tetris")
os.makedirs(FRONTEND_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        dead = []
        async with self.lock:
            for conn in self.active_connections:
                try:
                    await conn.send_text(json.dumps(message))
                except Exception:
                    dead.append(conn)
            for d in dead:
                self.active_connections.discard(d)


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        await manager.disconnect(ws)


@app.get("/video_feed")
async def video_feed():
    boundary = b"--frame"

    async def frame_gen():
        while True:
            frame = get_latest_frame()
            if frame is not None:
                yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.03)

    return StreamingResponse(
        frame_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


class GestureEvent(BaseModel):
    gesture: str
    confidence: float = 0.0
    params: Dict[str, Any] = {}


GESTURE_TO_ACTION = {
    "swipe_left": "left",
    "swipe_right": "right",
    "rotate": "rotate",
    "fist": "hardDrop",
}


@app.post("/gesture")
async def post_gesture(ev: GestureEvent):
    action = GESTURE_TO_ACTION.get(ev.gesture)
    payload = {
        "type": "gesture",
        "gesture": ev.gesture,
        "confidence": ev.confidence,
        "params": ev.params,
        "action": action,
    }
    await manager.broadcast(payload)
    return {"ok": True}


def run_tetris_server(host="127.0.0.1", port=8000):
    uvicorn.run(app, host=host, port=port, log_level="info")


def start_tetris_server_background(host="127.0.0.1", port=8000):
    def _run():
        uvicorn.run(app, host=host, port=port, log_level="info")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
