import os
import json
import asyncio
import threading
import time
from typing import Dict, Any, Set

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from .gesture_stream import get_latest_frame
from .telemetry_store import TELEMETRY
from .highscore_store import HighscoreStore

app = FastAPI(title="Gesture Tetris")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend_tetris")
FRONTEND_DIR = os.path.abspath(FRONTEND_DIR)
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Highscore folder in project root: ./Highscore_Tetris
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HIGHSCORE_DIR = os.path.join(PROJECT_ROOT, "Highscore_Tetris")
highscores = HighscoreStore(root_dir=os.path.abspath(HIGHSCORE_DIR))

# static assets
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def _no_cache_headers() -> Dict[str, str]:
    return {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }


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
            self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        dead = []
        msg_txt = json.dumps(message)
        async with self.lock:
            for conn in self.active_connections:
                try:
                    await conn.send_text(msg_txt)
                except Exception:
                    dead.append(conn)
            for d in dead:
                self.active_connections.discard(d)


manager = ConnectionManager()


@app.get("/__debug_index_path")
async def debug_index_path():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    return JSONResponse(
        {
            "FRONTEND_DIR": FRONTEND_DIR,
            "index_path": index_path,
            "exists": os.path.exists(index_path),
        },
        headers=_no_cache_headers(),
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    """
    Reload index.html on EVERY request (no cache)
    """
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            f"<h1>index.html nicht gefunden</h1><p>Erwartet unter: {index_path}</p>",
            status_code=500,
            headers=_no_cache_headers(),
        )

    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()

    return HTMLResponse(html, headers=_no_cache_headers())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)

    # Direct initial push: Telemetry + high scores
    try:
        await ws.send_text(
            json.dumps({"type": "telemetry", "data": TELEMETRY.snapshot()})
        )
        await ws.send_text(
            json.dumps({"type": "highscores", "items": highscores.list_highscores()})
        )
    except Exception:
        pass

    try:
        while True:
            # keep-alive
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
                yield (
                    boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            await asyncio.sleep(0.03)

    return StreamingResponse(
        frame_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=_no_cache_headers(),
    )


class GestureEvent(BaseModel):
    gesture: str
    confidence: float = 0.0
    params: Dict[str, Any] = {}


GESTURE_TO_ACTION = {
    "swipe_left": "left",
    "swipe_right": "right",
    "rotate_left": "rotate",
    "close_fist": "hardDrop",
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


class TelemetryIn(BaseModel):
    state: str
    label: str
    conf: float = 0.0
    seconds_left: float = 0.0
    armed_progress: float = 0.0
    armed_ready: bool = False
    push_history: bool = False


@app.post("/api/telemetry")
async def post_telemetry(t: TelemetryIn):
    TELEMETRY.update(
        state=t.state,
        label=t.label,
        conf=t.conf,
        seconds_left=t.seconds_left,
        armed_progress=t.armed_progress,
        armed_ready=t.armed_ready,
        push_history=t.push_history,
    )
    snap = TELEMETRY.snapshot()
    await manager.broadcast({"type": "telemetry", "data": snap})
    return {"ok": True}


@app.get("/api/telemetry")
async def get_telemetry():
    return JSONResponse(TELEMETRY.snapshot(), headers=_no_cache_headers())


# ----------------------------------------------------------------------
# HIGHSCORES
# ----------------------------------------------------------------------
class HighscoreSubmit(BaseModel):
    name: str
    score: int


@app.get("/api/highscores")
async def get_highscores():
    return JSONResponse(
        {"items": highscores.list_highscores()}, headers=_no_cache_headers()
    )


@app.post("/api/highscores/submit")
async def submit_highscore(payload: HighscoreSubmit):
    name = (payload.name or "").strip()
    score = int(payload.score or 0)

    if not name:
        return JSONResponse(
            {"ok": False, "error": "name required"},
            status_code=400,
            headers=_no_cache_headers(),
        )
    if score < 0:
        score = 0

    updated, best = highscores.update_best(name, score)
    items = highscores.list_highscores()

    # push update to all clients
    await manager.broadcast({"type": "highscores", "items": items})

    return JSONResponse(
        {"ok": True, "updated": updated, "best": best, "items": items},
        headers=_no_cache_headers(),
    )


# ----------------------------------------------------------------------
# QUIT (X in Browser) -> end Python process
# ----------------------------------------------------------------------
@app.post("/api/quit")
async def api_quit():
    """
    Terminates the entire process (server + run_live).
    We start a thread, wait briefly, then os._exit(0),
    so that the HTTP response is still sent.
    """

    def _killer():
        time.sleep(0.25)
        os._exit(0)

    threading.Thread(target=_killer, daemon=True).start()
    return JSONResponse({"ok": True}, headers=_no_cache_headers())


def run_tetris_server(host="127.0.0.1", port=8000):
    uvicorn.run(app, host=host, port=port, log_level="info")


def start_tetris_server_background(host="127.0.0.1", port=8000):
    def _run():
        uvicorn.run(app, host=host, port=port, log_level="info")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
