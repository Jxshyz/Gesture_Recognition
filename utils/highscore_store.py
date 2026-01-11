# utils/highscore_store.py
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class HighscoreEntry:
    key: str
    name: str
    score: int
    updated_at: float


class HighscoreStore:
    """
    Speichert pro User genau EINEN Bestscore (max).
    Case-insensitive per casefold().
    Pro User eine Datei: ./Highscore_Tetris/<userkey>.json
    """

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def normalize_name(name: str) -> str:
        return (name or "").strip().casefold()

    @staticmethod
    def _safe_filename(key: str) -> str:
        key = (key or "").strip().casefold()
        key = re.sub(r"[^a-z0-9_-]+", "_", key)
        key = key.strip("_")
        return key or "player"

    def _path_for_key(self, key: str) -> Path:
        return self.root_dir / f"{self._safe_filename(key)}.json"

    def _read_entry_file(self, path: Path) -> HighscoreEntry | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            key = str(data.get("key", "")).strip()
            name = str(data.get("name", "")).strip() or key
            score = int(data.get("score", 0))
            updated_at = float(data.get("updated_at", 0.0))
            if not key:
                return None
            return HighscoreEntry(key=key, name=name, score=max(0, score), updated_at=updated_at)
        except Exception:
            return None

    @staticmethod
    def _atomic_write_json(path: Path, data: Dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def update_best(self, name: str, score: int) -> Tuple[bool, int]:
        """
        Update nur wenn score > bisheriger Bestscore.
        Return: (updated?, best_after)
        """
        disp_name = (name or "").strip() or "Player"
        key = self.normalize_name(disp_name) or "player"

        score = int(score)
        if score < 0:
            score = 0

        path = self._path_for_key(key)

        with self._lock:
            current = self._read_entry_file(path) if path.exists() else None
            best = current.score if current else 0

            if score <= best:
                # Optional: Namen (Groß/Klein) aktualisieren, wenn User anders geschrieben hat
                if current and current.name != disp_name:
                    self._atomic_write_json(
                        path,
                        {"key": key, "name": disp_name, "score": best, "updated_at": time.time()},
                    )
                return False, best

            self._atomic_write_json(
                path,
                {"key": key, "name": disp_name, "score": score, "updated_at": time.time()},
            )
            return True, score

    def list_highscores(self) -> List[Dict]:
        entries: List[HighscoreEntry] = []
        for p in self.root_dir.glob("*.json"):
            e = self._read_entry_file(p)
            if e:
                entries.append(e)

        # desc score, asc name
        entries.sort(key=lambda x: (-x.score, x.name.casefold()))
        return [{"name": e.name, "score": e.score, "updated_at": e.updated_at} for e in entries]
