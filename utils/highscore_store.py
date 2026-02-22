"""
File-based high score storage for local game sessions.

This module provides a thread-safe high score store that:

- Maintains exactly one (maximum) score per user
- Stores each user in a separate JSON file
- Uses case-insensitive user keys
- Performs atomic file writes to prevent corruption

Intended for lightweight local game backends.
"""
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
    """
    Immutable data structure representing a single user's high score.

    Attributes:
        key (str):
            Normalized (case-insensitive) user identifier.

        name (str):
            Display name of the user.

        score (int):
            Best recorded score for the user.

        updated_at (float):
            Unix timestamp of the last update.
    """

    key: str
    name: str
    score: int
    updated_at: float


class HighscoreStore:
    """
    Stores exactly ONE high score (max) per user
    Case-insensitive via casefold()
    One file per user: ./Highscore_Tetris/<userkey>.json
    """

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize a user name for internal key usage.

        Performs trimming and case-insensitive normalization.

        Parameters:
            name (str): Raw user input name.

        Returns:
            str: Normalized key.
        """
        return (name or "").strip().casefold()

    @staticmethod
    def _safe_filename(key: str) -> str:
        """
        Convert a user key into a filesystem-safe filename.

        Non-alphanumeric characters (except '_' and '-') are replaced.
        Empty results default to "player".

        Parameters:
            key (str): Normalized user key.

        Returns:
            str: Safe filename component.
        """
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
            return HighscoreEntry(
                key=key, name=name, score=max(0, score), updated_at=updated_at
            )
        except Exception:
            return None

    @staticmethod
    def _atomic_write_json(path: Path, data: Dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def update_best(self, name: str, score: int) -> Tuple[bool, int]:
        """
        Update the stored high score for a user.

        The score is only updated if the new score is strictly
        greater than the previous best.

        If the score is not improved, the stored display name
        may still be updated to reflect formatting changes.

        Parameters:
            name (str): Display name of the user.
            score (int): New score value.

        Returns:
            Tuple[bool, int]:
                - updated (bool): True if a new high score was written.
                - best_after (int): The user's best score after the operation.
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
                        {
                            "key": key,
                            "name": disp_name,
                            "score": best,
                            "updated_at": time.time(),
                        },
                    )
                return False, best

            self._atomic_write_json(
                path,
                {
                    "key": key,
                    "name": disp_name,
                    "score": score,
                    "updated_at": time.time(),
                },
            )
            return True, score

    def list_highscores(self) -> List[Dict]:
        """
        Return all stored high scores sorted for leaderboard display.

        Sorting order:
            1. Descending score
            2. Ascending name (case-insensitive)

        Returns:
            List[Dict]:
                List of dictionaries containing:
                    - name
                    - score
                    - updated_at
        """
        entries: List[HighscoreEntry] = []
        for p in self.root_dir.glob("*.json"):
            e = self._read_entry_file(p)
            if e:
                entries.append(e)

        # desc score, asc name
        entries.sort(key=lambda x: (-x.score, x.name.casefold()))
        return [
            {"name": e.name, "score": e.score, "updated_at": e.updated_at}
            for e in entries
        ]
