"""Collection history backed by each LeRobot dataset directory."""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

_DATASET_STAMP = re.compile(r"^(?P<task>.+)-(?P<stamp>\d{8}T\d{6})$")
_DISCARD_CODES = {"episode_discard", "episode_empty"}


class CollectionEventJournal:
    """Append exact dashboard events beside the dataset they describe."""

    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.path = self.dataset_path / "meta" / "collection_events.jsonl"
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        payload = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                if os.write(descriptor, payload) != len(payload):
                    raise OSError(f"incomplete collection event write: {self.path}")
            finally:
                os.close(descriptor)

    def append_many(self, events: Iterable[dict[str, Any]]) -> None:
        for event in events:
            self.append(event)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_events(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def _dataset_identity(path: Path) -> tuple[str, str]:
    match = _DATASET_STAMP.match(path.name)
    if match is None:
        return path.name, datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
    stamp = datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%S").astimezone()
    return match.group("task"), stamp.isoformat(timespec="seconds")


def _aggregate_events(events: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    if not events:
        return None, None
    attempts = sum(event.get("code") == "episode_start" for event in events)
    discarded_attempts = {
        event.get("attempt")
        for event in events
        if event.get("code") in _DISCARD_CODES and event.get("attempt") is not None
    }
    unnumbered_discards = sum(
        event.get("code") in _DISCARD_CODES and event.get("attempt") is None for event in events
    )
    return attempts, len(discarded_attempts) + unnumbered_discards


def _session(path: Path) -> dict[str, Any] | None:
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        return None
    info = _read_json(info_path)
    task, started_at = _dataset_identity(path)
    events = _read_events(path / "meta" / "collection_events.jsonl")
    attempts, discarded = _aggregate_events(events)
    return {
        "time": started_at,
        "task": task,
        "attempts": attempts,
        "saved": max(0, int(info.get("total_episodes", 0) or 0)),
        "discarded": discarded,
        "dataset": path.name,
        "dataset_path": str(path),
        "has_event_log": bool(events),
    }


def build_collection_history(
    dataset_root: str | Path,
    *,
    current_status: dict[str, Any] | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    """Read committed counts from LeRobot metadata and attempt counts from event journals."""
    root = Path(dataset_root).expanduser().resolve()
    try:
        paths = [path for path in root.iterdir() if path.is_dir()]
    except OSError:
        paths = []
    sessions = [session for path in paths if (session := _session(path)) is not None]
    sessions.sort(key=lambda item: (str(item["time"]), str(item["dataset"])), reverse=True)

    status = current_status or {}
    current_path_value = status.get("dataset_path")
    current_path = (
        Path(str(current_path_value)).expanduser().resolve() if current_path_value else None
    )
    current = next(
        (item for item in sessions if current_path is not None and item["dataset_path"] == str(current_path)),
        sessions[0] if sessions else None,
    )
    if current is None:
        summary = {
            "task": str(status.get("task") or "等待采集任务"),
            "task_id": None,
            "dataset": None,
            "saved": 0,
            "attempts": 0 if current_path is not None else None,
            "discarded": 0 if current_path is not None else None,
            "history_saved": 0,
            "live": current_path is not None,
        }
    else:
        summary = {
            "task": str(status.get("task") or current["task"]),
            "task_id": current["task"],
            "dataset": current["dataset"],
            "saved": current["saved"],
            "attempts": current["attempts"],
            "discarded": current["discarded"],
            "history_saved": sum(
                int(item["saved"]) for item in sessions if item["task"] == current["task"]
            ),
            "live": current_path is not None,
        }
    return {
        "dataset_root": str(root),
        "summary": summary,
        "sessions": sessions[: max(0, int(limit))],
        "total_sessions": len(sessions),
    }
