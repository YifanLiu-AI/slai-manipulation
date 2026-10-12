"""Tests for dataset-backed collection history."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from slai_mi.collection.history import CollectionEventJournal, build_collection_history
from slai_mi.ui.collection_dashboard import CollectionDashboardProvider


def _dataset(root: Path, name: str, episodes: int) -> Path:
    dataset = root / name
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"total_episodes": episodes}),
        encoding="utf-8",
    )
    return dataset


def test_history_uses_lerobot_metadata_and_event_journal(tmp_path: Path) -> None:
    older = _dataset(tmp_path, "task1-20260822T212634", 2)
    current = _dataset(tmp_path, "task1-20260823T101500", 1)
    journal = CollectionEventJournal(current)
    journal.append({"code": "episode_start", "attempt": 1})
    journal.append({"code": "episode_save", "attempt": 1})
    journal.append({"code": "episode_start", "attempt": 2})
    journal.append({"code": "episode_discard", "attempt": 2})

    history = build_collection_history(
        tmp_path,
        current_status={"dataset_path": str(current), "task": "task1"},
    )

    assert history["summary"] == {
        "task": "task1",
        "task_id": "task1",
        "dataset": current.name,
        "saved": 1,
        "attempts": 2,
        "discarded": 1,
        "history_saved": 3,
        "live": True,
    }
    assert history["sessions"][1]["dataset"] == older.name
    assert history["sessions"][1]["attempts"] is None
    assert history["sessions"][1]["discarded"] is None


def test_history_reports_public_2dof_and_3dof_aggregate_counts(tmp_path: Path) -> None:
    last = tmp_path / "last"
    _dataset(last, "task2-20260828T040000", 8)
    _dataset(tmp_path / "aggregates", "task2-wrist-2dof", 3)
    _dataset(tmp_path / "aggregates", "task2-no-wrist-motion", 5)

    history = build_collection_history(last)

    assert history["dof_counts"] == {"2DoF": 3, "3DoF": 5}


def test_event_journal_keeps_complete_lines_when_threads_append(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "task1-20260823T110000", 0)
    journal = CollectionEventJournal(dataset)
    threads = [
        threading.Thread(target=journal.append, args=({"code": "test", "value": index},))
        for index in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = (dataset / "meta" / "collection_events.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert sorted(json.loads(line)["value"] for line in lines) == list(range(20))


def test_dashboard_events_are_persisted_inside_the_current_dataset(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "task1-20260823T120000", 0)
    provider = CollectionDashboardProvider(
        {"input_schema": "configs/input_schema.yaml", "cameras": {"devices": []}},
        "task1",
        task_id="task1",
    )
    provider.event("dashboard ready")
    provider.set_dataset_path(dataset)
    provider.start_episode(index=1, attempt=1)
    provider.finish_episode("discard", index=1)

    events = [
        json.loads(line)
        for line in (dataset / "meta" / "collection_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["code"] for event in events] == [
        "collection",
        "dataset",
        "episode_start",
        "episode_discard",
    ]
    assert events[-1]["attempt"] == 1
    assert events[-1]["dataset_path"] == str(dataset.resolve())
    summary = provider.collection_history()["summary"]
    assert summary["attempts"] == 1
    assert summary["discarded"] == 1
