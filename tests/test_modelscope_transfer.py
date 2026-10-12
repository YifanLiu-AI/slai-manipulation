from __future__ import annotations

import json
from pathlib import Path

from slai_mi.apps import modelscope_transfer


def test_modelscope_repository_names_are_stable() -> None:
    assert modelscope_transfer.dataset_repo("task2") == "tmp-task2-dataset"
    assert modelscope_transfer.dataset_repo("Pick Block") == "tmp-pick-block-dataset"
    assert modelscope_transfer.checkpoint_repo("task2") == "task2-checkpoint"


def test_checkpoint_business_kind_maps_to_modelscope_model_repo() -> None:
    assert modelscope_transfer._repo_type("checkpoint") == "model"
    assert modelscope_transfer._repo_type("dataset") == "dataset"


def test_dataset_upload_defaults_to_private_dry_run(tmp_path: Path, capsys) -> None:
    (tmp_path / "meta").mkdir()
    assert modelscope_transfer.main(
        [
            "dataset-upload",
            "--task",
            "task2",
            "--namespace",
            "alice",
            "--source",
            str(tmp_path),
        ]
    ) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "dry-run"
    assert plan["repo_id"] == "alice/tmp-task2-dataset"
    assert plan["visibility"] == "private"
    assert not (tmp_path / modelscope_transfer.MANIFEST).exists()


def test_manifest_detects_complete_round_trip(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text('{"frames": 1}', encoding="utf-8")
    (tmp_path / "data.bin").write_bytes(b"dataset")
    modelscope_transfer.write_manifest(
        tmp_path, kind="dataset", repo_id="alice/tmp-task2-dataset"
    )
    result = modelscope_transfer.verify_manifest(tmp_path)
    assert result["file_count"] == 2
    assert result["total_bytes"] > 0


def test_manifest_rejects_changed_download(tmp_path: Path) -> None:
    payload = tmp_path / "data.bin"
    payload.write_bytes(b"before")
    modelscope_transfer.write_manifest(
        tmp_path, kind="dataset", repo_id="alice/tmp-task2-dataset"
    )
    payload.write_bytes(b"after")
    try:
        modelscope_transfer.verify_manifest(tmp_path)
    except RuntimeError as exc:
        assert "changed" in str(exc)
    else:
        raise AssertionError("changed transfer must fail verification")
