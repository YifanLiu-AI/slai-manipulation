"""Training and inference CLI safety tests."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from slai_mi import inference, train
from slai_mi.policies.runtime import build_inference_plan, execute_inference
from slai_mi.training import runtime as training_runtime


def _yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_default_commands_are_dry_runs(capsys) -> None:
    assert train.main([]) == 0
    training = json.loads(capsys.readouterr().out)
    assert training["mode"] == "dry-run"
    assert inference.main([]) == 0
    prediction = json.loads(capsys.readouterr().out)
    assert prediction["mode"] == "dry-run"
    assert prediction["target"] == "offline"


def test_inference_refuses_real_target(tmp_path: Path) -> None:
    config = _yaml(
        tmp_path / "real.yaml",
        {"schema_version": 1, "target": "real", "backend": None, "checkpoint": "model"},
    )
    with pytest.raises(ValueError, match="safety supervisor"):
        build_inference_plan(config)


def test_inference_backend_is_lazy_and_testable(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.write_text("fake", encoding="utf-8")
    module = types.ModuleType("fake_policy_backend")

    def factory(**context):
        return lambda: {"target": context["target"], "loaded": context["checkpoint"].name}

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = _yaml(
        tmp_path / "inference.yaml",
        {
            "schema_version": 1,
            "target": "offline",
            "backend": "fake_policy_backend:factory",
            "checkpoint": str(checkpoint),
        },
    )
    result = execute_inference(build_inference_plan(config))
    assert result["backend_result"] == {"target": "offline", "loaded": "model"}


def test_training_backend_receives_validated_dataset(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    module = types.ModuleType("fake_training_backend")

    def factory(**context):
        return lambda: {"dataset": context["dataset_root"].name}

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        training_runtime,
        "validate_dataset_root",
        lambda root: {"episodes": 2, "root": root.name},
    )
    config = _yaml(
        tmp_path / "training.yaml",
        {
            "schema_version": 1,
            "backend": "fake_training_backend:factory",
            "dataset": {"root": str(dataset)},
            "policy": {"action_horizon": 8, "action_keys": ["actions"]},
        },
    )
    plan = training_runtime.build_training_plan(config)
    result = training_runtime.execute_training(plan)
    assert result["dataset_validation"]["episodes"] == 2
    assert result["backend_result"] == {"dataset": "dataset"}


def test_execute_requires_configured_backends() -> None:
    with pytest.raises(SystemExit, match="training backend is not configured"):
        train.main(["--execute"])
    with pytest.raises(SystemExit, match="inference backend is not configured"):
        inference.main(["--execute"])


def test_pi05_inference_reenters_model_environment(monkeypatch) -> None:
    monkeypatch.setattr(inference, "reexec_with_python", lambda *_args: 23)
    assert (
        inference.main(
            ["--config", "configs/inference_pi05_round10.yaml", "--execute"]
        )
        == 23
    )
