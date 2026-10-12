from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from slai_mi.apps.pi05 import _all_config, main
from slai_mi.datasets.pi05 import (
    discover_source_roots,
    policy_rgb,
    stage_episode,
    validate_pi05_source,
)
from slai_mi.datasets.pi05_native import validate_native_v30_stats
from slai_mi.evaluation.heldout import select_horizon_rows
from slai_mi.input_schema import load_input_schema
from slai_mi.training.lerobot_pi05 import build_lerobot_train_config, run_lerobot_train


class _Dataset:
    def __init__(self) -> None:
        self.items = []
        for index in range(4):
            self.items.append(
                {
                    "observation.state": np.full(26, index, dtype=np.float32),
                    "observation.tcp_pose": np.asarray(
                        [0.1 * index, 0.2, 0.3, 0.0, 0.0, 0.1 * index],
                        dtype=np.float32,
                    ),
                    "action": np.full(26, index + 20, dtype=np.float32),
                    "observation.images.primary_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
                    "observation.images.secondary_rgb": np.zeros((3, 224, 224), dtype=np.uint8),
                    "task": "pick block",
                }
            )

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def test_stage_episode_builds_rotation6d_state_and_15hz_view(tmp_path) -> None:
    path = tmp_path / "episode.npz"
    schema = load_input_schema()
    schema["capture"]["cameras"][1]["enabled"] = False
    assert stage_episode(_Dataset(), 0, 4, 2, path, schema=schema) == 2
    with np.load(path) as episode:
        assert episode["state"].shape == (2, 29)
        np.testing.assert_allclose(
            episode["state"][0, 3:9],
            np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        )
        assert episode["actions"].shape == (2, 26)
        assert episode["primary_rgb"].shape == (2, 224, 224, 3)
        assert episode["task"].item() == "pick block"


def test_stage_episode_dimensions_are_selected_by_yaml_schema(tmp_path) -> None:
    schema = load_input_schema()
    schema["capture"]["cameras"][1]["enabled"] = False
    schema["pi05"]["state"]["sources"][0]["indices"] = [0, 2]
    schema["pi05"]["state"]["sources"][1]["indices"] = [1, 4]
    schema["pi05"]["action"]["indices"] = [0, 5, 25]
    path = tmp_path / "selected.npz"
    stage_episode(_Dataset(), 0, 4, 2, path, schema=schema)
    with np.load(path) as episode:
        assert episode["state"].shape == (2, 4)
        assert episode["actions"].shape == (2, 3)


def test_policy_rgb_rejects_non_rgb() -> None:
    with pytest.raises(ValueError, match="RGB"):
        policy_rgb(np.zeros((224, 224)))


def test_legacy_two_camera_contract_is_an_explicit_pi05_source(tmp_path) -> None:
    root = tmp_path / "batch"
    (root / "meta").mkdir(parents=True)
    features = {
        key: {"shape": shape}
        for key, shape in {
            "observation.images.primary_rgb": [480, 640, 3],
            "observation.images.secondary_rgb": [480, 640, 3],
            "observation.state": [26],
            "observation.tcp_pose": [6],
            "action": [26],
        }.items()
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "features": features,
                "total_episodes": 2,
                "total_frames": 100,
            }
        )
    )
    (root / "meta" / "robot_teleoperation_contract.json").write_text(
        json.dumps({"contract_id": "robot_teleoperation.ur5_wuji.pi05_cartesian.v1"})
    )

    assert discover_source_roots(tmp_path) == (root,)
    schema = load_input_schema()
    schema["capture"]["cameras"][1]["enabled"] = False
    assert validate_pi05_source(root, schema)["episodes"] == 2
    assert validate_pi05_source(root, schema)["state_dim"] == 29


def test_interleaved_v4_contract_is_not_silently_reinterpreted(tmp_path) -> None:
    root = tmp_path / "interleaved-v4"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "features": {},
                "total_episodes": 1,
                "total_frames": 1,
            }
        )
    )
    (root / "meta" / "robot_teleoperation_contract.json").write_text(
        json.dumps(
            {"contract_id": "robot_teleoperation.ur5_wuji.three_rgb_cartesian_rot6d.v4"}
        )
    )

    with pytest.raises(ValueError, match="unsupported PI0.5 source contract"):
        validate_pi05_source(root)


def test_heldout_sampling_never_crosses_episode_boundary() -> None:
    episodes = np.repeat(np.arange(2), 40)
    frames = np.tile(np.arange(40), 2)
    selected = select_horizon_rows(episodes, frames, count=6)
    assert len(selected) == 6
    assert np.all(frames[selected] <= 11)


def test_pi05_cli_defaults_to_a_non_executing_plan(capsys) -> None:
    assert main(["train"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["app"] == "pi05"
    assert plan["mode"] == "dry-run"
    assert plan["repo_id"] == "local/slai-ur5-wujihand-pi05"
    assert plan["native_v21"].endswith(
        "block_into_box_20260816T194416_rot6d_columns_native_v21"
    )
    assert plan["native_v30"].endswith(
        "block_into_box_20260816T194416_rot6d_columns_native_v30"
    )


def test_jax_training_overrides_stay_on_openpi_backend(capsys) -> None:
    assert main(
        [
            "train",
            "--config",
            "configs/pi05_h100_jax.yaml",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--experiment",
            "1gpu-1step",
        ]
    ) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["backend"] == "openpi_jax"
    assert plan["steps"] == 1
    assert plan["batch_size"] == 1


def test_jax_training_invokes_openpi(tmp_path: Path, monkeypatch) -> None:
    config = yaml.safe_load(Path("configs/pi05_h100_jax.yaml").read_text(encoding="utf-8"))
    config_path = tmp_path / "jax.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    calls = []
    monkeypatch.setenv("SLAI_PI05_REEXEC", "1")
    monkeypatch.setattr(
        "slai_mi.training.pi05.run_openpi",
        lambda command, settings, *, smoke, max_frames: calls.append(
            (command, settings["backend"], settings["steps"], settings["batch_size"])
        ),
    )

    assert main(
        [
            "train",
            "--config",
            str(config_path),
            "--steps",
            "10",
            "--batch-size",
            "2",
            "--execute",
        ]
    ) == 0
    assert calls == [("train", "openpi_jax", 10, 2)]


def test_all_config_is_scoped_to_one_capture() -> None:
    config = yaml.safe_load(Path("configs/pi05.yaml").read_text(encoding="utf-8"))
    source = Path("data/lerobot/remove_objects_from_box-20mm-20260817T191606").resolve()
    generated, path = _all_config(config, source, "acceptance")
    assert generated["dataset"]["source"] == str(source)
    assert generated["dataset"]["native_v30"].endswith("acceptance_native_v30")
    assert generated["policy"]["task_prompt"] == "Remove the objects from the box."
    assert path.name == "pipeline.yaml"


def test_lerobot_training_dimensions_follow_yaml_schema(tmp_path: Path) -> None:
    schema = load_input_schema()
    schema["pi05"]["action_horizon"] = 7
    schema["pi05"]["state"]["model_pad_to"] = 40
    schema["pi05"]["action"]["model_pad_to"] = 48
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    settings = {
        "input_schema": schema_path,
        "native_repo_id": "local/test",
        "native_v30": tmp_path / "dataset",
        "base_checkpoint_dir": tmp_path / "model",
        "output_repo_id": "local/output",
        "experiment": "schema-test",
        "steps": 10,
        "batch_size": 2,
        "save_interval": 5,
        "lora_rank": 4,
    }
    config = build_lerobot_train_config(settings, smoke=False, output_dir=tmp_path / "output")
    assert config["policy"]["chunk_size"] == 7
    assert config["policy"]["n_action_steps"] == 7
    assert config["policy"]["max_state_dim"] == 40
    assert config["policy"]["max_action_dim"] == 48
    assert config["wandb"] == {"enable": True, "project": "slai-pi05"}


def test_full_lerobot_training_uses_swanlab_entrypoint(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    metadata = dataset / "meta"
    metadata.mkdir(parents=True)
    (metadata / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {"q01": [0.0], "q99": [1.0]},
                "action": {"q01": [0.0], "q99": [1.0]},
            }
        ),
        encoding="utf-8",
    )
    executable = tmp_path / "bin" / "lerobot-train"
    executable.parent.mkdir()
    executable.touch()
    settings = {
        "input_schema": Path("configs/input_schema.yaml"),
        "native_repo_id": "local/test",
        "native_v30": dataset,
        "base_checkpoint_dir": tmp_path / "model",
        "output_repo_id": "local/output",
        "experiment": "swanlab-test",
        "steps": 10,
        "batch_size": 2,
        "save_interval": 5,
        "lora_rank": 4,
        "lerobot_train": executable,
        "generated_train_config": tmp_path / "train.yaml",
        "training_output_dir": tmp_path / "output",
        "smoke_output_dir": tmp_path / "smoke",
    }
    calls = []
    monkeypatch.setattr("slai_mi.training.lerobot_pi05.subprocess.run", lambda command, check: calls.append(command))

    run_lerobot_train(settings, smoke=False)

    assert calls == [
        [
            str(executable.with_name("python")),
            "-m",
            "slai_mi.training.lerobot_pi05",
            f"--config_path={tmp_path / 'train.yaml'}",
        ]
    ]


def test_native_training_view_requires_quantile_stats(tmp_path: Path) -> None:
    metadata = tmp_path / "meta"
    metadata.mkdir()
    (metadata / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {"q01": [0.0], "q99": [1.0]},
                "action": {"q01": [0.0], "q99": [1.0]},
            }
        ),
        encoding="utf-8",
    )
    validate_native_v30_stats(tmp_path)
    (metadata / "stats.json").write_text(
        json.dumps({"observation.state": {"q01": [0], "q99": [1]}, "action": {"q01": [0]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="action stats are missing: q99"):
        validate_native_v30_stats(tmp_path)
