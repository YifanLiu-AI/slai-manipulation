"""Schema-driven LeRobot PI0.5 LoRA config generation and launcher."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from slai_mi.input_schema import load_input_schema


def build_lerobot_train_config(
    settings: dict[str, Any], *, smoke: bool, output_dir: Path
) -> dict[str, Any]:
    schema = load_input_schema(settings["input_schema"])
    pi05 = schema["pi05"]
    return {
        "dataset": {
            "repo_id": settings["native_repo_id"],
            "root": str(settings["native_v30"]),
            "video_backend": "pyav",
        },
        "policy": {
            "type": "pi05",
            "path": str(settings["base_checkpoint_dir"]),
            "device": "cuda",
            "dtype": "bfloat16",
            "chunk_size": int(pi05["action_horizon"]),
            "n_action_steps": int(pi05["action_horizon"]),
            "max_state_dim": int(pi05["state"]["model_pad_to"]),
            "max_action_dim": int(pi05["action"]["model_pad_to"]),
            "gradient_checkpointing": True,
            "train_expert_only": True,
            "repo_id": settings["output_repo_id"],
            "push_to_hub": False,
        },
        "output_dir": str(output_dir),
        "job_name": settings["experiment"],
        "steps": 1 if smoke else int(settings["steps"]),
        "batch_size": 1 if smoke else int(settings["batch_size"]),
        "num_workers": 0,
        "save_freq": 1 if smoke else int(settings["save_interval"]),
        "log_freq": 1 if smoke else 200,
        # SwanLab's WandB compatibility layer consumes LeRobot's native metrics.
        "wandb": {"enable": not smoke, "project": "slai-pi05"},
        "peft": {
            "target_modules": "all-linear",
            "r": int(settings.get("lora_rank", 4)),
            "lora_alpha": int(settings.get("lora_rank", 4)),
        },
    }


def write_lerobot_train_config(settings: dict[str, Any], *, smoke: bool) -> Path:
    path = Path(settings["generated_train_config"])
    output_dir = Path(settings["smoke_output_dir"] if smoke else settings["training_output_dir"])
    config = build_lerobot_train_config(settings, smoke=smoke, output_dir=output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_lerobot_train(settings: dict[str, Any], *, smoke: bool) -> Path:
    from slai_mi.datasets.pi05_native import validate_native_v30_stats

    executable = Path(settings["lerobot_train"])
    if not executable.is_file():
        raise FileNotFoundError(f"LeRobot train executable is missing: {executable}")
    if not Path(settings["native_v30"]).is_dir():
        raise FileNotFoundError(f"native PI0.5 training dataset is missing: {settings['native_v30']}")
    validate_native_v30_stats(Path(settings["native_v30"]))
    config_path = write_lerobot_train_config(settings, smoke=smoke)
    command = [str(executable), f"--config_path={config_path}"]
    if not smoke:
        command = [
            str(executable.with_name("python")),
            "-m",
            "slai_mi.training.lerobot_pi05",
            f"--config_path={config_path}",
        ]
    subprocess.run(command, check=True)
    return Path(settings["smoke_output_dir"] if smoke else settings["training_output_dir"])


def main() -> None:
    """Run LeRobot with its native WandB metrics redirected to SwanLab."""
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError("full PI0.5 training requires the pi05-train dependencies") from exc

    swanlab.sync_wandb(wandb_run=False)
    from lerobot.scripts.lerobot_train import main as lerobot_train

    lerobot_train()


if __name__ == "__main__":
    sys.exit(main())
