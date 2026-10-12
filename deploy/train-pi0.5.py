#!/usr/bin/env python3
"""Prepare one ModelScope-delivered dataset and run PI0.5 smoke/formal training."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT = Path(os.environ.get("PROJECT", Path(__file__).resolve().parents[1])).resolve()
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slai_mi.apps._common import load_yaml
from slai_mi.apps.pi05 import _all_config


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pi05_task2_9dof.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    smoke = args.mode == "smoke"
    num_gpus = args.num_gpus if args.num_gpus is not None else (1 if smoke else 4)
    steps = args.steps if args.steps is not None else (1 if smoke else 30000)
    if smoke and (num_gpus != 1 or steps != 1):
        parser.error("smoke acceptance is intentionally fixed to 1 GPU and 1 step")
    if not smoke and num_gpus != 4:
        parser.error("formal H100 training requires exactly 4 GPUs")
    if args.batch_size < 1 or steps < 1:
        parser.error("steps and batch size must be positive")

    source = (PROJECT / args.source).resolve() if not args.source.is_absolute() else args.source.resolve()
    config_path = (PROJECT / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    loaded = load_yaml(config_path)
    loaded, generated = _all_config(loaded, source, args.run_id)
    loaded["training"]["steps"] = steps
    loaded["training"]["batch_size"] = args.batch_size
    loaded["training"]["experiment"] = args.run_id
    plan = {
        "mode": args.mode,
        "source": str(source),
        "pipeline_config": str(generated),
        "num_gpus": num_gpus,
        "steps": steps,
        "batch_size_per_process": args.batch_size,
        "purpose": (
            "one-step data/model/backprop/checkpoint acceptance; not a quality run"
            if smoke
            else "formal four-GPU PI0.5 training"
        ),
    }
    print(yaml.safe_dump(plan, sort_keys=False), flush=True)
    if not args.execute:
        return 0
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"downloaded LeRobot dataset is invalid: {source}")
    if generated.exists():
        raise FileExistsError(f"training run already exists: {generated.parent}")
    generated.parent.mkdir(parents=True, exist_ok=False)
    generated.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")

    v3_python = PROJECT / ".venv-lerobot-v3/bin/python"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{SRC}{os.pathsep}{environment.get('PYTHONPATH', '')}"
    environment.setdefault("HOME", str(PROJECT / ".h100-home"))
    environment.setdefault("HF_HOME", str(PROJECT / ".cache/huggingface"))
    environment.setdefault("XDG_CACHE_HOME", str(PROJECT / ".cache"))
    environment.setdefault("WANDB_MODE", "offline")
    for command in ("convert", "norm", "config"):
        child = [str(v3_python), "-m", "slai_mi.apps.pi05", command, "--config", str(generated), "--execute"]
        if smoke and command == "config":
            child.append("--smoke")
        run(child, env=environment)

    settings = loaded["training"]
    train_config = (PROJECT / settings["generated_train_config"]).resolve()
    output = (PROJECT / (settings["smoke_output_dir"] if smoke else settings["training_output_dir"])).resolve()
    if smoke:
        run(
            [str(v3_python), "-m", "slai_mi.apps.pi05", "train", "--config", str(generated), "--smoke", "--execute"],
            env=environment,
        )
    else:
        accelerate = PROJECT / ".venv-lerobot-v3/bin/accelerate"
        command = [
            str(accelerate), "launch", "--mixed_precision", "bf16", "--num_processes", "4", "--multi_gpu",
            "-m", "slai_mi.training.lerobot_pi05", f"--config_path={train_config}",
            f"--steps={steps}", f"--batch_size={args.batch_size}", f"--output_dir={output}",
        ]
        run(command, env=environment)
    checkpoints = sorted((output / "checkpoints").glob("[0-9]*/pretrained_model"))
    if not checkpoints:
        raise RuntimeError(f"training produced no checkpoint: {output}")
    print(f"checkpoint={checkpoints[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
