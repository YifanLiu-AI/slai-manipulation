"""Prepare datasets and train PI0.5 for the UR5/Wujihand pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from ._common import load_yaml, project_path


def _project_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = str(project_path("src"))
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root if not current else source_root + os.pathsep + current
    return environment


def _settings(config: dict) -> dict:
    if config.get("schema_version") != 1:
        raise ValueError("PI0.5 config schema_version must be 1")
    dataset, policy, training = config["dataset"], config["policy"], config["training"]
    input_schema_path = project_path(dataset["input_schema"])
    input_schema = load_yaml(input_schema_path)
    openpi_root = training.get("openpi_root") or os.environ.get("OPENPI_ROOT")
    v21_python = training.get("lerobot_v21_python") or os.environ.get("LEROBOT_V21_PYTHON")
    v3_python = training.get("lerobot_v3_python") or os.environ.get("LEROBOT_V3_PYTHON")
    return {
        **policy,
        **training,
        "source": project_path(dataset["source"]),
        "converted": project_path(dataset["converted"]),
        "repo_id": dataset["repo_id"],
        "native_repo_id": dataset["native_repo_id"],
        "native_v21": project_path(dataset["native_v21"]),
        "native_v30": project_path(dataset["native_v30"]),
        "input_schema": input_schema_path,
        "source_fps": int(input_schema["capture"]["fps"]),
        "policy_fps": int(input_schema["pi05"]["fps"]),
        "action_horizon": int(input_schema["pi05"]["action_horizon"]),
        "openpi_root": openpi_root,
        "v21_python": v21_python,
        "v3_python": project_path(v3_python) if v3_python else None,
        "assets_dir": project_path(training["assets_dir"]),
        "checkpoints_dir": project_path(training["checkpoints_dir"]),
        "base_checkpoint_dir": project_path(training["base_checkpoint_dir"]),
        "lerobot_train": project_path(training["lerobot_train"]),
        "generated_train_config": project_path(training["generated_train_config"]),
        "training_output_dir": project_path(training["training_output_dir"]),
        "smoke_output_dir": project_path(training["smoke_output_dir"]),
        "output_repo_id": training["output_repo_id"],
    }


def _latest_source() -> Path:
    from slai_mi.datasets.pi05 import validate_pi05_source

    valid = []
    for info in project_path("data/lerobot").glob("*/meta/info.json"):
        root = info.parent.parent
        try:
            validate_pi05_source(root)
        except (OSError, TypeError, ValueError):
            continue
        valid.append(root)
    if not valid:
        raise ValueError("no compatible committed PI0.5 capture was found")
    return max(valid, key=lambda path: (path.stat().st_mtime_ns, path.name)).resolve()


def _all_config(config: dict, source: Path, run_id: str | None) -> tuple[dict, Path]:
    identifier = run_id or source.name
    if not identifier or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in identifier):
        raise ValueError("--run-id may contain only letters, digits, dot, dash, and underscore")
    dataset, policy, training = config["dataset"], config["policy"], config["training"]
    training_root = Path("data/training/pi05")
    output = Path("outputs/pi05") / identifier
    dataset.update(
        source=str(source),
        converted=str(training_root / f"{identifier}_v21"),
        repo_id=f"{identifier}_v21",
        native_repo_id=f"local/{identifier}-pi05",
        native_v21=str(training_root / f"{identifier}_native_v21"),
        native_v30=str(training_root / f"{identifier}_native_v30"),
    )
    for task_path in project_path("configs/tasks").glob("*.yaml"):
        task = load_yaml(task_path).get("task", {})
        if source.name.startswith(str(task.get("id")) + "-"):
            policy["task_prompt"] = task["instruction"]
    training.update(
        experiment=identifier,
        assets_dir=str(Path("data/normalization/pi05") / identifier),
        generated_train_config=str(output / "train.yaml"),
        training_output_dir=str(output / "full"),
        smoke_output_dir=str(output / "smoke"),
        output_repo_id=f"local/{identifier}-policy",
    )
    return config, project_path(output / "pipeline.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("convert", "config", "norm", "train", "all"))
    parser.add_argument("--config", default="configs/pi05.yaml")
    parser.add_argument("--source", type=Path, help="One v3 root or a directory of v3 roots")
    parser.add_argument("--target", type=Path, help="Output LeRobot v2.1 root")
    parser.add_argument("--native-v21", type=Path, help="Output native PI0.5 v2.1 root")
    parser.add_argument("--native-v30", type=Path, help="Output native PI0.5 v3 root")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--experiment")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        loaded = load_yaml(args.config)
        if args.command == "all":
            source = args.source.expanduser().resolve() if args.source else _latest_source()
            loaded, generated = _all_config(loaded, source, args.run_id)
            print(json.dumps({"app": "pi05", "command": "all", "source": str(source), "config": str(generated), "smoke": args.smoke, "mode": "execute" if args.execute else "dry-run"}, indent=2))
            if not args.execute:
                return 0
            if generated.exists():
                raise FileExistsError(f"PI0.5 run already exists: {generated.parent.name}")
            generated.parent.mkdir(parents=True, exist_ok=True)
            generated.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")
            for command in ("convert", "norm", "config", "train"):
                child = [sys.executable, "-m", "slai_mi.apps.pi05", command, "--config", str(generated), "--execute"]
                if args.smoke and command in {"config", "train"}:
                    child.append("--smoke")
                subprocess.run(child, check=True)
            settings = _settings(loaded)
            output = settings["smoke_output_dir"] if args.smoke else settings["training_output_dir"]
            checkpoints = sorted(
                (output / "checkpoints").glob("[0-9]*/pretrained_model"),
                key=lambda path: int(path.parent.name),
            )
            if not checkpoints:
                raise RuntimeError(f"training produced no checkpoint under {output}")
            inference = {
                "schema_version": 1,
                "target": "offline",
                "backend": "slai_mi.policies.pi05_lerobot:factory",
                "checkpoint": str(checkpoints[-1]),
                "model_python": str(settings["v3_python"]),
                "device": "cuda",
                "input_schema": str(settings["input_schema"]),
                "dataset": {"repo_id": settings["native_repo_id"], "root": str(settings["native_v30"]), "physical_v21_root": str(settings["converted"]), "video_backend": "pyav", "frame_index": 0},
                "deployment": {"task_prompt": loaded["policy"]["task_prompt"], "max_steps": 0, "inference_timeout_s": 5.0},
            }
            inference_path = generated.with_name("inference.yaml")
            inference_path.write_text(yaml.safe_dump(inference, sort_keys=False), encoding="utf-8")
            print(inference_path)
            return 0
        settings = _settings(loaded)
        if args.steps is not None:
            settings["steps"] = args.steps
        if args.batch_size is not None:
            settings["batch_size"] = args.batch_size
        if args.experiment is not None:
            settings["experiment"] = args.experiment
        if args.source is not None:
            settings["source"] = args.source.expanduser().resolve()
        if args.target is not None:
            settings["converted"] = args.target.expanduser().resolve()
        if args.native_v21 is not None:
            settings["native_v21"] = args.native_v21.expanduser().resolve()
        if args.native_v30 is not None:
            settings["native_v30"] = args.native_v30.expanduser().resolve()
        plan = {
            "app": "pi05",
            "command": args.command,
            "mode": "execute" if args.execute else "dry-run",
            "source": str(settings["source"]),
            "converted": str(settings["converted"]),
            "native_v21": str(settings["native_v21"]),
            "native_v30": str(settings["native_v30"]),
            "repo_id": settings["repo_id"],
            "backend": settings["backend"],
            "steps": settings["steps"],
            "batch_size": settings["batch_size"],
            "openpi_root": settings["openpi_root"],
            "v21_python": settings["v21_python"],
            "v3_python": str(settings["v3_python"]) if settings["v3_python"] else None,
            "smoke": args.smoke,
            "input_schema": str(settings["input_schema"]),
        }
        print(json.dumps(plan, indent=2))
        if not args.execute:
            return 0
        # OpenPI has a tightly pinned Python/JAX/LeRobot environment. Re-enter
        # this module with that interpreter so ``uv run slai-pi05 ...`` cannot
        # accidentally use the lightweight project environment.
        uses_openpi = args.command == "norm" or (
            args.command == "train" and settings["backend"] == "openpi_jax"
        )
        if uses_openpi and os.environ.get("SLAI_PI05_REEXEC") != "1":
            if not settings["v21_python"]:
                raise ValueError("set training.lerobot_v21_python or LEROBOT_V21_PYTHON")
            env = _project_environment()
            env["SLAI_PI05_REEXEC"] = "1"
            subprocess.run(
                [str(settings["v21_python"]), "-m", "slai_mi.apps.pi05", *sys.argv[1:]],
                check=True,
                env=env,
            )
            return 0
        if args.command == "convert":
            if not settings["v21_python"]:
                raise ValueError("set training.lerobot_v21_python or LEROBOT_V21_PYTHON")
            if not settings["v3_python"]:
                raise ValueError("set training.lerobot_v3_python or LEROBOT_V3_PYTHON")
            subprocess.run(
                [
                    str(settings["v3_python"]),
                    "-m",
                    "slai_mi.datasets.pi05_convert_entry",
                    str(settings["source"]),
                    str(settings["converted"]),
                    "--repo-id",
                    settings["repo_id"],
                    "--v21-python",
                    str(settings["v21_python"]),
                    "--schema",
                    str(settings["input_schema"]),
                ],
                check=True,
                env=_project_environment(),
            )
            subprocess.run(
                [
                    str(settings["v21_python"]),
                    "-m",
                    "slai_mi.datasets.pi05_native_entry",
                    "build-v21",
                    str(settings["converted"]),
                    str(settings["native_v21"]),
                    "--source-repo-id",
                    settings["repo_id"],
                    "--target-repo-id",
                    settings["native_repo_id"],
                    "--schema",
                    str(settings["input_schema"]),
                ],
                check=True,
                env=_project_environment(),
            )
            subprocess.run(
                [
                    str(settings["v3_python"]),
                    "-m",
                    "slai_mi.datasets.pi05_native_entry",
                    "upgrade-v30",
                    str(settings["native_v21"]),
                    str(settings["native_v30"]),
                    "--repo-id",
                    settings["native_repo_id"],
                ],
                check=True,
                env=_project_environment(),
            )
        elif args.command == "config":
            from slai_mi.training.lerobot_pi05 import write_lerobot_train_config

            print(write_lerobot_train_config(settings, smoke=args.smoke))
        elif args.command == "norm":
            if not settings["openpi_root"]:
                raise ValueError("set training.openpi_root or OPENPI_ROOT")
            from slai_mi.training.pi05 import run_openpi

            run_openpi(args.command, settings, smoke=args.smoke, max_frames=args.max_frames)
        elif settings["backend"] == "lerobot_pytorch":
            from slai_mi.training.lerobot_pi05 import run_lerobot_train

            print(run_lerobot_train(settings, smoke=args.smoke))
        elif settings["backend"] == "openpi_jax":
            from slai_mi.training.pi05 import run_openpi

            run_openpi("train", settings, smoke=args.smoke, max_frames=None)
        else:
            raise ValueError(f"unsupported PI0.5 training backend: {settings['backend']}")
        return 0
    except (ImportError, KeyError, OSError, RuntimeError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        raise SystemExit(f"PI0.5 {args.command} failed: {exc}") from exc


if __name__ == "__main__":
    sys.exit(main())
