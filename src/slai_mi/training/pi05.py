"""Lazy OpenPI PI0.5 LoRA configuration and launcher."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from slai_mi.input_schema import enabled_cameras, load_input_schema


def make_train_config(settings: dict[str, Any], *, smoke: bool = False) -> object:
    """Build the tested UR5/Wujihand PI0.5 config without importing OpenPI at startup."""
    try:
        from openpi import transforms
        from openpi.models import pi0_config
        from openpi.training import config, optimizer, weight_loaders
    except ImportError as exc:
        raise RuntimeError("PI0.5 training requires an OpenPI environment") from exc

    info = json.loads((Path(settings["converted"]) / "meta" / "info.json").read_text())
    state_dim = int(info["features"]["state"]["shape"][0])
    action_dim = int(info["features"]["actions"]["shape"][0])
    schema = load_input_schema(settings["input_schema"])
    cameras = tuple(enabled_cameras(schema))
    model_state_dim = int(schema["pi05"]["state"]["model_pad_to"])
    model_action_dim = int(schema["pi05"]["action"]["model_pad_to"])
    image_slots = tuple(str(slot) for slot in schema["pi05"]["model_image_slots"])
    delta = schema["pi05"]["action"]["delta_from_state"]
    delta_action_indices = tuple(int(index) for index in delta["action_indices"])
    delta_state_indices = tuple(int(index) for index in delta["state_indices"])
    prompt = str(settings["task_prompt"])

    @dataclasses.dataclass(frozen=True)
    class Inputs(transforms.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            from slai_mi.policies.openpi import hand_position_to_delta, make_pi05_observation

            state = np.asarray(data["state"], dtype=np.float32)
            if state.shape != (state_dim,):
                raise ValueError(f"expected PI0.5 state[{state_dim}], got {state.shape}")
            result = make_pi05_observation(
                state=state,
                prompt=data.get("prompt", prompt),
                images={
                    str(camera["policy_key"]).removeprefix("observation.images."): data[
                        str(camera["openpi_key"])
                    ]
                    for camera in cameras
                },
                image_slots=image_slots,
            )
            if "actions" in data:
                result["actions"] = hand_position_to_delta(
                    data["actions"],
                    state,
                    action_indices=delta_action_indices,
                    state_indices=delta_state_indices,
                )
            return result

    @dataclasses.dataclass(frozen=True)
    class Outputs(transforms.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            from slai_mi.policies.openpi import hand_delta_to_position

            return {
                "actions": hand_delta_to_position(
                    data["actions"],
                    data["state"],
                    action_dim=action_dim,
                    action_indices=delta_action_indices,
                    state_indices=delta_state_indices,
                )
            }

    @dataclasses.dataclass(frozen=True)
    class DataFactory(config.DataConfigFactory):
        def create(self, assets_dirs: Path, model_config: object) -> object:
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                data_transforms=transforms.Group(inputs=[Inputs()], outputs=[Outputs()]),
                model_transforms=config.ModelTransformFactory(default_prompt=prompt)(model_config),
                action_sequence_keys=("actions",),
            )

    model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=model_action_dim,
        action_horizon=int(settings["action_horizon"]),
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    if state_dim > model_state_dim or action_dim > model_action_dim:
        raise ValueError(
            f"dataset state/action dimensions {(state_dim, action_dim)} exceed configured model "
            f"padding {(model_state_dim, model_action_dim)}"
        )
    return config.TrainConfig(
        name="slai_pi05_real_lora",
        exp_name=str(settings["experiment"]),
        project_name=str(settings.get("project_name", "slai-pi05-real-vla")),
        model=model,
        data=DataFactory(
            repo_id=str(settings["repo_id"]),
            base_config=config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(settings["base_checkpoint"])),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6
        ),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        seed=int(settings.get("seed", 42)),
        batch_size=1 if smoke else int(settings.get("batch_size", 16)),
        num_train_steps=1 if smoke else int(settings.get("steps", 30_000)),
        save_interval=1 if smoke else int(settings.get("save_interval", 5000)),
        keep_period=1 if smoke else int(settings.get("save_interval", 5000)),
        overwrite=smoke,
        wandb_enabled=not smoke,
        fsdp_devices=int(settings.get("fsdp_devices", 1)),
        # The transform classes are defined per configuration and are not
        # pickleable under OpenPI's spawn multiprocessing context. Keep the
        # loader single-process; this also makes norm-stat generation reliable
        # on workstations and containers.
        num_workers=0,
        assets_base_dir=str(settings["assets_dir"]),
        checkpoint_base_dir=str(settings["checkpoints_dir"]),
    )


def add_openpi_source(root: str | Path) -> None:
    checkout = Path(root).expanduser().resolve()
    source_roots = (checkout / "src", checkout / "packages" / "openpi-client" / "src")
    missing = [path for path in source_roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"OpenPI source directories are missing: {missing}")
    for source_root in reversed(source_roots):
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))


def run_openpi(command: str, settings: dict[str, Any], *, smoke: bool, max_frames: int | None) -> None:
    os.environ["HF_LEROBOT_HOME"] = str(Path(settings["converted"]).parent)
    root = Path(settings["openpi_root"]).expanduser().resolve()
    add_openpi_source(root)
    script_name = "compute_norm_stats" if command == "norm" else "train"
    path = root / "scripts" / f"{script_name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"OpenPI script is missing: {path}")
    train_config = make_train_config(settings, smoke=smoke)
    if command == "train" and train_config.data.create(
        train_config.assets_dirs, train_config.model
    ).norm_stats is None:
        run_openpi("norm", settings, smoke=smoke, max_frames=None)
        train_config = make_train_config(settings, smoke=smoke)
    import openpi.training.config as registry

    registry._CONFIGS_DICT[train_config.name] = train_config
    if command == "train" and settings.get("swanlab_enabled", False):
        try:
            import swanlab
        except ImportError as exc:
            raise RuntimeError("JAX training with SwanLab enabled requires swanlab") from exc
        swanlab.sync_wandb(wandb_run=False)
    spec = importlib.util.spec_from_file_location(f"openpi_project_{script_name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load OpenPI script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if command == "norm":
        module.main(train_config.name, max_frames=max_frames)
    else:
        module.main(train_config)
