"""Offline LeRobot PI0.5 checkpoint inference backend."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from slai_mi.input_schema import load_input_schema, split_capture_vector


class PI05Policy:
    """Loaded LeRobot policy shared by offline checks and live supervised execution."""

    def __init__(self, config: dict[str, Any], checkpoint: Path) -> None:
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.datasets import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        dataset = config["dataset"]
        metadata = LeRobotDatasetMetadata(str(dataset["repo_id"]), root=Path(dataset["root"]))
        policy_config = PreTrainedConfig.from_pretrained(str(checkpoint))
        policy_config.pretrained_path = checkpoint
        policy_config.device = str(config.get("device", "cuda"))
        self.policy = make_policy(policy_config, ds_meta=metadata)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_config,
            pretrained_path=str(checkpoint),
            dataset_stats=metadata.stats,
            dataset_meta=metadata,
        )
        self.policy.eval()
        self.torch = torch

    def infer(self, batch: dict[str, object]) -> np.ndarray:
        processed = self.preprocessor(batch)
        with self.torch.inference_mode():
            action = self.postprocessor(self.policy.select_action(processed))
        result = np.asarray(action.detach().cpu(), dtype=np.float32)
        if result.ndim != 2 or not len(result) or not np.isfinite(result).all():
            raise RuntimeError(f"PI0.5 produced an invalid action {result.shape}")
        return result


class PI05OfflineBackend:
    def __init__(self, *, config: dict[str, Any], checkpoint: Path, target: str) -> None:
        if target != "offline":
            raise ValueError("PI0.5 dataset inference backend is offline-only")
        self.config = config
        self.checkpoint = checkpoint

    def run(self) -> dict[str, object]:
        import torch
        from lerobot.datasets import LeRobotDataset

        dataset_config = self.config.get("dataset")
        if not isinstance(dataset_config, dict):
            raise TypeError("inference.dataset must be a mapping")
        root = Path(dataset_config["root"]).expanduser().resolve()
        repo_id = str(dataset_config["repo_id"])
        frame_index = int(dataset_config.get("frame_index", 0))
        dataset = LeRobotDataset(repo_id, root=root, video_backend=dataset_config.get("video_backend"))
        if not 0 <= frame_index < len(dataset):
            raise ValueError(f"frame_index {frame_index} is outside dataset length {len(dataset)}")

        started = time.perf_counter()
        policy = PI05Policy(self.config, self.checkpoint)
        load_s = time.perf_counter() - started

        item = dataset[frame_index]
        batch = {
            key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
            for key, value in item.items()
        }
        device = str(self.config.get("device", "cuda"))
        torch.cuda.synchronize() if device.startswith("cuda") else None
        started = time.perf_counter()
        action_array = policy.infer(batch)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        inference_s = time.perf_counter() - started

        physical_root = Path(dataset_config["physical_v21_root"]).expanduser().resolve()
        physical_info = json.loads((physical_root / "meta" / "info.json").read_text())
        physical_dim = int(physical_info["features"]["actions"]["shape"][0])
        physical_action = action_array[..., :physical_dim]
        if not np.isfinite(physical_action).all():
            raise RuntimeError("PI0.5 produced a non-finite action")
        schema = load_input_schema(self.config.get("input_schema"))
        components = split_capture_vector(schema, "action", physical_action[0])
        return {
            "checkpoint_type": "peft_lora",
            "dataset_frames": len(dataset),
            "frame_index": frame_index,
            "input_shapes": {
                key: list(value.shape)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
                and (key == "observation.state" or key.startswith("observation.images."))
            },
            "model_action_shape": list(action_array.shape),
            "physical_action_shape": list(physical_action.shape),
            "physical_action": physical_action.tolist(),
            "real_policy_components": {
                channel: {attribute: values.tolist() for attribute, values in fields.items()}
                for channel, fields in components.items()
            },
            "load_s": load_s,
            "inference_s": inference_s,
        }


def factory(**context: Any) -> PI05OfflineBackend:
    return PI05OfflineBackend(**context)
