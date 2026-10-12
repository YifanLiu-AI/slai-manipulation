"""LeRobot Dataset v3 writer configuration for canonical real demonstrations."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .contract import (
    validate_dataset_root,
    validate_frame,
    write_contract_manifest,
)
from .schema import FPS, lerobot_features


class ContractDatasetWriter:
    """Apply the canonical contract at frame input and after finalization."""

    def __init__(self, dataset: Any, root: Path) -> None:
        self._dataset = dataset
        self._root = root
        self.validation_report: dict[str, Any] | None = None
        write_contract_manifest(root)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    def add_frame(self, frame: dict[str, Any]) -> None:
        validate_frame(frame)
        self._dataset.add_frame(frame)

    def finalize(self) -> None:
        self._dataset.finalize()
        if int(self._dataset.meta.total_episodes) == 0:
            if self._root.exists():
                shutil.rmtree(self._root)
            return
        self.validation_report = validate_dataset_root(self._root)


def create_dataset(*, repo_id: str, root: Path, fps: int = FPS):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        raise FileExistsError(f"refusing to overwrite dataset root: {root}")
    if fps != FPS:
        raise ValueError(f"canonical VLA v3 capture must use {FPS} Hz, got {fps}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="ur5_wuji_hand1",
        fps=fps,
        features=lerobot_features(),
        use_videos=True,
        tolerance_s=1e-4,
        batch_encoding_size=1,
        rgb_encoder=RGBEncoderConfig(
            vcodec="h264",
            pix_fmt="yuv420p",
            crf=23,
            preset="veryfast",
            video_backend="pyav",
        ),
        streaming_encoding=True,
        encoder_queue_maxsize=90,
        encoder_threads=2,
    )
    return ContractDatasetWriter(dataset, root)
