"""Merge 30 Hz VLA collection sessions."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from .configured import ConfiguredDatasetContract
from .contract import validate_dataset_root, write_contract_manifest


def merge_vla_v3_datasets(
    source_roots: list[Path],
    target_root: Path,
    *,
    repo_id: str,
    contract: ConfiguredDatasetContract | None = None,
) -> Path:
    """Merge independently recorded sessions into one training root."""
    sources = [path.resolve() for path in source_roots]
    target = target_root.resolve()
    if len(sources) < 2:
        raise ValueError("at least two source datasets are required")
    if len(set(sources)) != len(sources):
        raise ValueError("source dataset roots must be unique")
    if not repo_id.strip():
        raise ValueError("repo_id must not be empty")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite merged dataset: {target}")
    if target in sources:
        raise ValueError("target dataset must differ from every source dataset")

    validate = contract.validate_root if contract is not None else validate_dataset_root
    for source in sources:
        validate(source)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.merge-{uuid.uuid4().hex}"
    try:
        _merge_with_lerobot(sources, staging, repo_id)
        if contract is None:
            write_contract_manifest(staging)
        else:
            contract.write_manifest(staging)
        validate(staging)
        staging.replace(target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def _merge_with_lerobot(sources: list[Path], staging: Path, repo_id: str) -> None:
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets = [
        LeRobotDataset(repo_id=f"local/vla-merge-source-{index:03d}", root=root)
        for index, root in enumerate(sources)
    ]
    merge_datasets(
        datasets,
        output_repo_id=repo_id,
        output_dir=staging,
        concatenate_videos=False,
        concatenate_data=False,
    )
