"""Write staged PI0.5 episodes with the intentionally separate LeRobot v2.1 runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from slai_mi.input_schema import enabled_cameras, load_input_schema


def write_dataset(
    staging: Path, root: Path, repo_id: str, *, schema_path: Path
) -> None:
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(f"PI0.5 writer requires LeRobot v2.1, found {CODEBASE_VERSION}")
    if root.exists():
        raise FileExistsError(f"refusing to overwrite PI0.5 dataset: {root}")
    episodes = sorted(staging.glob("episode-*.npz"))
    if not episodes:
        raise ValueError(f"no staged episodes found in {staging}")
    schema = load_input_schema(schema_path)
    cameras = enabled_cameras(schema)
    with np.load(episodes[0], allow_pickle=False) as first:
        state_dim = int(first["state"].shape[1])
        action_dim = int(first["actions"].shape[1])
    image = {"dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channel"]}
    features = {str(camera["openpi_key"]): dict(image) for camera in cameras}
    features.update(
        {
            "state": {"dtype": "float32", "shape": (state_dim,), "names": ["pi05_state"]},
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["pi05_actions"],
            },
        }
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="ur5_wujihand_pi05",
        fps=int(schema["pi05"]["fps"]),
        features=features,
        use_videos=True,
        tolerance_s=1e-6,
        image_writer_threads=4,
    )
    try:
        for path in episodes:
            with np.load(path, allow_pickle=False) as episode:
                image_arrays = [episode[str(camera["openpi_key"])] for camera in cameras]
                arrays = [*image_arrays, episode["state"], episode["actions"]]
                if len({len(value) for value in arrays}) != 1 or not len(arrays[0]):
                    raise ValueError(f"inconsistent or empty episode: {path}")
                for index in range(len(arrays[0])):
                    dataset.add_frame(
                        {
                            **{
                                str(camera["openpi_key"]): np.ascontiguousarray(
                                    image_arrays[camera_index][index]
                                )
                                for camera_index, camera in enumerate(cameras)
                            },
                            "state": np.ascontiguousarray(episode["state"][index], dtype=np.float32),
                            "actions": np.ascontiguousarray(
                                episode["actions"][index], dtype=np.float32
                            ),
                            "task": str(episode["task"].item()),
                        }
                    )
            dataset.save_episode()
    finally:
        dataset.stop_image_writer()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args()
    write_dataset(
        args.staging.resolve(),
        args.root.resolve(),
        args.repo_id,
        schema_path=args.schema.resolve(),
    )


if __name__ == "__main__":
    main()
