"""Write staged PI0.5 episodes with the intentionally separate LeRobot v2.1 runtime."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from slai_mi.input_schema import enabled_cameras, load_input_schema


def _create_dataset(staging: Path, root: Path, repo_id: str, *, schema_path: Path):
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
    image_writer_threads = int(os.environ.get("SLAI_PI05_IMAGE_WRITER_THREADS", "8"))
    if not 1 <= image_writer_threads <= 32:
        raise ValueError("SLAI_PI05_IMAGE_WRITER_THREADS must be between 1 and 32")
    print(f"PI05_WRITER_THREADS threads={image_writer_threads}", flush=True)
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
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="ur5_wujihand_pi05",
        fps=int(schema["pi05"]["fps"]),
        features=features,
        use_videos=True,
        tolerance_s=1e-6,
        image_writer_threads=image_writer_threads,
    )


def _write_episode(dataset, path: Path, cameras: list[dict]) -> int:
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
        frame_count = len(arrays[0])
    dataset.save_episode()
    return frame_count


def write_dataset(
    staging: Path, root: Path, repo_id: str, *, schema_path: Path
) -> None:
    episodes = sorted(staging.glob("episode-*.npz"))
    if not episodes:
        raise ValueError(f"no staged episodes found in {staging}")
    schema = load_input_schema(schema_path)
    cameras = enabled_cameras(schema)
    dataset = _create_dataset(staging, root, repo_id, schema_path=schema_path)
    try:
        for path in episodes:
            _write_episode(dataset, path, cameras)
    finally:
        dataset.stop_image_writer()


def follow_dataset(
    staging: Path,
    root: Path,
    repo_id: str,
    *,
    schema_path: Path,
    completion_marker: Path,
    poll_interval_s: float = 0.1,
    parent_pid: int | None = None,
) -> None:
    """Append atomically staged episodes until the producer publishes completion."""
    if poll_interval_s <= 0:
        raise ValueError("follow poll interval must be positive")
    schema = load_input_schema(schema_path)
    cameras = enabled_cameras(schema)

    def require_parent() -> None:
        if parent_pid is not None and os.getppid() != parent_pid:
            raise RuntimeError(
                f"incremental v2.1 producer exited before completion: parent={parent_pid}"
            )

    while not any(staging.glob("episode-*.npz")) and not completion_marker.is_file():
        require_parent()
        time.sleep(poll_interval_s)
    if completion_marker.is_file():
        expected = int(json.loads(completion_marker.read_text(encoding="utf-8"))["total_episodes"])
        if expected == 0:
            return
    dataset = _create_dataset(staging, root, repo_id, schema_path=schema_path)
    next_episode = 0
    try:
        while True:
            path = staging / f"episode-{next_episode:06d}.npz"
            if path.is_file():
                frames = _write_episode(dataset, path, cameras)
                print(
                    "PI05_INCREMENTAL_EPISODE_OK "
                    f"episode={next_episode} frames={frames} path={path}",
                    flush=True,
                )
                next_episode += 1
                continue
            if completion_marker.is_file():
                payload = json.loads(completion_marker.read_text(encoding="utf-8"))
                expected = int(payload["total_episodes"])
                if next_episode != expected:
                    raise ValueError(
                        "incremental staging has an episode gap: "
                        f"next={next_episode} expected={expected}"
                    )
                break
            require_parent()
            time.sleep(poll_interval_s)
    finally:
        dataset.stop_image_writer()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--completion-marker", type=Path)
    parser.add_argument("--poll-interval-s", type=float, default=0.1)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    if args.follow:
        if args.completion_marker is None:
            raise ValueError("--completion-marker is required with --follow")
        follow_dataset(
            args.staging.resolve(),
            args.root.resolve(),
            args.repo_id,
            schema_path=args.schema.resolve(),
            completion_marker=args.completion_marker.resolve(),
            poll_interval_s=args.poll_interval_s,
            parent_pid=args.parent_pid,
        )
    else:
        if args.completion_marker is not None:
            raise ValueError("--completion-marker requires --follow")
        if args.parent_pid is not None:
            raise ValueError("--parent-pid requires --follow")
        write_dataset(
            args.staging.resolve(),
            args.root.resolve(),
            args.repo_id,
            schema_path=args.schema.resolve(),
        )


if __name__ == "__main__":
    main()
