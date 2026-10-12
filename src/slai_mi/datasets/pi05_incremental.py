"""Asynchronously build a PI0.5 LeRobot v2.1 sidecar during v3 collection."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from slai_mi.input_schema import enabled_cameras, load_input_schema

from .pi05 import select_policy_frame

STATE_FILENAME = "incremental_state.json"
COMPLETE_FILENAME = "FOLLOW_COMPLETE.json"


def _absolute(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else project_root / candidate


def incremental_target_for_source(source: Path, configured_root: Path) -> Path:
    """Return the unique policy-ready target corresponding to one v3 source."""
    return configured_root / f"{source.name}_v21"


def _writer_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    inherited = [
        value
        for value in environment.get("PYTHONPATH", "").split(os.pathsep)
        if value and Path(value).resolve() != Path(source_root).resolve()
    ]
    environment["PYTHONPATH"] = os.pathsep.join([source_root, *inherited])
    environment["PYTHONSAFEPATH"] = "1"
    return environment


def _safe_generated_building(target: Path) -> Path:
    building = target.with_name(f"{target.name}.building")
    if building.parent != target.parent or building.name != f"{target.name}.building":
        raise ValueError("incremental v2.1 building path is not safely derived from target")
    return building


def validate_incremental_v21(
    root: Path,
    *,
    schema: dict[str, Any],
    expected_episodes: int,
) -> dict[str, int | str]:
    """Validate the complete file-level v2.1 contract without importing LeRobot v2.1."""
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise ValueError(f"incremental v2.1 metadata is incomplete: {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total_episodes = int(info.get("total_episodes", -1))
    total_frames = int(info.get("total_frames", -1))
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"incremental target is not LeRobot v2.1: {root}")
    if total_episodes != expected_episodes or len(episodes) != expected_episodes:
        raise ValueError(
            "incremental v2.1 episode count differs: "
            f"expected={expected_episodes} info={total_episodes} metadata={len(episodes)}"
        )
    if total_frames != sum(int(episode["length"]) for episode in episodes):
        raise ValueError("incremental v2.1 frame metadata is inconsistent")
    if int(info.get("fps", -1)) != int(schema["pi05"]["fps"]):
        raise ValueError("incremental v2.1 FPS differs from the input schema")
    cameras = enabled_cameras(schema)
    features = info.get("features", {})
    for camera in cameras:
        key = str(camera["openpi_key"])
        feature = features.get(key, {})
        if feature.get("dtype") != "video" or feature.get("shape") != [224, 224, 3]:
            raise ValueError(f"incremental v2.1 camera feature is invalid: {key}")
    parquet_count = len(tuple((root / "data").rglob("*.parquet")))
    video_count = len(tuple((root / "videos").rglob("*.mp4")))
    if parquet_count != expected_episodes:
        raise ValueError(
            f"incremental v2.1 parquet count differs: {parquet_count} != {expected_episodes}"
        )
    expected_videos = expected_episodes * len(cameras)
    if video_count != expected_videos:
        raise ValueError(
            f"incremental v2.1 video count differs: {video_count} != {expected_videos}"
        )
    return {
        "backend": "jax",
        "episodes": total_episodes,
        "frames": total_frames,
        "videos": video_count,
    }


def recover_incremental_v21(staging: Path) -> dict[str, int | str]:
    """Rebuild and publish a sidecar whose canonical v3 source was finalized."""
    staging = staging.expanduser().resolve()
    state_path = staging / STATE_FILENAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != 1 or state.get("backend") != "jax":
        raise ValueError("incremental recovery state requires schema 1 and backend=jax")
    if Path(state["staging"]).resolve() != staging:
        raise ValueError("incremental recovery staging path differs from its state file")
    source = Path(state["source"]).resolve()
    target = Path(state["target"]).resolve()
    building = Path(state["building"]).resolve()
    if building != _safe_generated_building(target):
        raise ValueError("incremental recovery building path is not derived from target")
    schema_path = Path(state["schema"]).resolve()
    v21_python = Path(state["v21_python"]).resolve()
    if not v21_python.is_file():
        raise FileNotFoundError(f"LeRobot v2.1 Python is missing: {v21_python}")
    source_info_path = source / "meta" / "info.json"
    source_info = json.loads(source_info_path.read_text(encoding="utf-8"))
    expected_episodes = int(source_info.get("total_episodes", -1))
    if expected_episodes < 1:
        raise ValueError("canonical v3 source has no committed episodes")
    episodes = sorted(staging.glob("episode-*.npz"))
    expected_names = [f"episode-{index:06d}.npz" for index in range(expected_episodes)]
    if [path.name for path in episodes] != expected_names:
        raise ValueError(
            "incremental staging does not exactly match canonical v3 episode count"
        )
    schema = load_input_schema(schema_path)
    if target.exists():
        return validate_incremental_v21(
            target,
            schema=schema,
            expected_episodes=expected_episodes,
        )
    if building.exists():
        shutil.rmtree(building)
    writer = Path(__file__).with_name("pi05_writer.py")
    subprocess.run(
        [
            str(v21_python),
            str(writer),
            str(staging),
            str(building),
            "--repo-id",
            str(state["repo_id"]),
            "--schema",
            str(schema_path),
        ],
        check=True,
        env=_writer_environment(),
    )
    report = validate_incremental_v21(
        building,
        schema=schema,
        expected_episodes=expected_episodes,
    )
    building.replace(target)
    shutil.rmtree(staging)
    print(
        "PI05_INCREMENTAL_RECOVERY_COMPLETE "
        f"backend=jax episodes={report['episodes']} target={target}",
        flush=True,
    )
    return report


class IncrementalV21Sidecar:
    """Tee selected capture frames to a persistent v2.1 writer without blocking capture."""

    def __init__(
        self,
        *,
        source: Path,
        target: Path,
        repo_id: str,
        schema_path: Path,
        v21_python: Path,
        backend: str,
        queue_max_frames: int = 128,
        writer_timeout_s: float = 1800.0,
    ) -> None:
        if backend != "jax":
            raise ValueError("incremental PI0.5 sidecar requires explicit backend=jax")
        if queue_max_frames < 1:
            raise ValueError("incremental v2.1 queue_max_frames must be positive")
        if writer_timeout_s <= 0:
            raise ValueError("incremental v2.1 writer_timeout_s must be positive")
        self.source = source.resolve()
        self.target = target.resolve()
        self.repo_id = repo_id
        self.schema_path = schema_path.resolve()
        self.v21_python = Path(os.path.abspath(v21_python.expanduser()))
        self.backend = backend
        self.schema = load_input_schema(self.schema_path)
        source_fps = int(self.schema["capture"]["fps"])
        policy_fps = int(self.schema["pi05"]["fps"])
        if source_fps <= 0 or policy_fps <= 0 or source_fps % policy_fps:
            raise ValueError("capture FPS must be a positive multiple of PI0.5 FPS")
        if not self.v21_python.is_file():
            raise FileNotFoundError(f"LeRobot v2.1 Python is missing: {self.v21_python}")
        if self.target.exists():
            raise FileExistsError(f"refusing to overwrite incremental v2.1 target: {self.target}")

        self.stride = source_fps // policy_fps
        self.staging = self.target.with_name(f"{self.target.name}.staging")
        self.building = _safe_generated_building(self.target)
        for path in (self.staging, self.building):
            if path.exists():
                raise FileExistsError(f"incremental v2.1 work path already exists: {path}")
        self.staging.mkdir(parents=True)
        self.complete_marker = self.staging / COMPLETE_FILENAME
        self.writer_timeout_s = writer_timeout_s
        self._queue: queue.Queue[tuple[str, object | None]] = queue.Queue(
            maxsize=queue_max_frames
        )
        self._frame_index = 0
        self._committed_episodes = 0
        self._closed = False
        self._failure: BaseException | None = None
        self._failure_lock = threading.Lock()
        self.report: dict[str, int | str] | None = None
        self._write_state()
        self._writer = self._start_writer(follow=True, root=self.building)
        self._thread = threading.Thread(
            target=self._run_stager,
            name="pi05-v21-stager",
            daemon=True,
        )
        self._thread.start()
        print(
            "PI05_INCREMENTAL_STARTED "
            f"backend={self.backend} source={self.source} target={self.target}",
            flush=True,
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        source: Path,
        repo_id: str,
        schema_path: Path,
        project_root: Path,
    ) -> IncrementalV21Sidecar | None:
        selected = config.get("pi05_v21_sidecar")
        if not isinstance(selected, Mapping) or not selected.get("enabled", False):
            return None
        backend = str(selected.get("backend") or "")
        target_root = _absolute(str(selected["root"]), project_root).resolve()
        target = incremental_target_for_source(source.resolve(), target_root)
        v21_python = _absolute(str(selected["lerobot_v21_python"]), project_root)
        return cls(
            source=source,
            target=target,
            repo_id=repo_id,
            schema_path=schema_path,
            v21_python=v21_python,
            backend=backend,
            queue_max_frames=int(selected.get("queue_max_frames", 128)),
            writer_timeout_s=float(selected.get("writer_timeout_s", 1800.0)),
        )

    def _write_state(self) -> None:
        state = {
            "schema_version": 1,
            "backend": self.backend,
            "source": str(self.source),
            "target": str(self.target),
            "building": str(self.building),
            "staging": str(self.staging),
            "repo_id": self.repo_id,
            "schema": str(self.schema_path),
            "v21_python": str(self.v21_python),
        }
        (self.staging / STATE_FILENAME).write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _start_writer(self, *, follow: bool, root: Path) -> subprocess.Popen[bytes]:
        writer = Path(__file__).with_name("pi05_writer.py")
        command = [
            str(self.v21_python),
            str(writer),
            str(self.staging),
            str(root),
            "--repo-id",
            self.repo_id,
            "--schema",
            str(self.schema_path),
        ]
        if follow:
            command.extend(
                [
                    "--follow",
                    "--completion-marker",
                    str(self.complete_marker),
                    "--parent-pid",
                    str(os.getpid()),
                ]
            )
        return subprocess.Popen(command, env=_writer_environment())

    def _set_failure(self, failure: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = failure
                print(
                    f"PI05_INCREMENTAL_BACKGROUND_FAILED {type(failure).__name__}: {failure}",
                    flush=True,
                )

    def _get_failure(self) -> BaseException | None:
        with self._failure_lock:
            return self._failure

    def prepare_frame(self, frame: Mapping[str, Any]) -> dict[str, Any] | None:
        """Copy one selected raw frame before the v3 writer mutates its input mapping."""
        if self._closed or self._get_failure() is not None:
            return None
        if self._frame_index % self.stride:
            return None
        try:
            snapshot: dict[str, Any] = {"task": str(frame["task"])}
            for camera in enabled_cameras(self.schema):
                key = next(
                    (str(key) for key in camera["source_keys"] if str(key) in frame),
                    None,
                )
                if key is None:
                    raise ValueError(
                        f"configured camera {camera['role']} is absent from capture frame"
                    )
                snapshot[key] = np.array(frame[key], copy=True)
            required = {
                str(source["key"])
                for source in self.schema["pi05"]["state"]["sources"]
            }
            required.add(str(self.schema["pi05"]["action"]["key"]))
            for key in required:
                snapshot[key] = np.array(frame[key], copy=True)
            return snapshot
        except BaseException as exc:  # noqa: BLE001 - sidecar must not drop the v3 frame
            self._set_failure(exc)
            return None

    def accept_prepared_frame(self, snapshot: dict[str, Any] | None) -> None:
        """Advance the source timeline and enqueue the selected snapshot, if any."""
        if self._closed:
            return
        self._frame_index += 1
        if snapshot is None or self._get_failure() is not None:
            return
        try:
            self._queue.put_nowait(("frame", snapshot))
        except queue.Full:
            self._set_failure(
                RuntimeError(
                    "incremental v2.1 staging queue is full; v3 capture continued unchanged"
                )
            )

    def commit_episode(self) -> None:
        self._frame_index = 0
        if self._closed or self._get_failure() is not None:
            return
        self._committed_episodes += 1
        self._put_control("commit")

    def discard_episode(self) -> None:
        self._frame_index = 0
        if self._closed or self._get_failure() is not None:
            return
        self._put_control("discard")

    def _put_control(self, operation: str) -> None:
        try:
            self._queue.put((operation, None), timeout=5.0)
        except queue.Full:
            self._set_failure(RuntimeError(f"could not enqueue v2.1 {operation} event"))

    def _run_stager(self) -> None:
        cameras = enabled_cameras(self.schema)
        images: dict[str, list[np.ndarray]] = {
            str(camera["openpi_key"]): [] for camera in cameras
        }
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        task = ""

        def reset() -> None:
            nonlocal images, states, actions, task
            images = {str(camera["openpi_key"]): [] for camera in cameras}
            states = []
            actions = []
            task = ""

        try:
            output_episode = 0
            while True:
                operation, payload = self._queue.get()
                if operation == "frame":
                    selected_images, state, action, selected_task = select_policy_frame(
                        payload, schema=self.schema
                    )
                    for key, image in selected_images.items():
                        images[key].append(image)
                    states.append(state)
                    actions.append(action)
                    task = selected_task
                elif operation == "discard":
                    reset()
                elif operation == "commit":
                    if not states:
                        raise ValueError("committed episode has no PI0.5 frames")
                    path = self.staging / f"episode-{output_episode:06d}.npz"
                    temporary = path.with_suffix(".npz.tmp")
                    with temporary.open("wb") as stream:
                        np.savez(
                            stream,
                            **{key: np.stack(values) for key, values in images.items()},
                            state=np.stack(states),
                            actions=np.stack(actions),
                            task=np.asarray(task),
                        )
                        stream.flush()
                        os.fsync(stream.fileno())
                    temporary.replace(path)
                    print(
                        "PI05_INCREMENTAL_STAGE_OK "
                        f"episode={output_episode} frames={len(states)} path={path}",
                        flush=True,
                    )
                    output_episode += 1
                    reset()
                elif operation == "finish":
                    if states:
                        raise RuntimeError("incremental v2.1 stager has an uncommitted episode")
                    return
                else:
                    raise ValueError(f"unknown incremental v2.1 operation: {operation}")
        except BaseException as exc:  # noqa: BLE001 - report background failure at finalize
            self._set_failure(exc)

    def _terminate_writer(self) -> None:
        if self._writer.poll() is not None:
            return
        self._writer.terminate()
        try:
            self._writer.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._writer.kill()
            self._writer.wait(timeout=10.0)

    def _rebuild_from_staging(self) -> None:
        if self.building.exists():
            shutil.rmtree(self.building)
        writer = Path(__file__).with_name("pi05_writer.py")
        subprocess.run(
            [
                str(self.v21_python),
                str(writer),
                str(self.staging),
                str(self.building),
                "--repo-id",
                self.repo_id,
                "--schema",
                str(self.schema_path),
            ],
            check=True,
            env=_writer_environment(),
        )
        print("PI05_INCREMENTAL_WRITER_RECOVERED_FROM_STAGING", flush=True)

    def finalize(self, *, expected_episodes: int) -> dict[str, int | str]:
        """Drain staging, finish or recover the v2.1 writer, validate, and publish."""
        if self._closed:
            return self.report or {"backend": self.backend, "episodes": expected_episodes}
        self._closed = True
        self._put_control("finish")
        self._thread.join(timeout=self.writer_timeout_s)
        if self._thread.is_alive():
            self._set_failure(RuntimeError("incremental v2.1 stager did not stop"))
        failure = self._get_failure()
        if failure is not None:
            self._terminate_writer()
            raise RuntimeError(
                "incremental v2.1 staging failed; canonical v3 data remains intact at "
                f"{self.source}: {failure}"
            ) from failure
        if expected_episodes != self._committed_episodes:
            self._terminate_writer()
            raise RuntimeError(
                "incremental v2.1 commit count differs from v3: "
                f"sidecar={self._committed_episodes} v3={expected_episodes}"
            )

        self.complete_marker.write_text(
            json.dumps({"total_episodes": expected_episodes}) + "\n",
            encoding="utf-8",
        )
        try:
            returncode = self._writer.wait(timeout=self.writer_timeout_s)
        except subprocess.TimeoutExpired:
            self._terminate_writer()
            returncode = -1

        if expected_episodes == 0:
            if self.building.exists():
                shutil.rmtree(self.building)
            shutil.rmtree(self.staging)
            self.report = {"backend": self.backend, "episodes": 0, "frames": 0, "videos": 0}
            return self.report

        if returncode != 0:
            self._rebuild_from_staging()
        self.report = validate_incremental_v21(
            self.building,
            schema=self.schema,
            expected_episodes=expected_episodes,
        )
        self.building.replace(self.target)
        shutil.rmtree(self.staging)
        print(
            "PI05_INCREMENTAL_COMPLETE "
            f"backend={self.backend} episodes={self.report['episodes']} "
            f"frames={self.report['frames']} target={self.target}",
            flush=True,
        )
        return self.report
