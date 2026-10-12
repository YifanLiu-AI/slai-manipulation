"""Schema-driven LeRobot v3 writer for strategy-specific real datasets."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from slai_mi.input_schema import capture_vector_names, enabled_cameras
from slai_mi.rotation import rotation6d_columns_to_matrix

CONTRACT_FILENAME = "robot_teleoperation_contract.json"
WRIST_8DOF_CONTRACT_ID = "robot_teleoperation.ur5_wrist.three_rgb_cartesian_rot6d_columns.v1"
GRIPPER_9DOF_CONTRACT_ID = "robot_teleoperation.ur5_wrist_gripper.three_rgb_cartesian_rot6d_columns.v1"
BUTTON_NAMES = (
    "menu",
    "fit",
    "r",
    "f",
    "one",
    "two",
    "three",
    "home",
    "esc",
    "alt",
    "shift",
    "ctrl",
)


class ConfiguredDatasetContract:
    def __init__(self, schema: Mapping[str, Any]) -> None:
        self.schema = schema
        self.capture = schema["capture"]
        self.cameras = enabled_cameras(schema)
        has_gripper = any(
            name.startswith("gripper.") for name in capture_vector_names(schema, "state")
        )
        self.contract_id = GRIPPER_9DOF_CONTRACT_ID if has_gripper else WRIST_8DOF_CONTRACT_ID
        self.robot_type = "ur5_wrist_gripper_9dof" if has_gripper else "ur5_wrist_8dof"
        self.fps = int(self.capture["fps"])
        self.state_names = capture_vector_names(schema, "state")
        self.action_names = capture_vector_names(schema, "action")
        self.source_names = (
            *(str(item["role"]) for item in self.cameras),
            *(str(item["name"]) for item in schema["synchronization"]["state_channels"]),
            str(schema["synchronization"]["command_channel"]["name"]),
        )

    def features(self) -> dict[str, dict[str, Any]]:
        height, width, channels = map(int, self.capture["image_shape"])
        rgb = {
            "dtype": "video",
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        }
        tcp = self.capture["tcp_pose"]
        roles = tuple(str(item["role"]) for item in self.cameras)
        primary = str(self.capture["primary_timeline_role"])
        features: dict[str, dict[str, Any]] = {
            **{
                str(item["dataset_key"]): {**rgb, "info": dict(rgb["info"])}
                for item in self.cameras
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.state_names),),
                "names": list(self.state_names),
            },
            str(tcp["key"]): {
                "dtype": "float32",
                "shape": (len(tcp["names"]),),
                "names": list(tcp["names"]),
            },
            "action": {
                "dtype": "float32",
                "shape": (len(self.action_names),),
                "names": list(self.action_names),
            },
            "telemetry.ur5_target_qd": {
                "dtype": "float32",
                "shape": (6,),
                "names": [f"joint_{i}" for i in range(6)],
            },
            "telemetry.actual_tcp_speed": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["vx", "vy", "vz", "wx", "wy", "wz"],
            },
            "telemetry.camera_skew_ms": {
                "dtype": "float32",
                "shape": (max(0, len(roles) - 1),),
                "names": [f"{role}_to_{primary}" for role in roles if role != primary],
            },
            "telemetry.spacemouse_axes": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["x", "y", "z", "rx", "ry", "rz"],
            },
            "telemetry.spacemouse_buttons": {
                "dtype": "int64",
                "shape": (12,),
                "names": list(BUTTON_NAMES),
            },
        }
        for key, dtype in (
            ("telemetry.source_age_ms", "float32"),
            ("telemetry.device_timestamps_s", "float64"),
            ("telemetry.host_receive_timestamps_s", "float64"),
            ("telemetry.source_sequence_numbers", "int64"),
            ("telemetry.source_dropped_before", "int64"),
            ("telemetry.source_restart_counts", "int64"),
            ("telemetry.validity_mask", "int64"),
        ):
            features[key] = {
                "dtype": dtype,
                "shape": (len(self.source_names),),
                "names": list(self.source_names),
            }
        return features

    def validate_frame(self, frame: Mapping[str, Any]) -> None:
        expected = set(self.features()) | {"task"}
        if set(frame) != expected:
            raise ValueError(
                f"configured frame keys differ: missing={sorted(expected - set(frame))}, extra={sorted(set(frame) - expected)}"
            )
        if not isinstance(frame["task"], str) or not frame["task"].strip():
            raise ValueError("task must be a non-empty string")
        image_keys = {str(item["dataset_key"]) for item in self.cameras}
        for key, feature in self.features().items():
            value = np.asarray(frame[key])
            dtype = np.dtype("uint8" if feature["dtype"] == "video" else feature["dtype"])
            if value.shape != tuple(feature["shape"]):
                raise ValueError(f"{key} shape is {value.shape}, expected {feature['shape']}")
            if value.dtype != dtype:
                raise ValueError(f"{key} dtype is {value.dtype}, expected {dtype}")
            if key not in image_keys and not np.isfinite(value).all():
                raise ValueError(f"{key} contains a non-finite value")
        rotation6d_columns_to_matrix(np.asarray(frame[str(self.capture["tcp_pose"]["key"])])[3:])

    def manifest(self) -> dict[str, Any]:
        features = self.features()
        normalized = {
            key: {
                "dtype": item["dtype"],
                "shape": list(item["shape"]),
                "names": list(item["names"]),
            }
            for key, item in features.items()
        }
        digest = hashlib.sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        policy = self.schema["pi05"]
        return {
            "contract_id": self.contract_id,
            "schema_sha256": digest,
            "lerobot_codebase_version": "v3.0",
            "fps": self.fps,
            "robot_type": self.robot_type,
            "capture": {
                "state": list(self.capture["state"]["components"]),
                "tcp_pose": dict(self.capture["tcp_pose"]),
                "action": list(self.capture["action"]["components"]),
            },
            "pi05_training_view": {
                "physical_state_dim": len(self.state_names),
                "physical_action_dim": len(self.action_names),
                "model_state_dim": int(policy["state"]["model_pad_to"]),
                "model_action_dim": int(policy["action"]["model_pad_to"]),
            },
        }

    def write_manifest(self, root: Path) -> None:
        path = root / "meta" / CONTRACT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def validate_root(self, root: Path) -> dict[str, Any]:
        if json.loads((root / "meta" / CONTRACT_FILENAME).read_text()) != self.manifest():
            raise ValueError("dataset manifest differs from selected strategy schema")
        info = json.loads((root / "meta" / "info.json").read_text())
        if (
            info.get("codebase_version") != "v3.0"
            or info.get("fps") != self.fps
            or info.get("robot_type") != self.robot_type
        ):
            raise ValueError("final dataset metadata differs from selected strategy")
        if int(info.get("total_episodes", 0)) < 1 or int(info.get("total_frames", 0)) < 1:
            raise ValueError("final dataset contains no committed frames")
        if not set(self.features()).issubset(info.get("features", {})):
            raise ValueError("final dataset is missing configured features")
        import av

        video_count = 0
        expected_height, expected_width, _ = map(int, self.capture["image_shape"])
        all_videos = tuple((root / "videos").rglob("*.mp4"))
        for camera in self.cameras:
            key = str(camera["dataset_key"])
            paths = tuple(path for path in all_videos if key in path.parts)
            if not paths:
                raise ValueError(f"final dataset is missing video for {key}")
            for path in paths:
                with av.open(str(path)) as container:
                    if not container.streams.video:
                        raise ValueError(f"video has no stream: {path}")
                    stream = container.streams.video[0]
                    if (stream.height, stream.width) != (
                        expected_height,
                        expected_width,
                    ):
                        raise ValueError(f"video dimensions differ from schema: {path}")
                    try:
                        next(container.decode(stream))
                    except StopIteration as exc:
                        raise ValueError(f"video has no decodable frame: {path}") from exc
                video_count += 1
        return {
            "episodes": int(info["total_episodes"]),
            "frames": int(info["total_frames"]),
            "contract_id": self.contract_id,
            "videos": video_count,
        }


class ConfiguredDatasetWriter:
    def __init__(self, dataset: Any, root: Path, contract: ConfiguredDatasetContract) -> None:
        self._dataset, self._root, self.contract = dataset, root, contract
        self.validation_report: dict[str, Any] | None = None
        contract.write_manifest(root)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.contract.validate_frame(frame)
        self._dataset.add_frame(frame)

    def finalize(self) -> None:
        self._dataset.finalize()
        if int(self._dataset.meta.total_episodes) == 0:
            if self._root.exists():
                shutil.rmtree(self._root)
            return
        self.validation_report = self.contract.validate_root(self._root)


def create_configured_dataset(
    *, repo_id: str, root: Path, contract: ConfiguredDatasetContract
) -> ConfiguredDatasetWriter:
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        raise FileExistsError(f"refusing to overwrite dataset root: {root}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type=contract.robot_type,
        fps=contract.fps,
        features=contract.features(),
        use_videos=True,
        tolerance_s=1e-4,
        batch_encoding_size=1,
        rgb_encoder=RGBEncoderConfig(
            vcodec="h264", pix_fmt="yuv420p", crf=23, preset="veryfast", video_backend="pyav"
        ),
        streaming_encoding=True,
        encoder_queue_maxsize=90,
        encoder_threads=2,
    )
    return ConfiguredDatasetWriter(dataset, root, contract)
