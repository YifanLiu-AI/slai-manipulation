"""Schema-driven LeRobot v3 writer for strategy-specific real datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from slai_mi.input_schema import (
    BINARY_GRIPPER_CONTRACT,
    CONTINUOUS_GRIPPER_CONTRACT,
    capture_vector_names,
    enabled_cameras,
    gripper_value_contract,
)
from slai_mi.rotation import rotation6d_columns_to_matrix

CONTRACT_FILENAME = "robot_teleoperation_contract.json"
WRIST_8DOF_CONTRACT_ID = "robot_teleoperation.ur5_wrist.three_rgb_cartesian_rot6d_columns.v1"
GRIPPER_9DOF_CONTRACT_ID = "robot_teleoperation.ur5_wrist_gripper.three_rgb_cartesian_rot6d_columns.v1"
CONTINUOUS_GRIPPER_9DOF_CONTRACT_ID = (
    "robot_teleoperation.ur5_wrist_gripper.three_rgb_cartesian_rot6d_columns."
    "continuous_position.v2"
)
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


def _replace_dataset_alias(alias: Path, target: Path) -> None:
    alias = Path(os.path.abspath(alias.expanduser()))
    target = target.resolve()
    if alias == target or alias.parent == target:
        raise ValueError("dataset alias must be outside its target dataset")
    alias.parent.mkdir(parents=True, exist_ok=True)
    temporary = alias.with_name(f".{alias.name}.{os.getpid()}.tmp")
    try:
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(os.path.relpath(target, alias.parent), target_is_directory=True)
        os.replace(temporary, alias)
    finally:
        temporary.unlink(missing_ok=True)


def validate_continuous_gripper_samples(
    actual: object,
    target: object,
) -> dict[str, Any]:
    actual_values = np.asarray(actual, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if (
        actual_values.ndim != 1
        or target_values.shape != actual_values.shape
        or len(actual_values) == 0
        or not np.isfinite(actual_values).all()
        or not np.isfinite(target_values).all()
    ):
        raise ValueError("continuous gripper samples must be matching finite vectors")
    if (
        np.any(actual_values < 0.0)
        or np.any(actual_values > 1.0)
        or np.any(target_values < 0.0)
        or np.any(target_values > 1.0)
    ):
        raise ValueError("continuous gripper dataset values must remain in [0, 1]")
    rounded_targets = np.unique(np.round(target_values, decimals=4))
    intermediate = (target_values > 1e-4) & (target_values < 1.0 - 1e-4)
    if len(rounded_targets) < 3 or not np.any(intermediate):
        raise ValueError(
            "continuous gripper dataset contains only endpoint labels; "
            "collect hold-to-move intermediate targets before training"
        )
    return {
        "actual_range": [float(actual_values.min()), float(actual_values.max())],
        "target_range": [float(target_values.min()), float(target_values.max())],
        "target_unique_positions_4dp": len(rounded_targets),
        "target_intermediate_fraction": float(intermediate.mean()),
        "mean_actual_target_gap": float(np.mean(np.abs(actual_values - target_values))),
    }


class ConfiguredDatasetContract:
    def __init__(
        self,
        schema: Mapping[str, Any],
        *,
        gripper_calibration: Mapping[str, Any] | None = None,
    ) -> None:
        self.schema = schema
        self.capture = schema["capture"]
        self.cameras = enabled_cameras(schema)
        self.gripper_contract = gripper_value_contract(schema)
        if self.gripper_contract == CONTINUOUS_GRIPPER_CONTRACT:
            self.contract_id = CONTINUOUS_GRIPPER_9DOF_CONTRACT_ID
            self.robot_type = "ur5_wrist_gripper_continuous_9dof"
        elif self.gripper_contract == BINARY_GRIPPER_CONTRACT:
            self.contract_id = GRIPPER_9DOF_CONTRACT_ID
            self.robot_type = "ur5_wrist_gripper_9dof"
        else:
            self.contract_id = WRIST_8DOF_CONTRACT_ID
            self.robot_type = "ur5_wrist_8dof"
        self.fps = int(self.capture["fps"])
        self.state_names = capture_vector_names(schema, "state")
        self.action_names = capture_vector_names(schema, "action")
        self.source_names = (
            *(str(item["role"]) for item in self.cameras),
            *(str(item["name"]) for item in schema["synchronization"]["state_channels"]),
            str(schema["synchronization"]["command_channel"]["name"]),
        )
        self.gripper_calibration = self._validate_gripper_calibration(gripper_calibration)

    def _validate_gripper_calibration(
        self, calibration: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        if calibration is None:
            return None
        if self.gripper_contract != CONTINUOUS_GRIPPER_CONTRACT:
            raise ValueError("gripper calibration requires a continuous gripper contract")
        driver = str(calibration["driver"]).strip()
        open_position = int(calibration["open_position_raw"])
        closed_position = int(calibration["closed_position_raw"])
        feedback_bias = int(calibration.get("feedback_position_bias_raw", 0))
        if not driver:
            raise ValueError("gripper calibration driver must be non-empty")
        if not 0 <= open_position <= 4095 or not 0 <= closed_position <= 4095:
            raise ValueError("gripper calibration endpoints must be in [0, 4095]")
        if open_position == closed_position:
            raise ValueError("gripper calibration endpoints must differ")
        if not -4095 <= feedback_bias <= 4095:
            raise ValueError("gripper feedback position bias must be in [-4095, 4095]")
        validated = {
            "driver": driver,
            "open_position_raw": open_position,
            "closed_position_raw": closed_position,
            "target_normalization": "clip((raw-open)/(closed-open),0,1)",
        }
        if feedback_bias:
            validated.update(
                {
                    "feedback_position_bias_raw": feedback_bias,
                    "feedback_normalization": (
                        "clip(((raw-feedback_position_bias_raw)-open)/(closed-open),0,1)"
                    ),
                }
            )
        else:
            validated["feedback_normalization"] = "clip((raw-open)/(closed-open),0,1)"
        return validated

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
        if self.schema.get("telemetry", {}).get("wrist_motor", {}).get("enabled", False):
            for key, names in (
                ("telemetry.wrist_motor_position_raw", ["j1_present_raw", "j2_present_raw"]),
                ("telemetry.wrist_motor_goal_raw", ["j1_goal_raw", "j2_goal_raw"]),
                ("telemetry.wrist_motor_rate_raw_s", ["j1_rate_raw_s", "j2_rate_raw_s"]),
            ):
                features[key] = {"dtype": "int64", "shape": (2,), "names": names}
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
        if self.gripper_contract == CONTINUOUS_GRIPPER_CONTRACT:
            features.update(
                {
                    "telemetry.gripper_position_raw": {
                        "dtype": "int64",
                        "shape": (1,),
                        "names": ["servo_position_raw"],
                    },
                    "telemetry.gripper_target_position_raw": {
                        "dtype": "int64",
                        "shape": (1,),
                        "names": ["servo_target_position_raw"],
                    },
                }
            )
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
        if self.gripper_contract is not None:
            state = np.asarray(frame["observation.state"])
            action = np.asarray(frame["action"])
            state_index = self.state_names.index("gripper.position")
            action_index = self.action_names.index("gripper.target_position")
            actual = float(state[state_index])
            target = float(action[action_index])
            if self.gripper_contract == BINARY_GRIPPER_CONTRACT:
                if actual not in (0.0, 1.0):
                    raise ValueError("9DoF gripper state must be binary 0/1")
                if target not in (0.0, 1.0):
                    raise ValueError("9DoF gripper action must be binary 0/1")
            elif not 0.0 <= actual <= 1.0 or not 0.0 <= target <= 1.0:
                raise ValueError("continuous 9DoF gripper state/action must be in [0, 1]")
        if self.gripper_contract == CONTINUOUS_GRIPPER_CONTRACT:
            for key in (
                "telemetry.gripper_position_raw",
                "telemetry.gripper_target_position_raw",
            ):
                value = int(np.asarray(frame[key])[0])
                if not 0 <= value <= 4095:
                    raise ValueError(
                        f"{key} must contain an adapter-native position in [0, 4095]"
                    )
            if self.gripper_calibration is not None:
                open_position = int(self.gripper_calibration["open_position_raw"])
                closed_position = int(self.gripper_calibration["closed_position_raw"])
                feedback_bias = int(
                    self.gripper_calibration.get("feedback_position_bias_raw", 0)
                )

                def normalized(raw: int) -> float:
                    value = (raw - open_position) / (closed_position - open_position)
                    return min(1.0, max(0.0, value))

                raw_actual = int(np.asarray(frame["telemetry.gripper_position_raw"])[0])
                raw_target = int(
                    np.asarray(frame["telemetry.gripper_target_position_raw"])[0]
                )
                if not np.isclose(
                    actual, normalized(raw_actual - feedback_bias), atol=1e-5
                ):
                    raise ValueError(
                        "normalized gripper state does not match its raw feedback and calibration"
                    )
                if not np.isclose(target, normalized(raw_target), atol=1e-5):
                    raise ValueError(
                        "normalized gripper action does not match its raw target and calibration"
                    )
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
        manifest = {
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
        if self.gripper_contract == CONTINUOUS_GRIPPER_CONTRACT:
            manifest["gripper_value_contract"] = CONTINUOUS_GRIPPER_CONTRACT
            if self.gripper_calibration is not None:
                manifest["gripper_calibration"] = dict(self.gripper_calibration)
        return manifest

    def write_manifest(self, root: Path) -> None:
        path = root / "meta" / CONTRACT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def validate_root(self, root: Path) -> dict[str, Any]:
        recorded_manifest = json.loads((root / "meta" / CONTRACT_FILENAME).read_text())
        expected_manifest = self.manifest()
        recorded_calibration = recorded_manifest.pop("gripper_calibration", None)
        expected_calibration = expected_manifest.pop("gripper_calibration", None)
        if recorded_manifest != expected_manifest:
            raise ValueError("dataset manifest differs from selected strategy schema")
        if expected_calibration is not None and recorded_calibration != expected_calibration:
            raise ValueError("dataset gripper calibration differs from selected hardware")
        if recorded_calibration is not None:
            recorded_calibration = self._validate_gripper_calibration(recorded_calibration)
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
        report = {
            "episodes": int(info["total_episodes"]),
            "frames": int(info["total_frames"]),
            "contract_id": self.contract_id,
            "videos": video_count,
        }
        if recorded_calibration is not None:
            report["gripper_calibration"] = recorded_calibration
        if self.gripper_contract == CONTINUOUS_GRIPPER_CONTRACT:
            import pyarrow.parquet as pq

            state_index = self.state_names.index("gripper.position")
            action_index = self.action_names.index("gripper.target_position")
            actual, target = [], []
            for path in sorted((root / "data").rglob("*.parquet")):
                table = pq.read_table(path, columns=["observation.state", "action"])
                states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
                actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
                actual.extend(states[:, state_index].tolist())
                target.extend(actions[:, action_index].tolist())
            report["continuous_gripper"] = validate_continuous_gripper_samples(
                actual,
                target,
            )
        return report


class ConfiguredDatasetWriter:
    def __init__(
        self,
        dataset: Any,
        root: Path,
        contract: ConfiguredDatasetContract,
        *,
        pi05_v21_sidecar: Any | None = None,
        active_aliases: Mapping[Path, Path] | None = None,
        auto_merge_config: Mapping[str, Any] | None = None,
        input_schema: str | Path | None = None,
    ) -> None:
        self._dataset, self._root, self.contract = dataset, root, contract
        self.pi05_v21_sidecar = pi05_v21_sidecar
        self.auto_merge_config = auto_merge_config
        self.input_schema = input_schema
        self.active_aliases = tuple((Path(alias), Path(target)) for alias, target in (active_aliases or {}).items())
        self.validation_report: dict[str, Any] | None = None
        self.incremental_v21_report: dict[str, Any] | None = None
        contract.write_manifest(root)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.contract.validate_frame(frame)
        prepared = (
            self.pi05_v21_sidecar.prepare_frame(frame)
            if self.pi05_v21_sidecar is not None
            else None
        )
        self._dataset.add_frame(frame)
        if self.pi05_v21_sidecar is not None:
            self.pi05_v21_sidecar.accept_prepared_frame(prepared)

    def save_episode(self, *args: Any, **kwargs: Any) -> None:
        self._dataset.save_episode(*args, **kwargs)
        if self.pi05_v21_sidecar is not None:
            self.pi05_v21_sidecar.commit_episode()

    def clear_episode_buffer(self, *args: Any, **kwargs: Any) -> None:
        self._dataset.clear_episode_buffer(*args, **kwargs)
        if self.pi05_v21_sidecar is not None:
            self.pi05_v21_sidecar.discard_episode()

    def finalize(self) -> None:
        self._dataset.finalize()
        episodes = int(self._dataset.meta.total_episodes)
        if self.pi05_v21_sidecar is not None:
            self.incremental_v21_report = self.pi05_v21_sidecar.finalize(
                expected_episodes=episodes
            )
        if episodes == 0:
            if self._root.exists():
                shutil.rmtree(self._root)
            return
        self.validation_report = self.contract.validate_root(self._root)
        for alias, target in self.active_aliases:
            _replace_dataset_alias(alias, target)
        if self.auto_merge_config is not None and self.input_schema is not None:
            from .auto_merge import update_wrist_aggregate

            update_wrist_aggregate(
                self._root,
                dataset_config=self.auto_merge_config,
                input_schema=self.input_schema,
            )


def create_configured_dataset(
    *,
    repo_id: str,
    root: Path,
    contract: ConfiguredDatasetContract,
    pi05_v21_sidecar: Any | None = None,
    active_aliases: Mapping[Path, Path] | None = None,
    auto_merge_config: Mapping[str, Any] | None = None,
    input_schema: str | Path | None = None,
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
    return ConfiguredDatasetWriter(
        dataset,
        root,
        contract,
        pi05_v21_sidecar=pi05_v21_sidecar,
        active_aliases=active_aliases,
        auto_merge_config=auto_merge_config,
        input_schema=input_schema,
    )
