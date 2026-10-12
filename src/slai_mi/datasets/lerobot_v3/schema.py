"""Minimal LeRobot v3 contract for UR5 + Wuji demonstrations."""

from __future__ import annotations

import numpy as np

from slai_mi.input_schema import (
    capture_vector_names,
    compose_capture_vector,
    enabled_cameras,
    load_input_schema,
)

RECORDED_BUTTON_NAMES = (
    "menu", "fit", "r", "f", "one", "two", "three", "home", "esc", "alt", "shift", "ctrl"
)

INPUT_SCHEMA = load_input_schema()
CAPTURE_SCHEMA = INPUT_SCHEMA["capture"]
CAMERA_SCHEMAS = enabled_cameras(INPUT_SCHEMA)
FPS = int(CAPTURE_SCHEMA["fps"])
MAX_CAMERA_SKEW_MS = 20.0
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = map(int, CAPTURE_SCHEMA["image_shape"])

OBSERVATION_STATE = "observation.state"
TCP_POSE_SCHEMA = CAPTURE_SCHEMA["tcp_pose"]
OBSERVATION_TCP_POSE = str(TCP_POSE_SCHEMA["key"])
ACTION = "action"
UR5_TARGET_QD = "telemetry.ur5_target_qd"
ACTUAL_TCP_SPEED = "telemetry.actual_tcp_speed"
CAMERA_SKEW_MS = "telemetry.camera_skew_ms"
SOURCE_AGE_MS = "telemetry.source_age_ms"
DEVICE_TIMESTAMPS_S = "telemetry.device_timestamps_s"
HOST_RECEIVE_TIMESTAMPS_S = "telemetry.host_receive_timestamps_s"
SOURCE_SEQUENCES = "telemetry.source_sequence_numbers"
SOURCE_DROPS = "telemetry.source_dropped_before"
SOURCE_RESTARTS = "telemetry.source_restart_counts"
VALIDITY_MASK = "telemetry.validity_mask"
SPACEMOUSE_AXES = "telemetry.spacemouse_axes"
SPACEMOUSE_BUTTONS = "telemetry.spacemouse_buttons"
CAMERA_SOURCE_NAMES = tuple(str(camera["role"]) for camera in CAMERA_SCHEMAS)
STATE_SOURCE_NAMES = tuple(
    str(item["name"]) for item in INPUT_SCHEMA["synchronization"]["state_channels"]
)
COMMAND_SOURCE_NAME = str(INPUT_SCHEMA["synchronization"]["command_channel"]["name"])
SOURCE_NAMES = (*CAMERA_SOURCE_NAMES, *STATE_SOURCE_NAMES, COMMAND_SOURCE_NAME)
SPACEMOUSE_BUTTON_NAMES = RECORDED_BUTTON_NAMES

UR5_JOINT_NAMES = tuple(
    name.removeprefix("ur5.").removesuffix(".position")
    for component in CAPTURE_SCHEMA["state"]["components"]
    if component["channel"] == "ur5"
    for name in component["names"]
)
WUJI_JOINT_NAMES = tuple(
    name.removeprefix("wuji.").removesuffix(".position")
    for component in CAPTURE_SCHEMA["state"]["components"]
    if component["channel"] == "wuji"
    for name in component["names"]
)
STATE_NAMES = capture_vector_names(INPUT_SCHEMA, "state")
TCP_POSE_NAMES = tuple(str(name) for name in TCP_POSE_SCHEMA["names"])
TCP_TWIST_NAMES = ("vx", "vy", "vz", "wx", "wy", "wz")
ACTION_NAMES = capture_vector_names(INPUT_SCHEMA, "action")
STATE_DIM = len(STATE_NAMES)
ACTION_DIM = len(ACTION_NAMES)


def lerobot_features() -> dict[str, dict]:
    """Return the exact v3 feature declaration used by the real-data writer."""
    rgb = {
        "dtype": "video",
        "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": False},
    }
    return {
        **{
            str(camera["dataset_key"]): {**rgb, "info": dict(rgb["info"])}
            for camera in CAMERA_SCHEMAS
        },
        OBSERVATION_STATE: {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": list(STATE_NAMES),
        },
        OBSERVATION_TCP_POSE: {
            "dtype": "float32",
            "shape": (len(TCP_POSE_NAMES),),
            "names": list(TCP_POSE_NAMES),
        },
        ACTION: {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": list(ACTION_NAMES),
        },
        UR5_TARGET_QD: {
            "dtype": "float32",
            "shape": (6,),
            "names": list(UR5_JOINT_NAMES),
        },
        ACTUAL_TCP_SPEED: {
            "dtype": "float32",
            "shape": (6,),
            "names": list(TCP_TWIST_NAMES),
        },
        CAMERA_SKEW_MS: {
            "dtype": "float32",
            "shape": (max(0, len(CAMERA_SOURCE_NAMES) - 1),),
            "names": [
                f"{role}_to_{CAPTURE_SCHEMA['primary_timeline_role']}"
                for role in CAMERA_SOURCE_NAMES
                if role != CAPTURE_SCHEMA["primary_timeline_role"]
            ],
        },
        SOURCE_AGE_MS: {
            "dtype": "float32",
            "shape": (len(SOURCE_NAMES),),
            "names": list(SOURCE_NAMES),
        },
        DEVICE_TIMESTAMPS_S: {
            "dtype": "float64",
            "shape": (len(SOURCE_NAMES),),
            "names": list(SOURCE_NAMES),
        },
        HOST_RECEIVE_TIMESTAMPS_S: {
            "dtype": "float64",
            "shape": (len(SOURCE_NAMES),),
            "names": list(SOURCE_NAMES),
        },
        SOURCE_SEQUENCES: {
            "dtype": "int64",
            "shape": (len(SOURCE_NAMES),),
            "names": list(SOURCE_NAMES),
        },
        SOURCE_DROPS: {
            "dtype": "int64",
            "shape": (len(SOURCE_NAMES),),
            "names": list(SOURCE_NAMES),
        },
        SOURCE_RESTARTS: {
            "dtype": "int64",
            "shape": (len(SOURCE_NAMES),),
            "names": list(SOURCE_NAMES),
        },
        VALIDITY_MASK: {
            "dtype": "int64",
            "shape": (len(SOURCE_NAMES),),
            "names": list(SOURCE_NAMES),
        },
        SPACEMOUSE_AXES: {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x", "y", "z", "rx", "ry", "rz"],
        },
        SPACEMOUSE_BUTTONS: {
            "dtype": "int64",
            "shape": (12,),
            "names": list(SPACEMOUSE_BUTTON_NAMES),
        },
    }


def compose_state(ur5_actual_q: object, wuji_actual_q: object) -> np.ndarray:
    from types import SimpleNamespace

    return compose_capture_vector(
        INPUT_SCHEMA,
        "state",
        {
            "ur5": SimpleNamespace(actual_q=ur5_actual_q),
            "wuji": SimpleNamespace(actual_q=wuji_actual_q),
        },
    )


def compose_action(ur5_target_tcp_speed: object, wuji_target_q: object) -> np.ndarray:
    from types import SimpleNamespace

    return compose_capture_vector(
        INPUT_SCHEMA,
        "action",
        {
            "ur5": SimpleNamespace(target_tcp_speed=ur5_target_tcp_speed),
            "wuji": SimpleNamespace(command_q=wuji_target_q),
        },
    )
