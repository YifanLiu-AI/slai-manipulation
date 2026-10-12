"""Episode recording loop for canonical three-camera LeRobot v3 data."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from slai_mi.datasets.lerobot_v3.contract import validate_frame
from slai_mi.datasets.lerobot_v3.schema import (
    ACTION,
    ACTUAL_TCP_SPEED,
    CAMERA_SCHEMAS,
    CAMERA_SKEW_MS,
    CAPTURE_SCHEMA,
    COMMAND_SOURCE_NAME,
    DEVICE_TIMESTAMPS_S,
    HOST_RECEIVE_TIMESTAMPS_S,
    INPUT_SCHEMA,
    OBSERVATION_STATE,
    OBSERVATION_TCP_POSE,
    SOURCE_AGE_MS,
    SOURCE_DROPS,
    SOURCE_RESTARTS,
    SOURCE_SEQUENCES,
    SPACEMOUSE_AXES,
    SPACEMOUSE_BUTTONS,
    STATE_SOURCE_NAMES,
    UR5_TARGET_QD,
    VALIDITY_MASK,
)
from slai_mi.input_schema import compose_capture_vector
from slai_mi.rotation import convert_tcp_pose


@dataclass(frozen=True)
class SourceSample:
    """Transport-neutral sample metadata used to align device streams."""

    value: Any
    device_timestamp_s: float
    host_timestamp_s: float
    sequence: int
    dropped_before: int = 0
    restart_count: int = 0
    valid: bool = True


@dataclass(frozen=True)
class SynchronizedInputs:
    cameras: dict[str, SourceSample]
    channels: dict[str, SourceSample]

    @property
    def ur5(self) -> SourceSample:
        return self.channels["ur5"]

    @property
    def wuji(self) -> SourceSample:
        return self.channels["wuji"]

    @property
    def spacemouse(self) -> SourceSample:
        return self.channels[COMMAND_SOURCE_NAME]


def assemble_frame(inputs: SynchronizedInputs, task_prompt: str, *, now: float | None = None) -> dict[str, object]:
    """Build and validate one canonical frame from timestamp-aligned inputs."""
    current = time.monotonic() if now is None else float(now)
    camera_sources = tuple(inputs.cameras[str(camera["role"])] for camera in CAMERA_SCHEMAS)
    state_sources = tuple(inputs.channels[name] for name in STATE_SOURCE_NAMES)
    sources = (*camera_sources, *state_sources, inputs.channels[COMMAND_SOURCE_NAME])
    ur5 = inputs.ur5.value
    mouse = inputs.spacemouse.value
    frame: dict[str, object] = {
        **{
            str(camera["dataset_key"]): np.asarray(inputs.cameras[str(camera["role"])].value)
            for camera in CAMERA_SCHEMAS
        },
        OBSERVATION_STATE: compose_capture_vector(
            INPUT_SCHEMA,
            "state",
            {name: inputs.channels[name].value for name in STATE_SOURCE_NAMES},
        ),
        OBSERVATION_TCP_POSE: convert_tcp_pose(
            ur5.actual_tcp_pose,
            source_representation=str(CAPTURE_SCHEMA["tcp_pose"]["source_representation"]),
            target_representation=str(CAPTURE_SCHEMA["tcp_pose"]["dataset_representation"]),
        ),
        ACTION: compose_capture_vector(
            INPUT_SCHEMA,
            "action",
            {name: inputs.channels[name].value for name in STATE_SOURCE_NAMES},
        ),
        UR5_TARGET_QD: np.asarray(ur5.target_qd, dtype=np.float32),
        ACTUAL_TCP_SPEED: np.asarray(ur5.actual_tcp_speed, dtype=np.float32),
        CAMERA_SKEW_MS: np.asarray(
            [
                abs(
                    inputs.cameras[str(camera["role"])].host_timestamp_s
                    - inputs.cameras[str(CAPTURE_SCHEMA["primary_timeline_role"])].host_timestamp_s
                )
                * 1000.0
                for camera in CAMERA_SCHEMAS
                if camera["role"] != CAPTURE_SCHEMA["primary_timeline_role"]
            ],
            dtype=np.float32,
        ),
        SOURCE_AGE_MS: np.asarray(
            [max(0.0, current - source.host_timestamp_s) * 1000.0 for source in sources],
            dtype=np.float32,
        ),
        DEVICE_TIMESTAMPS_S: np.asarray(
            [source.device_timestamp_s for source in sources], dtype=np.float64
        ),
        HOST_RECEIVE_TIMESTAMPS_S: np.asarray(
            [source.host_timestamp_s for source in sources], dtype=np.float64
        ),
        SOURCE_SEQUENCES: np.asarray([source.sequence for source in sources], dtype=np.int64),
        SOURCE_DROPS: np.asarray([source.dropped_before for source in sources], dtype=np.int64),
        SOURCE_RESTARTS: np.asarray([source.restart_count for source in sources], dtype=np.int64),
        VALIDITY_MASK: np.asarray([source.valid for source in sources], dtype=np.int64),
        SPACEMOUSE_AXES: np.asarray(mouse.axes, dtype=np.float32),
        SPACEMOUSE_BUTTONS: np.asarray(mouse.buttons, dtype=np.int64),
        "task": task_prompt,
    }
    validate_frame(frame)
    return frame


@dataclass
class EpisodeRecorder:
    dataset: Any
    synchronizer: Any
    task_prompt: str
    timeout_s: float = 1.0
    monotonic: Any = time.monotonic
    on_frame: Any | None = None
    frame_count: int = field(default=0, init=False)

    def record(self, stop_event: threading.Event) -> int:
        """Record until requested to stop; errors propagate to the owning app."""
        self.frame_count = 0
        while not stop_event.is_set():
            inputs = self.synchronizer.read(timeout_s=self.timeout_s)
            frame = assemble_frame(inputs, self.task_prompt, now=self.monotonic())
            self.dataset.add_frame(frame)
            if self.on_frame is not None:
                self.on_frame(frame)
            self.frame_count += 1
        return self.frame_count
