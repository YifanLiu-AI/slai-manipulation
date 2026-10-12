from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

import yaml

from slai_mi.runtime import CollectionDependencies, RealCollectionWorkflow

ROOT = Path(__file__).resolve().parents[1]


def test_last_point_preserves_parent_schema_and_gripper_allocation() -> None:
    task_path = ROOT / "configs/tasks/task2_continuous.yaml"
    child_path = ROOT / "configs/poses/tasks/last-point.yaml"
    child = yaml.safe_load(child_path.read_text(encoding="utf-8"))
    parent = yaml.safe_load(
        (child_path.parent / "task2_continuous_start.yaml").read_text(encoding="utf-8")
    )

    assert child_path.name == "last-point.yaml"
    assert child["schema_ref"] == parent["schema_ref"]
    assert child["joint_names"] == parent["joint_names"]
    assert child["joint_positions"][8] == parent["joint_positions"][8] == 0.0
    assert child["provenance"]["allocation_contract"] == {
        "ur5_6dof": [0, 1, 2, 3, 4, 5],
        "wrist_2dof": [6, 7],
        "gripper_1dof": [8],
    }
    assert len(child["joint_positions"]) == 9


def test_menu_waits_for_effective_descent_before_recording_and_fit_saves_after_home() -> None:
    events: list[str] = []
    zero = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    up = (0.0, 0.0, 0.4, 0.0, 0.0, 0.0)
    down = (0.0, 0.0, -0.4, 0.0, 0.0, 0.0)

    class Mouse:
        def __init__(self) -> None:
            self.states = [
                (zero, {}),       # supervised input synchronization
                (zero, {}),       # initial HOME completes
                (zero, {0: True}),# Menu requests fresh HOME
                (zero, {}),       # pre-Episode HOME completes: armed, not recording
                (zero, {}),       # stationary wait must not record
                (up, {}),         # upward motion must not record
                (down, {}),       # first effective descent starts recorder
                (zero, {1: True}),# Fit requests post-Episode HOME
                (zero, {}),       # HOME completes, then save
            ]
            self.home_requests = 0

        def state(self):
            motion, buttons = self.states.pop(0)
            events.append("input:down" if motion[2] < 0 else "input:not-down")
            return motion, buttons

        def request_home(self) -> None:
            self.home_requests += 1
            events.append("home:open")

        def request_home_preserve_gripper(self) -> None:
            self.home_requests += 1
            events.append("home:preserve-gripper")

        def clear_home(self) -> None:
            events.append("home-clear")

        def home_status(self):
            return {"at_home": True, "detail": "test"}

        def consume_external_home_request(self) -> bool:
            return False

    @contextmanager
    def opened(value=None):
        yield object() if value is None else value

    class Dataset:
        def save_episode(self): events.append("save")
        def clear_episode_buffer(self): events.append("clear")
        def finalize(self): events.append("finalize")

    class Synchronizer:
        def read(self, timeout_s=1.0): return object()

    class Recorder:
        frame_count = 1
        def record(self, stop: threading.Event) -> None:
            events.append("record")
            stop.wait()
            events.append("record-stop")

    mouse = Mouse()
    dependencies = CollectionDependencies(
        ur5_factory=lambda _config: opened(),
        wuji_factory=lambda _config: opened(),
        spacemouse_factory=lambda _config: opened(mouse),
        cameras_factory=lambda _config: opened(),
        dataset_factory=lambda *_args: Dataset(),
        synchronizer_factory=lambda *_args: Synchronizer(),
        recorder_factory=lambda *_args: Recorder(),
        sleep=lambda _seconds: None,
    )
    hardware = {
        "configured": True,
        "ur5": {"enabled": True, "host": "robot.invalid"},
        "wujihand": {"enabled": True, "usb_serial": "fake"},
        "spacemouse": {"enabled": True},
        "cameras": {"enabled": True, "devices": [
            {"role": "primary", "serial": "1"},
            {"role": "secondary", "serial": "2"},
            {"role": "wrist", "serial": "3"},
        ]},
    }
    workflow = RealCollectionWorkflow(
        hardware, {}, {"task": {"instruction": "task"}}, dependencies, episode_limit=1
    )

    assert workflow.run() == 1
    assert mouse.states == []
    assert mouse.home_requests == 3
    assert events.count("home:open") == 2  # startup + Menu
    assert events.count("home:preserve-gripper") == 1  # Fit only
    down_index = events.index("input:down")
    record_index = events.index("record")
    assert record_index > down_index
    assert "record" not in events[:down_index]
    assert events.index("record") < events.index("home:preserve-gripper")
    assert (
        events.index("home:preserve-gripper")
        < events.index("record-stop")
        < events.index("save")
    )
    assert events.index("home-clear", events.index("record")) < events.index("record-stop")
