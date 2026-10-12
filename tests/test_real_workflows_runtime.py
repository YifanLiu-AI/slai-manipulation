from __future__ import annotations

import threading
from contextlib import contextmanager

import pytest

from slai_mi.runtime import (
    CollectionDependencies,
    RealCollectionWorkflow,
    RealTeleopWorkflow,
    TeleopDependencies,
    validate_real_hardware_config,
)


def hardware_config(*, cameras: bool = True) -> dict:
    return {
        "configured": True,
        "ur5": {"enabled": True, "host": "robot.invalid"},
        "wujihand": {"enabled": True, "usb_serial": "fake"},
        "spacemouse": {"enabled": True},
        "cameras": {
            "enabled": cameras,
            "devices": [
                {"role": "primary", "serial": "fake-1"},
                {"role": "secondary", "serial": "fake-2"},
                {"role": "wrist", "serial": "fake-3"},
            ],
        },
    }


def test_config_gate_runs_before_factories() -> None:
    config = hardware_config()
    config["configured"] = False
    with pytest.raises(ValueError, match="configured must be true"):
        validate_real_hardware_config(config, required=("ur5",))


def test_teleop_supervises_workers_and_closes_input() -> None:
    events: list[str] = []

    @contextmanager
    def mouse_factory(_config):
        events.append("mouse-open")
        try:
            yield object()
        finally:
            events.append("mouse-close")

    class Runtime:
        def __init__(self, name: str) -> None:
            self.name = name

        def run(self, stop: threading.Event) -> None:
            events.append(f"{self.name}-run")
            stop.set()

    dependencies = TeleopDependencies(
        ur5_factory=lambda _config, _mouse, _stop: Runtime("ur5"),
        wuji_factory=lambda _config, _mouse, _stop: Runtime("wuji"),
        spacemouse_factory=mouse_factory,
        preflight=lambda _config: events.append("preflight"),
    )
    RealTeleopWorkflow(hardware_config(), dependencies).run()
    assert events[0:2] == ["preflight", "mouse-open"]
    assert "ur5-run" in events
    assert "wuji-run" in events
    assert events[-1] == "mouse-close"


def test_collection_saves_episode_and_finalizes_all_resources() -> None:
    events: list[str] = []

    class Mouse:
        def __init__(self) -> None:
            self.states = [
                {0: True},
                {0: False},
                {1: True},
            ]

        def state(self):
            buttons = self.states.pop(0)
            return (), buttons

    @contextmanager
    def resource(name: str, value=None):
        events.append(f"{name}-open")
        try:
            yield value if value is not None else object()
        finally:
            events.append(f"{name}-close")

    class Dataset:
        def save_episode(self) -> None:
            events.append("save")

        def clear_episode_buffer(self) -> None:
            events.append("clear")

        def finalize(self) -> None:
            events.append("finalize")

    class Recorder:
        def record(self, stop: threading.Event) -> None:
            events.append("record")
            stop.wait()

    mouse = Mouse()
    dependencies = CollectionDependencies(
        ur5_factory=lambda _config: resource("ur5"),
        wuji_factory=lambda _config: resource("wuji"),
        spacemouse_factory=lambda _config: resource("mouse", mouse),
        cameras_factory=lambda _config: resource("cameras"),
        dataset_factory=lambda _dataset, _task: Dataset(),
        synchronizer_factory=lambda _sources, _dataset: object(),
        recorder_factory=lambda _dataset, _sync, _prompt: Recorder(),
        sleep=lambda _seconds: None,
    )
    workflow = RealCollectionWorkflow(
        hardware_config(),
        {"format": "lerobot_v3"},
        {"task": {"instruction": "Do the task."}},
        dependencies,
        episode_limit=1,
    )
    assert workflow.run() == 1
    assert "record" in events
    assert events.count("save") == 1
    assert events[-6:] == [
        "clear",
        "finalize",
        "cameras-close",
        "mouse-close",
        "wuji-close",
        "ur5-close",
    ]
    assert events[-1] == "ur5-close"


def test_collection_rejects_missing_camera_identity_before_opening() -> None:
    config = hardware_config()
    config["cameras"]["devices"][1]["serial"] = None
    opened = False

    def forbidden(_config):
        nonlocal opened
        opened = True
        raise AssertionError("factory must not run")

    dependencies = CollectionDependencies(
        ur5_factory=forbidden,
        wuji_factory=forbidden,
        spacemouse_factory=forbidden,
        cameras_factory=forbidden,
        dataset_factory=lambda _dataset, _task: None,
        synchronizer_factory=lambda _sources, _dataset: None,
        recorder_factory=lambda _dataset, _sync, _prompt: None,
    )
    workflow = RealCollectionWorkflow(
        config,
        {},
        {"task": {"instruction": "task"}},
        dependencies,
        episode_limit=1,
    )
    with pytest.raises(ValueError, match="requires a serial"):
        workflow.run()
    assert not opened


def test_collection_dashboard_receives_workflow_lifecycle(monkeypatch) -> None:
    events: list[str] = []

    class Provider:
        def set_phase(self, phase, _label, **_fields):
            events.append(f"phase:{phase}")

        def event(self, _message, *, code="collection", **_fields):
            events.append(f"event:{code}")

        def set_dataset_path(self, path):
            events.append(f"dataset:{path}")

        def start_episode(self, *, index, attempt):
            events.append(f"start:{index}:{attempt}")

        def finish_episode(self, action, *, index):
            events.append(f"finish:{action}:{index}")

        def record_frame(self, _frame):
            return

        def observe_inputs(self, _inputs):
            events.append("observe")

        def observe_spacemouse(self, _motion, _buttons):
            events.append("observe-mouse")

    class Dashboard:
        url = "http://127.0.0.1:8765"

        def __init__(self, *_args, **_kwargs):
            self.provider = Provider()

        def __enter__(self):
            events.append("dashboard-open")
            return self

        def __exit__(self, *_args):
            events.append("dashboard-close")

    import slai_mi.ui.collection_dashboard as dashboard_module

    monkeypatch.setattr(dashboard_module, "CollectionDashboard", Dashboard)

    class Mouse:
        def __init__(self):
            self.states = [{}, {0: True}, {0: False}, {1: True}]

        def state(self):
            return (), self.states.pop(0)

    class Dataset:
        _root = "/tmp/dashboard-dataset"

        def save_episode(self):
            events.append("save")

        def clear_episode_buffer(self):
            events.append("clear")

        def finalize(self):
            events.append("finalize")

    class Synchronizer:
        def read(self, timeout_s=1.0):
            events.append(f"sync:{timeout_s}")
            return object()

    class Recorder:
        on_frame = None

        def record(self, stop):
            events.append("record")
            stop.wait()

    @contextmanager
    def resource(value=None):
        yield object() if value is None else value

    dependencies = CollectionDependencies(
        ur5_factory=lambda _config: resource(),
        wuji_factory=lambda _config: resource(),
        spacemouse_factory=lambda _config: resource(Mouse()),
        cameras_factory=lambda _config: resource(),
        dataset_factory=lambda _dataset, _task: Dataset(),
        synchronizer_factory=lambda _sources, _dataset: Synchronizer(),
        recorder_factory=lambda _dataset, _sync, _prompt: Recorder(),
        sleep=lambda _seconds: None,
    )
    workflow = RealCollectionWorkflow(
        hardware_config(),
        {},
        {"task": {"instruction": "task"}},
        dependencies,
        episode_limit=1,
        dashboard_enabled=True,
    )
    assert workflow.run() == 1
    assert events[0:2] == ["dashboard-open", "phase:preflight"]
    assert "sync:5.0" in events
    assert "phase:ready" in events
    assert "start:1:1" in events
    assert "phase:saving" in events
    assert "finish:save:1" in events
    assert events[-1] == "dashboard-close"


class _HomeMouse:
    def __init__(self, states: list[dict[int, bool]], *, at_home: bool = True) -> None:
        self.states = list(states)
        self.at_home = at_home
        self.home_requests = 0
        self.home_clears = 0

    def state(self):
        if not self.states:
            raise RuntimeError("test mouse state sequence exhausted")
        return (), self.states.pop(0)

    def request_home(self) -> None:
        self.home_requests += 1

    def clear_home(self) -> None:
        self.home_clears += 1

    def home_status(self) -> dict[str, object]:
        return {"at_home": self.at_home, "detail": "test home status"}


def _home_collection_workflow(
    mouse: _HomeMouse,
    events: list[str],
    *,
    frame_counts: list[int] | None = None,
) -> RealCollectionWorkflow:
    @contextmanager
    def resource(value=None):
        yield object() if value is None else value

    class Dataset:
        def save_episode(self) -> None:
            events.append("save")

        def clear_episode_buffer(self) -> None:
            events.append("clear")

        def finalize(self) -> None:
            events.append("finalize")

    class Synchronizer:
        def read(self, timeout_s=1.0):
            events.append(f"sync:{timeout_s}")
            return object()

    class Recorder:
        def __init__(self) -> None:
            self.calls = 0
            self.frame_count = 1

        def record(self, stop: threading.Event) -> None:
            self.calls += 1
            if frame_counts is not None:
                self.frame_count = frame_counts[self.calls - 1]
            events.append(f"record:{self.calls}")
            stop.wait()

    dependencies = CollectionDependencies(
        ur5_factory=lambda _config: resource(),
        wuji_factory=lambda _config: resource(),
        spacemouse_factory=lambda _config: resource(mouse),
        cameras_factory=lambda _config: resource(),
        dataset_factory=lambda _dataset, _task: Dataset(),
        synchronizer_factory=lambda _sources, _dataset: Synchronizer(),
        recorder_factory=lambda _dataset, _sync, _prompt: Recorder(),
        sleep=lambda _seconds: None,
    )
    return RealCollectionWorkflow(
        hardware_config(),
        {},
        {"task": {"instruction": "task"}},
        dependencies,
        episode_limit=1,
    )


def test_collection_save_waits_for_final_home_before_exit() -> None:
    events: list[str] = []
    mouse = _HomeMouse(
        [
            {},  # Initial supervised state and input synchronization.
            {},  # Initial home completes.
            {0: True},  # Menu starts recording.
            {},
            {1: True},  # Fit saves.
            {},  # Final home completes before workflow exit.
        ]
    )

    workflow = _home_collection_workflow(mouse, events)
    assert workflow.run() == 1
    assert events.count("record:1") == 1
    assert events.count("save") == 1
    assert mouse.home_requests == 2
    assert mouse.home_clears == 2
    assert mouse.states == []


def test_collection_discard_homes_then_automatically_restarts() -> None:
    events: list[str] = []
    mouse = _HomeMouse(
        [
            {},
            {},  # Initial home completes.
            {0: True},
            {},
            {22: True},  # Esc discards the first attempt.
            {},  # Home completes and attempt two auto-starts.
            {1: True},  # Fit saves attempt two.
            {},  # Final home completes.
        ]
    )

    workflow = _home_collection_workflow(mouse, events)
    assert workflow.run() == 1
    assert events.count("record:1") == 1
    assert events.count("record:2") == 1
    assert events.count("save") == 1
    assert events.count("clear") == 2  # Operator discard plus final cleanup.
    assert mouse.home_requests == 3
    assert mouse.home_clears == 3


def test_collection_zero_frame_fit_discards_homes_and_restarts() -> None:
    events: list[str] = []
    mouse = _HomeMouse(
        [
            {},
            {},
            {0: True},
            {},
            {1: True},  # First Fit has no synchronized frames.
            {},  # Home and automatic retry.
            {1: True},  # Second Fit saves one frame.
            {},
        ]
    )

    workflow = _home_collection_workflow(mouse, events, frame_counts=[0, 1])
    assert workflow.run() == 1
    assert events.count("record:1") == 1
    assert events.count("record:2") == 1
    assert events.count("save") == 1
    assert events.count("clear") == 2
    assert mouse.home_requests == 3
    assert mouse.home_clears == 3


def test_collection_home_timeout_blocks_recording(monkeypatch) -> None:
    import slai_mi.runtime.real_workflows as workflow_module
    import slai_mi.ui.collection_dashboard as dashboard_module

    events: list[str] = []

    class Mouse(_HomeMouse):
        def state(self):
            if not self.states:
                raise RuntimeError("stop after timeout")
            return super().state()

    class Provider:
        def set_phase(self, phase, _label, **_fields):
            events.append(f"phase:{phase}")

        def event(self, _message, *, code="collection", **_fields):
            events.append(f"event:{code}")

        def set_dataset_path(self, _path):
            return

        def observe_spacemouse(self, _motion, _buttons):
            return

        def record_frame(self, _frame):
            return

    class Dashboard:
        url = "http://127.0.0.1:8765"

        def __init__(self, *_args, **_kwargs):
            self.provider = Provider()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return

    mouse = Mouse([{}, {}], at_home=False)
    monkeypatch.setattr(workflow_module, "HOME_TIMEOUT_S", 0.0)
    monkeypatch.setattr(dashboard_module, "CollectionDashboard", Dashboard)
    monkeypatch.setattr(
        dashboard_module, "DashboardSynchronizer", lambda synchronizer, _provider: synchronizer
    )
    workflow = _home_collection_workflow(mouse, events)
    workflow.dashboard_enabled = True

    with pytest.raises(RuntimeError, match="stop after timeout"):
        workflow.run()
    assert "phase:blocked" in events
    assert "event:home_timeout" in events
    assert not any(event.startswith("record:") for event in events)
    assert mouse.home_requests == 1
    assert mouse.home_clears == 1
