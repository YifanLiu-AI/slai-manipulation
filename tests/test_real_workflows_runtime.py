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


def test_teleop_runs_only_named_strategy_workers() -> None:
    events: list[str] = []

    @contextmanager
    def mouse_factory(_config):
        events.append("mouse-open")
        try:
            yield object()
        finally:
            events.append("mouse-close")

    class Runtime:
        def __init__(self, name: str, *, finish: bool = False) -> None:
            self.name = name
            self.finish = finish

        def run(self, stop: threading.Event) -> None:
            events.append(f"{self.name}-run")
            if self.finish:
                return
            stop.wait(1.0)
            events.append(f"{self.name}-stop")

    def forbidden(*_args):
        raise AssertionError("legacy factory must not be used")

    dependencies = TeleopDependencies(
        ur5_factory=forbidden,
        wuji_factory=forbidden,
        spacemouse_factory=mouse_factory,
        runtime_factories={
            "ur5-teleop": lambda *_args: Runtime("ur5"),
            "wrist-teleop": lambda *_args: Runtime("wrist", finish=True),
        },
        required_devices=("ur5", "wrist_sensor", "spacemouse"),
    )
    config = {
        "configured": True,
        "ur5": {"enabled": True, "host": "robot.invalid"},
        "wrist_sensor": {"enabled": True},
        "spacemouse": {"enabled": True},
    }

    RealTeleopWorkflow(config, dependencies).run()

    assert "ur5-run" in events
    assert "wrist-run" in events
    assert "ur5-stop" in events
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


def test_three_menu_fit_cycles_start_then_exit_continuous_collection() -> None:
    events: list[str] = []
    chord = {0: True, 1: True}

    class Mouse:
        def __init__(self):
            # The held startup chord is ignored until both buttons are released.
            self.states = [
                chord, {}, chord, {}, chord, {}, chord,
                {}, chord, {}, chord, {}, chord,
            ]

        def state(self):
            return (), self.states.pop(0)

    @contextmanager
    def resource(value=None):
        yield object() if value is None else value

    class Dataset:
        def save_episode(self):
            raise AssertionError("exit gesture must not save an episode")

        def clear_episode_buffer(self):
            events.append("clear")

        def finalize(self):
            events.append("finalize")

    class Synchronizer:
        def read(self, timeout_s=1.0):
            return None

    class Recorder:
        def record(self, stop):
            while not stop.is_set():
                pass

    dependencies = CollectionDependencies(
        ur5_factory=lambda _config: resource(),
        wuji_factory=lambda _config: resource(),
        spacemouse_factory=lambda _config: resource(Mouse()),
        cameras_factory=lambda _config: resource(),
        dataset_factory=lambda _dataset, _task: Dataset(),
        synchronizer_factory=lambda _sources, _dataset: Synchronizer(),
        recorder_factory=lambda *_args: Recorder(),
        sleep=lambda _seconds: None,
    )
    workflow = RealCollectionWorkflow(
        hardware_config(),
        {},
        {"task": {"instruction": "task"}},
        dependencies,
        episode_limit=None,
    )

    assert workflow.run() == 0
    assert events == ["clear", "finalize"]


def test_collection_named_resources_exclude_wuji() -> None:
    events: list[str] = []

    class Mouse:
        def __init__(self):
            self.states = [{0: True}, {0: False}, {1: True}]

        def state(self):
            return (), self.states.pop(0)

    @contextmanager
    def opened(name, value=None):
        events.append(f"{name}-open")
        try:
            yield value if value is not None else object()
        finally:
            events.append(f"{name}-close")

    class Dataset:
        def save_episode(self):
            events.append("save")

        def clear_episode_buffer(self):
            pass

        def finalize(self):
            events.append("finalize")

    class Recorder:
        frame_count = 1

        def record(self, stop):
            stop.wait()

    forbidden = lambda *_args: (_ for _ in ()).throw(AssertionError("legacy factory used"))
    mouse = Mouse()
    dependencies = CollectionDependencies(
        ur5_factory=forbidden,
        wuji_factory=forbidden,
        spacemouse_factory=forbidden,
        cameras_factory=forbidden,
        dataset_factory=lambda *_args: Dataset(),
        synchronizer_factory=lambda *_args: object(),
        recorder_factory=lambda *_args: Recorder(),
        sleep=lambda _seconds: None,
        resource_factories={
            "ur5": lambda _config: opened("ur5"),
            "wrist": lambda _config: opened("wrist"),
            "spacemouse": lambda _config: opened("mouse", mouse),
            "cameras": lambda _config: opened("cameras"),
        },
        required_devices=("ur5", "wrist_sensor", "spacemouse", "cameras"),
    )
    config = hardware_config()
    config["wujihand"]["enabled"] = False
    config["wrist_sensor"] = {"enabled": True}
    workflow = RealCollectionWorkflow(
        config,
        {},
        {"task": {"instruction": "task"}},
        dependencies,
        episode_limit=1,
    )

    assert workflow.run() == 1
    assert "wrist-open" in events
    assert not any(event.startswith("wuji-") for event in events)


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

    def consume_external_home_request(self) -> bool:
        return False


def _home_collection_workflow(
    mouse: _HomeMouse,
    events: list[str],
    *,
    frame_counts: list[int] | None = None,
    episode_limit: int | None = 1,
    idle_camera_timeouts: int = 0,
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
        def __init__(self) -> None:
            self.idle_reads = 0

        def read(self, timeout_s=1.0):
            events.append(f"sync:{timeout_s}")
            if timeout_s == 0.25:
                self.idle_reads += 1
                if self.idle_reads <= idle_camera_timeouts:
                    raise TimeoutError("transient camera phase miss")
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
            events.append(f"record-stop:{self.calls}")

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
        episode_limit=episode_limit,
    )


def test_idle_fit_requests_coordinated_home_without_recording() -> None:
    events: list[str] = []
    mouse = _HomeMouse(
        [
            {},  # Initial supervised state and input synchronization.
            {},  # Initial home completes.
            {1: True},  # Idle Fit requests a coordinated home.
            {},  # Fit home completes; remain idle.
            {26: True},  # LOCK exits without recording.
        ]
    )

    workflow = _home_collection_workflow(mouse, events, episode_limit=None)
    assert workflow.run() == 0
    assert mouse.home_requests == 2
    assert mouse.home_clears == 2
    assert not any(event.startswith("record:") for event in events)
    assert "save" not in events
    assert mouse.states == []


def test_collection_save_waits_for_final_home_before_exit() -> None:
    events: list[str] = []
    mouse = _HomeMouse(
        [
            {},  # Initial supervised state and input synchronization.
            {},  # Initial home completes.
            {0: True},  # Menu starts recording.
            {},
            {1: True},  # Fit starts the final home while recording continues.
            {},  # Final home completes, then recording stops and saves.
        ]
    )
    clear_home = mouse.clear_home

    def observed_clear_home() -> None:
        events.append("home-clear")
        clear_home()

    mouse.clear_home = observed_clear_home

    workflow = _home_collection_workflow(mouse, events)
    assert workflow.run() == 1
    assert events.count("record:1") == 1
    assert events.count("save") == 1
    # Initial HOME + fresh pre-Episode HOME + post-Fit HOME.
    assert mouse.home_requests == 3
    assert mouse.home_clears == 3
    assert mouse.states == []
    assert events.index("save") > max(
        index for index, event in enumerate(events) if event == "home-clear"
    )
    final_home_clear = max(
        index for index, event in enumerate(events) if event == "home-clear"
    )
    # Fit does not stop the data window. HOME telemetry remains in the active
    # Episode, and only HOME completion stops and commits the recorder.
    assert final_home_clear < events.index("record-stop:1") < events.index("save")


def test_low_level_home_discards_active_episode_before_homing() -> None:
    events: list[str] = []

    class Mouse(_HomeMouse):
        def __init__(self) -> None:
            super().__init__([{}, {}, {0: True}, {}, {}, {26: True}])
            self.calls = 0

        def consume_external_home_request(self) -> bool:
            self.calls += 1
            return self.calls == 4

    mouse = Mouse()
    workflow = _home_collection_workflow(mouse, events, episode_limit=None)

    assert workflow.run() == 0
    assert events.count("record:1") == 1
    assert events.count("record-stop:1") == 1
    # One clear is the immediate HOME preemption; finalization defensively
    # clears the already-empty buffer once more.
    assert events.count("clear") == 2
    assert events.index("record-stop:1") < events.index("clear")
    assert "save" not in events
    assert mouse.home_requests == 3


def test_collection_lock_during_post_fit_home_does_not_save_episode() -> None:
    events: list[str] = []

    class Mouse(_HomeMouse):
        def home_status(self) -> dict[str, object]:
            return {
                "at_home": self.home_requests < 3,
                "detail": "returning after Fit",
            }

    mouse = Mouse([{}, {}, {0: True}, {}, {1: True}, {26: True}])
    workflow = _home_collection_workflow(mouse, events)

    assert workflow.run() == 0
    assert "save" not in events
    assert events.count("clear") == 1
    assert events[-1] == "finalize"
    assert mouse.home_requests == 3


def test_collection_lock_after_post_fit_home_saves_before_finalizing() -> None:
    events: list[str] = []
    mouse = _HomeMouse([{}, {}, {0: True}, {}, {1: True}, {26: True}])
    workflow = _home_collection_workflow(mouse, events)

    assert workflow.run() == 1
    assert events.count("save") == 1
    assert events.index("save") < events.index("finalize")
    assert mouse.home_requests == 3


def test_collection_post_fit_home_timeout_discards_without_saving(monkeypatch) -> None:
    import slai_mi.runtime.real_workflows as workflow_module

    events: list[str] = []

    class Mouse(_HomeMouse):
        def home_status(self) -> dict[str, object]:
            return {
                "at_home": self.home_requests < 3,
                "detail": "post-Fit home timeout",
            }

    mouse = Mouse([{}, {}, {0: True}, {}, {1: True}, {}, {26: True}])
    monkeypatch.setattr(workflow_module, "HOME_TIMEOUT_S", 0.0)
    workflow = _home_collection_workflow(mouse, events)

    assert workflow.run() == 0
    assert "save" not in events
    assert events.count("clear") == 2  # Timeout discard plus final cleanup.
    assert events[-1] == "finalize"


def test_continuous_collection_saves_each_episode_only_after_home() -> None:
    events: list[str] = []
    mouse = _HomeMouse(
        [
            {},
            {},  # Initial home completes.
            {0: True},
            {},
            {1: True},
            {},  # Episode 1 home completes, then it is saved.
            {0: True},
            {},
            {1: True},
            {},  # Episode 2 home completes, then it is saved.
            {26: True},  # LOCK finalizes continuous collection.
        ]
    )
    clear_home = mouse.clear_home

    def observed_clear_home() -> None:
        events.append("home-clear")
        clear_home()

    mouse.clear_home = observed_clear_home
    workflow = _home_collection_workflow(mouse, events, episode_limit=None)

    assert workflow.run() == 2
    assert events.count("save") == 2
    save_indexes = [index for index, event in enumerate(events) if event == "save"]
    home_indexes = [index for index, event in enumerate(events) if event == "home-clear"]
    # Initial HOME, then a distinct HOME immediately before and after every
    # Episode. Episode 2 must not inherit Episode 1's completed HOME state.
    assert len(home_indexes) == 5
    assert home_indexes[1] < events.index("record:1") < home_indexes[2] < save_indexes[0]
    assert home_indexes[3] < events.index("record:2") < home_indexes[4] < save_indexes[1]
    assert mouse.home_requests == 5
    assert events[-1] == "finalize"


def test_continuous_collection_homes_before_all_99_episode_starts() -> None:
    events: list[str] = []
    states: list[dict[int, bool]] = [
        {},  # Initial supervised synchronization.
        {},  # Initial HOME completes.
    ]
    for _episode in range(99):
        states.extend(
            [
                {0: True},  # Menu requests the next Episode.
                {},  # Its fresh pre-Episode HOME completes; recording starts.
                {1: True},  # Fit requests the post-Episode HOME.
                {},  # HOME completes; the Episode is saved.
            ]
        )

    mouse = _HomeMouse(states)
    clear_home = mouse.clear_home

    def observed_clear_home() -> None:
        events.append("home-clear")
        clear_home()

    mouse.clear_home = observed_clear_home
    workflow = _home_collection_workflow(mouse, events, episode_limit=99)

    assert workflow.run() == 99
    assert mouse.states == []
    assert mouse.home_requests == 199  # Initial HOME + two HOMEs per Episode.
    assert mouse.home_clears == 199
    assert events.count("save") == 99
    home_indexes = [index for index, event in enumerate(events) if event == "home-clear"]
    save_indexes = [index for index, event in enumerate(events) if event == "save"]
    for episode in range(1, 100):
        pre_home = home_indexes[2 * episode - 1]
        post_home = home_indexes[2 * episode]
        record = events.index(f"record:{episode}")
        save = save_indexes[episode - 1]
        assert pre_home < record < post_home < save


def test_collection_discard_homes_then_waits_for_menu() -> None:
    events: list[str] = []
    mouse = _HomeMouse(
        [
            {},
            {},  # Initial home completes.
            {0: True},
            {},
            {22: True},  # Esc discards the first attempt.
            {},  # Home completes without starting a new recording.
            {},  # Idle input still must not start recording.
            {0: True},  # Operator explicitly starts attempt two.
            {},
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
    # Initial + pre-attempt 1 + discard + pre-attempt 2 + save.
    assert mouse.home_requests == 5
    assert mouse.home_clears == 5


def test_collection_discard_never_restarts_without_another_menu() -> None:
    events: list[str] = []
    mouse = _HomeMouse([{}, {}, {0: True}, {}, {22: True}, {}, {26: True}])
    workflow = _home_collection_workflow(mouse, events)

    assert workflow.run() == 0
    assert events.count("record:1") == 1
    assert "record:2" not in events
    assert "save" not in events


def test_collection_zero_frame_fit_discards_homes_then_waits_for_menu() -> None:
    events: list[str] = []
    mouse = _HomeMouse(
        [
            {},
            {},
            {0: True},
            {},
            {1: True},  # First Fit has no synchronized frames.
            {},  # Home completes without an automatic retry.
            {},
            {0: True},  # Operator explicitly retries.
            {},
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
    # Initial + pre-attempt 1 + zero-frame Fit + pre-attempt 2 + save.
    assert mouse.home_requests == 5
    assert mouse.home_clears == 5


def test_collection_zero_frame_discard_never_restarts_without_menu() -> None:
    events: list[str] = []
    mouse = _HomeMouse([{}, {}, {0: True}, {}, {1: True}, {}, {26: True}])
    workflow = _home_collection_workflow(mouse, events, frame_counts=[0])

    assert workflow.run() == 0
    assert events.count("record:1") == 1
    assert "record:2" not in events
    assert "save" not in events


def test_collection_rotation_lock_finalizes_without_saving_active_episode() -> None:
    events: list[str] = []
    mouse = _HomeMouse([{}, {}, {0: True}, {}, {26: True}])

    workflow = _home_collection_workflow(mouse, events)
    assert workflow.run() == 0
    assert events.count("record:1") == 1
    assert "save" not in events
    assert events.count("clear") == 1
    assert events[-1] == "finalize"


def test_collection_transient_idle_camera_timeout_retries_instead_of_exiting() -> None:
    events: list[str] = []
    mouse = _HomeMouse([{}, {}, {}, {26: True}])

    workflow = _home_collection_workflow(mouse, events, idle_camera_timeouts=1)
    assert workflow.run() == 0
    assert events.count("sync:0.25") == 2
    assert events[-1] == "finalize"


def test_collection_rotation_lock_finalizes_during_homing() -> None:
    events: list[str] = []
    mouse = _HomeMouse([{}, {26: True}], at_home=False)

    workflow = _home_collection_workflow(mouse, events)
    assert workflow.run() == 0
    assert not any(event.startswith("record:") for event in events)
    assert events.count("clear") == 1
    assert events[-1] == "finalize"
    assert mouse.home_requests == 1


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


def test_collection_dashboard_recovers_when_home_feedback_settles(monkeypatch) -> None:
    import slai_mi.ui.collection_dashboard as dashboard_module

    phases: list[str] = []

    class Mouse(_HomeMouse):
        def __init__(self) -> None:
            super().__init__([{}, {}, {}, {}])
            self.home_states = iter((True, False, True))

        def state(self):
            if not self.states:
                raise RuntimeError("stop after home recovery")
            return super().state()

        def home_status(self) -> dict[str, object]:
            return {"at_home": next(self.home_states), "detail": "settling"}

    class Provider:
        def set_phase(self, phase, _label, **_fields):
            phases.append(phase)

        def event(self, _message, **_fields):
            return

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

    monkeypatch.setattr(dashboard_module, "CollectionDashboard", Dashboard)
    monkeypatch.setattr(
        dashboard_module, "DashboardSynchronizer", lambda synchronizer, _provider: synchronizer
    )
    workflow = _home_collection_workflow(Mouse(), [])
    workflow.dashboard_enabled = True

    with pytest.raises(RuntimeError, match="stop after home recovery"):
        workflow.run()

    blocked = phases.index("blocked")
    assert "ready" in phases[blocked + 1 :]
