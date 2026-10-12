"""Dependency-injected orchestration for physical teleoperation and collection."""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, Protocol, Self

from slai_mi.collection.operator_control import EpisodeAction, SpaceMouseEpisodeControls
from slai_mi.devices.spacemouse.buttons import Button
from slai_mi.devices.spacemouse.gestures import RepeatedChordGesture
from slai_mi.input_schema import enabled_cameras, load_input_schema

HOME_TIMEOUT_S = 30.0
DOWNWARD_START_THRESHOLD_M_S = 1e-4
SPACEMOUSE_DESCENT_DEADZONE = 0.12


def _effective_downward_command(mouse: Any, motion: Sequence[float]) -> bool:
    """True only when the controlled UR5 is receiving an effective downward command."""
    commanded_twist = getattr(mouse, "commanded_twist", None)
    if callable(commanded_twist):
        twist = tuple(float(value) for value in commanded_twist())
        return len(twist) == 6 and twist[2] < -DOWNWARD_START_THRESHOLD_M_S
    # Dependency-injected workflow tests may use a plain SpaceMouse. Its XYZ
    # mapping is direct, so apply the production deadzone before accepting Z.
    values = tuple(float(value) for value in motion)
    return len(values) == 6 and values[2] < -SPACEMOUSE_DESCENT_DEADZONE


class StoppableRuntime(Protocol):
    def run(self, stop_event: threading.Event) -> None: ...


def validate_real_hardware_config(config: Mapping[str, Any], *, required: Sequence[str]) -> None:
    """Validate the identity gate before any dependency may open hardware."""
    if config.get("configured") is not True:
        raise ValueError("hardware configuration is not confirmed (configured must be true)")
    for name in required:
        section = config.get(name)
        if not isinstance(section, Mapping) or section.get("enabled") is not True:
            raise ValueError(f"required device is not enabled: {name}")
    ur5 = config.get("ur5", {})
    if "ur5" in required and not str(ur5.get("host") or "").strip():
        raise ValueError("ur5.host must be configured")
    cameras = config.get("cameras", {})
    if "cameras" in required:
        devices = cameras.get("devices")
        schema = load_input_schema(config.get("input_schema"))
        expected_roles = {str(camera["role"]) for camera in enabled_cameras(schema)}
        if not isinstance(devices, list) or len(devices) != len(expected_roles):
            raise ValueError(
                f"cameras.devices must define the {len(expected_roles)} enabled schema roles"
            )
        roles = {item.get("role") for item in devices if isinstance(item, Mapping)}
        if roles != expected_roles:
            raise ValueError(f"camera roles must match input schema: {sorted(expected_roles)}")
        if any(not str(item.get("serial") or "").strip() for item in devices):
            raise ValueError("every enabled camera requires a serial")


@dataclass(frozen=True)
class TeleopDependencies:
    ur5_factory: Callable[[Mapping[str, Any], Any, threading.Event], StoppableRuntime]
    wuji_factory: Callable[[Mapping[str, Any], Any, threading.Event], StoppableRuntime]
    spacemouse_factory: Callable[[Mapping[str, Any]], Any]
    preflight: Callable[[Mapping[str, Any]], None] = lambda _config: None
    runtime_factories: (
        Mapping[str, Callable[[Mapping[str, Any], Any, threading.Event], StoppableRuntime]] | None
    ) = None
    required_devices: tuple[str, ...] = ("ur5", "wujihand", "spacemouse")


@dataclass(frozen=True)
class CollectionDependencies:
    ur5_factory: Callable[[Mapping[str, Any]], Any]
    wuji_factory: Callable[[Mapping[str, Any]], Any]
    spacemouse_factory: Callable[[Mapping[str, Any]], Any]
    cameras_factory: Callable[[Mapping[str, Any]], Any]
    dataset_factory: Callable[[Mapping[str, Any], Mapping[str, Any]], Any]
    synchronizer_factory: Callable[[Mapping[str, Any], Mapping[str, Any]], Any]
    recorder_factory: Callable[[Any, Any, str], Any]
    preflight: Callable[[Mapping[str, Any]], None] = lambda _config: None
    sleep: Callable[[float], None] = time.sleep
    resource_factories: Mapping[str, Callable[[Mapping[str, Any]], Any]] | None = None
    required_devices: tuple[str, ...] = ("ur5", "wujihand", "spacemouse", "cameras")


@dataclass
class _Worker:
    name: str
    target: Callable[[threading.Event], None]
    stop_event: threading.Event
    failure: list[BaseException]
    thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        def guarded() -> None:
            try:
                self.target(self.stop_event)
            except BaseException as exc:  # noqa: BLE001 - transfer worker failures
                self.failure.append(exc)
            finally:
                self.stop_event.set()

        self.thread = threading.Thread(target=guarded, name=self.name)


class _SignalStop:
    def __init__(self, stop_event: threading.Event) -> None:
        self.stop_event = stop_event
        self.previous: dict[int, Any] = {}

    def __enter__(self) -> Self:
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                self.previous[number] = signal.getsignal(number)
                signal.signal(number, self._stop)
        return self

    def _stop(self, _number: int, _frame: FrameType | None) -> None:
        self.stop_event.set()

    def __exit__(self, *_args: object) -> None:
        for number, handler in self.previous.items():
            signal.signal(number, handler)


class RealTeleopWorkflow:
    """Supervise every runtime in one selected hardware group."""

    def __init__(self, hardware: Mapping[str, Any], dependencies: TeleopDependencies) -> None:
        self.hardware = hardware
        self.dependencies = dependencies

    def run(self) -> None:
        validate_real_hardware_config(self.hardware, required=self.dependencies.required_devices)
        self.dependencies.preflight(self.hardware)
        stop_event = threading.Event()
        failures: list[BaseException] = []
        with ExitStack() as resources, _SignalStop(stop_event):
            mouse = resources.enter_context(self.dependencies.spacemouse_factory(self.hardware))
            factories = self.dependencies.runtime_factories or {
                "ur5-teleop": self.dependencies.ur5_factory,
                "wuji-teleop": self.dependencies.wuji_factory,
            }
            workers = []
            for name, factory in factories.items():
                runtime = factory(self.hardware, mouse, stop_event)
                workers.append(_Worker(name, runtime.run, stop_event, failures))
            for worker in workers:
                worker.thread.start()
            while not stop_event.wait(0.05) and any(w.thread.is_alive() for w in workers):
                pass
            stop_event.set()
            for worker in workers:
                worker.thread.join(timeout=5.0)
            alive = [worker.name for worker in workers if worker.thread.is_alive()]
            if alive:
                raise RuntimeError(f"hardware workers did not stop: {', '.join(alive)}")
            if failures:
                raise RuntimeError(f"hardware worker failed: {failures[0]}") from failures[0]


class RealCollectionWorkflow:
    """Own real sensors, episode recording, dataset commit/discard, and cleanup."""

    def __init__(
        self,
        hardware: Mapping[str, Any],
        dataset_config: Mapping[str, Any],
        task: Mapping[str, Any],
        dependencies: CollectionDependencies,
        *,
        episode_limit: int | None,
        dashboard_enabled: bool = False,
        dashboard_host: str = "0.0.0.0",
        dashboard_port: int = 8765,
        dashboard_open_browser: bool = True,
    ) -> None:
        if episode_limit is not None and episode_limit < 1:
            raise ValueError("episode_limit must be at least one")
        self.hardware = hardware
        self.dataset_config = dataset_config
        self.task = task
        self.dependencies = dependencies
        self.episode_limit = episode_limit
        self.dashboard_enabled = dashboard_enabled
        self.dashboard_host = dashboard_host
        self.dashboard_port = dashboard_port
        self.dashboard_open_browser = dashboard_open_browser

    def run(self) -> int:
        validate_real_hardware_config(self.hardware, required=self.dependencies.required_devices)
        instruction = str(self.task.get("task", {}).get("instruction") or "").strip()
        if not instruction:
            raise ValueError("task.task.instruction must be configured")
        self.dependencies.preflight(self.hardware)
        stop_event = threading.Event()
        with ExitStack() as stack, _SignalStop(stop_event):
            dashboard = None
            if self.dashboard_enabled:
                from slai_mi.ui.collection_dashboard import CollectionDashboard

                dashboard = stack.enter_context(
                    CollectionDashboard(
                        dict(self.hardware),
                        instruction,
                        task_id=str(self.task.get("task", {}).get("id") or "") or None,
                        host=self.dashboard_host,
                        port=self.dashboard_port,
                        open_browser=self.dashboard_open_browser,
                    )
                )
                print(f"Collection dashboard: {dashboard.url}")
                dashboard.provider.set_phase(
                    "preflight", "正在连接真实设备", can_record=False, recording=False
                )
            if self.dependencies.resource_factories is None:
                sources = {
                    "ur5": stack.enter_context(self.dependencies.ur5_factory(self.hardware)),
                    "wuji": stack.enter_context(self.dependencies.wuji_factory(self.hardware)),
                    "spacemouse": stack.enter_context(
                        self.dependencies.spacemouse_factory(self.hardware)
                    ),
                    "cameras": stack.enter_context(
                        self.dependencies.cameras_factory(self.hardware)
                    ),
                }
            else:
                sources = {
                    name: stack.enter_context(factory(self.hardware))
                    for name, factory in self.dependencies.resource_factories.items()
                }
            mouse = sources["spacemouse"]
            dataset = self.dependencies.dataset_factory(self.dataset_config, self.task)
            stack.callback(dataset.finalize)
            synchronizer = self.dependencies.synchronizer_factory(
                sources,
                self.dataset_config,
            )
            if dashboard is not None:
                from slai_mi.ui.collection_dashboard import DashboardSynchronizer

                dashboard.provider.set_dataset_path(getattr(dataset, "_root", "pending"))
                synchronizer = DashboardSynchronizer(synchronizer, dashboard.provider)
            recorder = self.dependencies.recorder_factory(dataset, synchronizer, instruction)
            if dashboard is not None and hasattr(recorder, "on_frame"):
                recorder.on_frame = dashboard.provider.record_frame
            controls = SpaceMouseEpisodeControls()
            cancel_gesture = RepeatedChordGesture(
                (int(Button.MENU), int(Button.FIT)),
                repeats=2,
                timeout_s=5.0,
                require_initial_release=True,
            )
            saved = 0
            attempts = 0
            active_stop: threading.Event | None = None
            worker: _Worker | None = None
            failures: list[BaseException] = []
            supports_home = all(
                callable(getattr(mouse, name, None))
                for name in ("request_home", "clear_home", "home_status")
            )
            homing = False
            pending_descent_start = False
            pending_save = False
            pending_discard = False
            home_started_at: float | None = None
            previous_buttons: dict[int, bool] = {}
            motion_receipt_reported = False

            def limit_reached() -> bool:
                return self.episode_limit is not None and saved >= self.episode_limit

            def begin_episode(*, automatic: bool) -> None:
                nonlocal active_stop, worker, attempts
                if automatic:
                    controls.recording = True
                attempts += 1
                print(f"Episode started: index={saved + 1} attempt={attempts}", flush=True)
                if dashboard is not None:
                    dashboard.provider.start_episode(index=saved + 1, attempt=attempts)
                    if automatic:
                        dashboard.provider.event(
                            f"已自动开始 Episode {saved + 1} 录制",
                            level="success",
                            code="episode_auto_start",
                        )
                active_stop = threading.Event()
                worker = _Worker("episode-recorder", recorder.record, active_stop, failures)
                worker.thread.start()

            def start_homing(
                message: str, *, auto_start: bool, preserve_gripper: bool = False
            ) -> None:
                nonlocal homing, pending_descent_start, home_started_at
                if not supports_home:
                    return
                controls.abort()
                request_preserving_home = getattr(
                    mouse, "request_home_preserve_gripper", None
                )
                if preserve_gripper and callable(request_preserving_home):
                    request_preserving_home()
                else:
                    mouse.request_home()
                print(f"HOME requested: {message}", flush=True)
                homing = True
                # Menu arms an Episode, but recording must not begin until HOME
                # has completed and the first effective downward UR5 command occurs.
                pending_descent_start = auto_start
                home_started_at = time.monotonic()
                if dashboard is not None:
                    dashboard.provider.set_phase(
                        "homing",
                        "归零中（等待保存，已停止录制）" if pending_save else "归零中",
                        can_record=False,
                        recording=pending_save,
                    )
                    dashboard.provider.event(message, code="homing")

            def stop_episode_recorder() -> int:
                nonlocal active_stop, worker
                if active_stop is None or worker is None:
                    raise RuntimeError("episode recorder is not active")
                active_stop.set()
                worker.thread.join(timeout=5.0)
                if worker.thread.is_alive():
                    raise RuntimeError("episode recorder did not stop")
                if failures:
                    raise RuntimeError(f"episode recorder failed: {failures[0]}") from failures[0]
                frame_count = int(getattr(recorder, "frame_count", 1))
                active_stop = None
                worker = None
                return frame_count

            def emergency_save_active_episode(reason: BaseException) -> None:
                """Persist already-captured frames before fail-closed cleanup."""
                nonlocal active_stop, worker, saved
                if active_stop is None or worker is None:
                    return
                active_stop.set()
                worker.thread.join(timeout=5.0)
                if worker.thread.is_alive():
                    return
                frame_count = int(getattr(recorder, "frame_count", 0))
                active_stop = None
                worker = None
                if frame_count <= 0:
                    return
                dataset.save_episode()
                saved += 1
                if dashboard is not None:
                    dashboard.provider.finish_episode(
                        "emergency", index=saved
                    )
                    dashboard.provider.event(
                        f"已紧急保存中断 Episode {saved}（{type(reason).__name__}: {reason}）",
                        level="error",
                        code="emergency_save",
                    )

            def finalize_from_lock() -> None:
                if active_stop is not None and worker is not None:
                    stop_episode_recorder()
                if dashboard is not None:
                    dashboard.provider.set_phase(
                        "finalizing",
                        "正在结束",
                        can_record=False,
                        recording=False,
                    )
                    dashboard.provider.event(
                        "LOCK 已按下：丢弃未提交的当前段，保留并完成已保存 Episode 写入",
                        level="success",
                        code="finalize",
                    )

            idle_camera_timeouts = 0

            def refresh_idle_inputs() -> None:
                """Tolerate transient camera phase misses outside an Episode."""
                nonlocal idle_camera_timeouts
                try:
                    synchronizer.read(timeout_s=0.25)
                    idle_camera_timeouts = 0
                except TimeoutError as exc:
                    idle_camera_timeouts += 1
                    if idle_camera_timeouts == 1:
                        print(
                            "Camera synchronization transient during idle/HOME; retrying",
                            flush=True,
                        )
                        if dashboard is not None:
                            dashboard.provider.event(
                                "待机/回零阶段相机短暂不同步，正在自动重试",
                                level="warning",
                                code="camera_sync_retry",
                            )
                    if idle_camera_timeouts >= 20:
                        raise RuntimeError(
                            "camera synchronization unavailable for about 5 seconds"
                        ) from exc

            try:
                if dashboard is not None or supports_home:
                    # Arm through the normal supervised input boundary before the
                    # synchronizer starts publishing live telemetry.
                    initial_motion, initial_buttons = mouse.state()
                    controls.synchronize(initial_buttons)
                    if dashboard is not None:
                        dashboard.provider.observe_spacemouse(initial_motion, initial_buttons)
                        diagnostics = getattr(mouse, "diagnostics", dict)()
                        dashboard.provider.event(
                            "采集控制循环已收到 SpaceMouse state "
                            f"(packets={diagnostics.get('packet_count', '?')}, "
                            f"samples={diagnostics.get('sample_count', '?')})",
                            level="success",
                            code="spacemouse_state_received",
                        )
                    diagnostics = getattr(mouse, "diagnostics", dict)()
                    print(
                        "SpaceMouse state reached collection loop: "
                        f"packets={diagnostics.get('packet_count', '?')} "
                        f"samples={diagnostics.get('sample_count', '?')} "
                        f"events={diagnostics.get('event_count', '?')}",
                        flush=True,
                    )
                    synchronizer.read(timeout_s=5.0)
                if supports_home:
                    start_homing("设备同步完成，正在自动归零", auto_start=False)
                elif dashboard is not None:
                    dashboard.provider.set_phase(
                        "ready", "可以开始录制", can_record=True, recording=False
                    )
                    dashboard.provider.event(
                        "真实输入已接入；Menu 开始、Fit 继续录制回零过程并在到位后保存、Esc 丢弃",
                        level="success",
                        code="ready",
                    )
                while not stop_event.is_set():
                    if failures:
                        raise RuntimeError(f"episode recorder failed: {failures[0]}") from failures[
                            0
                        ]
                    if limit_reached() and not homing and active_stop is None:
                        break
                    motion, buttons = mouse.state()
                    consume_external_home = getattr(
                        mouse, "consume_external_home_request", None
                    )
                    if callable(consume_external_home) and consume_external_home():
                        interrupted_episode = active_stop is not None or pending_save
                        if active_stop is not None:
                            stop_episode_recorder()
                        if interrupted_episode:
                            dataset.clear_episode_buffer()
                        pending_save = False
                        pending_discard = interrupted_episode
                        start_homing(
                            "底层 Home 命令：已抢占当前控制并返回 TASK2 零位",
                            auto_start=False,
                        )
                        print(
                            "Low-level Home accepted; UR5, wrist, and gripper are homing",
                            flush=True,
                        )
                        if dashboard is not None:
                            dashboard.provider.event(
                                "底层 Home 已执行；未完成的当前 Episode 已丢弃",
                                level="warning" if interrupted_episode else "success",
                                code="external_home",
                            )
                    if not motion_receipt_reported and any(
                        abs(float(value)) > 0.0 for value in motion
                    ):
                        print(
                            "SpaceMouse motion reached collection control loop: "
                            f"motion={[round(float(value), 4) for value in motion]}",
                            flush=True,
                        )
                        motion_receipt_reported = True
                    for code, name in (
                        (int(Button.MENU), "Menu"),
                        (int(Button.FIT), "Fit"),
                        (int(Button.CTRL), "Ctrl"),
                        (int(Button.THREE), "3"),
                        (int(Button.FOUR), "4"),
                    ):
                        if buttons.get(code, False) and not previous_buttons.get(code, False):
                            print(f"SpaceMouse {name} detected", flush=True)
                    previous_buttons = buttons.copy()
                    if dashboard is not None:
                        dashboard.provider.observe_spacemouse(motion, buttons)
                    if bool(any(abs(float(value)) > 0.0 for value in motion)):
                        cancel_gesture.reset()
                    elif cancel_gesture.update(buttons, time.monotonic()):
                        # Menu+Fit double chord is a lifecycle toggle:
                        # first use starts an episode; while recording it exits.
                        if active_stop is None and not homing and not limit_reached():
                            home = mouse.home_status() if supports_home else {
                                "at_home": True,
                                "detail": "",
                            }
                            if supports_home:
                                # Every operator-started episode gets a fresh
                                # coordinated HOME, including Episode 2+.
                                # Being already at HOME is not a reason to skip
                                # the request: the robot must re-enter the known
                                # task start state before recording.
                                start_homing(
                                    f"Menu+Fit 双击：正在归零，准备开始 Episode {saved + 1}",
                                    auto_start=True,
                                )
                            else:
                                begin_episode(automatic=True)
                            if dashboard is not None:
                                dashboard.provider.event(
                                    f"Menu+Fit 双击：开始 Episode {saved + 1} 采集",
                                    level="success",
                                    code="physical_collection_start",
                                )
                        elif active_stop is not None:
                            finalize_from_lock()
                            if dashboard is not None:
                                dashboard.provider.event(
                                    "Menu+Fit 双击：退出数采并恢复待机网页",
                                    level="success",
                                    code="physical_collection_exit",
                                )
                            break
                    home = mouse.home_status() if supports_home else {"at_home": True, "detail": ""}
                    if homing:
                        homing_action = controls.update(buttons)
                        controls.abort()
                        if bool(home.get("at_home")):
                            mouse.clear_home()
                            frame_count = stop_episode_recorder() if pending_save else None
                            homing = False
                            home_started_at = None
                            if dashboard is not None:
                                dashboard.provider.event(
                                    "已自动回到任务零位",
                                    level="success",
                                    code="home_complete",
                                )
                            print("HOME complete; collection ready", flush=True)
                            if pending_save:
                                if dashboard is not None:
                                    dashboard.provider.set_phase(
                                        "saving",
                                        "归零完成，正在保存",
                                        can_record=False,
                                        recording=False,
                                    )
                                pending_save = False
                                if frame_count == 0:
                                    dataset.clear_episode_buffer()
                                    pending_discard = True
                                    if dashboard is not None:
                                        dashboard.provider.event(
                                            f"Episode {saved + 1} 已丢弃：没有有效帧",
                                            level="warning",
                                            code="episode_empty",
                                        )
                                else:
                                    dataset.save_episode()
                                    saved += 1
                                    if dashboard is not None:
                                        dashboard.provider.finish_episode("save", index=saved)
                            if pending_discard:
                                pending_discard = False
                                if dashboard is not None:
                                    dashboard.provider.event(
                                        "当前 Episode 已丢弃，已归零，等待 Menu；不会自动录制",
                                        level="warning",
                                        code="episode_discard_wait",
                                    )
                            if homing_action is EpisodeAction.FINALIZE:
                                finalize_from_lock()
                                break
                            if pending_descent_start and not limit_reached():
                                if dashboard is not None:
                                    dashboard.provider.set_phase(
                                        "armed",
                                        "归零完成，等待首次下降后开始录制",
                                        can_record=True,
                                        recording=False,
                                    )
                                    dashboard.provider.event(
                                        "Menu 已就绪；首次有效下降命令将启动 Episode",
                                        level="success",
                                        code="episode_armed_for_descent",
                                    )
                            elif dashboard is not None:
                                pending_descent_start = False
                                dashboard.provider.set_phase(
                                    "ready",
                                    "归零完成，按 Menu 开始",
                                    can_record=not limit_reached(),
                                    recording=False,
                                )
                        elif homing_action is EpisodeAction.FINALIZE:
                            finalize_from_lock()
                            break
                        elif (
                            home_started_at is not None
                            and time.monotonic() - home_started_at >= HOME_TIMEOUT_S
                        ):
                            mouse.clear_home()
                            homing = False
                            pending_descent_start = False
                            home_started_at = None
                            if pending_save:
                                dataset.clear_episode_buffer()
                                stop_episode_recorder()
                                pending_save = False
                                if dashboard is not None:
                                    dashboard.provider.event(
                                        f"Episode {saved + 1} 未保存：自动归零超时",
                                        level="warning",
                                        code="episode_discard",
                                    )
                            pending_discard = False
                            if dashboard is not None:
                                dashboard.provider.set_phase(
                                    "blocked",
                                    "自动归零超时，禁止录制",
                                    can_record=False,
                                    recording=False,
                                )
                                dashboard.provider.event(
                                    f"自动归零超时，禁止录制：{home.get('detail', '')}",
                                    level="error",
                                    code="home_timeout",
                                )
                        if active_stop is None:
                            refresh_idle_inputs()
                        self.dependencies.sleep(0.005)
                        continue

                    if (
                        pending_descent_start
                        and active_stop is None
                        and not limit_reached()
                        and _effective_downward_command(mouse, motion)
                    ):
                        pending_descent_start = False
                        begin_episode(automatic=True)
                        if dashboard is not None:
                            dashboard.provider.event(
                                "检测到首次有效下降；Episode 从当前帧开始录制",
                                level="success",
                                code="episode_started_on_descent",
                            )
                    action = controls.update(buttons)
                    if (
                        supports_home
                        and active_stop is None
                        and not bool(home.get("at_home"))
                        and dashboard is not None
                    ):
                        dashboard.provider.set_phase(
                            "blocked",
                            "尚未回到零位，禁止录制",
                            can_record=False,
                            recording=False,
                        )
                    elif supports_home and active_stop is None and dashboard is not None:
                        # Feedback can briefly leave the home tolerance while the
                        # hand settles. Restore the idle controls once it is stable
                        # again instead of leaving the dashboard blocked forever.
                        dashboard.provider.set_phase(
                            "ready",
                            "归零完成，按 Menu 开始",
                            can_record=not limit_reached(),
                            recording=False,
                        )
                    if action is EpisodeAction.FINALIZE:
                        finalize_from_lock()
                        break
                    if action is EpisodeAction.START:
                        if supports_home:
                            # Always home before a new episode. Do not use the
                            # current at_home flag as a shortcut for Episode 2+.
                            start_homing(
                                f"开始 Episode {saved + 1} 前正在自动归零",
                                auto_start=True,
                            )
                        else:
                            begin_episode(automatic=False)
                    elif action is EpisodeAction.HOME:
                        if supports_home:
                            start_homing("Fit：正在返回任务零位", auto_start=False)
                    elif action in {EpisodeAction.SAVE, EpisodeAction.DISCARD}:
                        if active_stop is None or worker is None:
                            raise RuntimeError("episode control changed without an active recorder")
                        if action is EpisodeAction.SAVE:
                            if supports_home:
                                # Continue recording through the coordinated HOME.
                                # The Episode is committed only after HOME feedback
                                # confirms that every device reached the task start.
                                pending_save = True
                                start_homing(
                                    f"Episode {saved + 1} 已结束操作，正在保持夹爪状态并回零后保存",
                                    auto_start=False,
                                    preserve_gripper=True,
                                )
                            else:
                                frame_count = stop_episode_recorder()
                                if frame_count == 0:
                                    dataset.clear_episode_buffer()
                                    if dashboard is not None:
                                        dashboard.provider.finish_episode(
                                            "discard", index=saved + 1
                                        )
                                else:
                                    if dashboard is not None:
                                        dashboard.provider.set_phase(
                                            "saving",
                                            "正在保存",
                                            can_record=False,
                                            recording=False,
                                        )
                                    dataset.save_episode()
                                    saved += 1
                                    if dashboard is not None:
                                        dashboard.provider.finish_episode("save", index=saved)
                        else:
                            stop_episode_recorder()
                            dataset.clear_episode_buffer()
                            pending_discard = True
                            if dashboard is not None:
                                dashboard.provider.event(
                                    f"Episode {saved + 1} 已丢弃，正在自动归零",
                                    level="warning",
                                    code="episode_discard",
                                )
                            start_homing(
                                f"Episode {saved + 1} 已丢弃，正在自动归零；归零后等待 Menu",
                                auto_start=False,
                            )
                            if not supports_home:
                                pending_discard = False
                                if dashboard is not None:
                                    dashboard.provider.finish_episode("discard", index=saved + 1)
                    elif active_stop is None:
                        refresh_idle_inputs()
                    self.dependencies.sleep(0.005)
                if dashboard is not None:
                    dashboard.provider.set_phase(
                        "stopped", "采集已完成", can_record=False, recording=False
                    )
                    dashboard.provider.event(
                        f"已完成 {saved} 个 Episode", level="success", code="complete"
                    )
            except BaseException as exc:
                # Safety failures must not discard frames already captured in
                # the current Menu-to-Fit window. Save before the resource
                # stack disables hardware and tears down the dataset.
                try:
                    emergency_save_active_episode(exc)
                except BaseException as emergency_exc:  # noqa: BLE001 - preserve primary failure
                    exc.add_note(f"emergency save failed: {emergency_exc}")
                if dashboard is not None:
                    dashboard.provider.set_phase(
                        "error", "采集流程错误", can_record=False, recording=False
                    )
                    dashboard.provider.event(
                        f"{type(exc).__name__}: {exc}", level="error", code="failure"
                    )
                raise
            finally:
                try:
                    if active_stop is not None:
                        active_stop.set()
                    if worker is not None:
                        worker.thread.join(timeout=5.0)
                        if worker.thread.is_alive():
                            raise RuntimeError("episode recorder did not stop")
                    if failures:
                        raise RuntimeError(f"episode recorder failed: {failures[0]}") from failures[
                            0
                        ]
                finally:
                    dataset.clear_episode_buffer()
            return saved
