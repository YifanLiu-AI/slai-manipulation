"""Non-interactive ESP32 master to OpenRB wrist adapter.

The protocol implementation stays in the vendor repository.  This module only
owns lifecycle, automatic master-zero establishment, and radians at the SLAI
boundary.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_vendor_root() -> Path:
    configured = os.environ.get("SLAI_WRIST_VENDOR_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        PROJECT_ROOT / "third_party" / "02_Python_Client_CLI",
        PROJECT_ROOT.parent
        / "slai-manipulation"
        / "third_party"
        / "02_Python_Client_CLI",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "openrb_bridge").is_dir():
            return candidate.resolve()
    return PROJECT_ROOT / "third_party" / "02_Python_Client_CLI"


VENDOR_ROOT = _resolve_vendor_root()
OPEN_LOOP_ROOT = VENDOR_ROOT / "open_loop_record"
HOME_ACCEPTANCE_TOLERANCE_DEG = 0.60
SPACEMOUSE_WRIST_COMMAND_LEASE_S = 0.10
SPACEMOUSE_WRIST_ACCEL_DEG_S2 = 300.0
SPACEMOUSE_WRIST_DECEL_DEG_S2 = 500.0


def _slew_velocity(current: np.ndarray, desired: np.ndarray, dt: float) -> np.ndarray:
    interval = max(0.0, float(dt))
    reversing = (current != 0.0) & (np.sign(current) != np.sign(desired))
    effective_desired = np.where(reversing, 0.0, desired)
    braking = np.abs(effective_desired) < np.abs(current)
    rate = np.where(
        braking, SPACEMOUSE_WRIST_DECEL_DEG_S2, SPACEMOUSE_WRIST_ACCEL_DEG_S2
    )
    max_delta = rate * interval
    return current + np.clip(effective_desired - current, -max_delta, max_delta)


@dataclass(frozen=True)
class WristTeleopState:
    actual_q: np.ndarray
    target_q: np.ndarray
    host_timestamp_s: float
    sequence: int
    motor_position_raw: np.ndarray
    motor_goal_raw: np.ndarray
    motor_rate_raw_s: np.ndarray


def _vendor_imports() -> tuple[Any, ...]:
    for path in (str(VENDOR_ROOT), str(OPEN_LOOP_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from example_teleop_control import (  # type: ignore[import-not-found]
            AngleUnwrapper,
            TeleopSerial,
            field_float_deg,
            parse_line,
        )
        from openrb_bridge.output_v2 import (  # type: ignore[import-not-found]
            OneEuroFilter,
            OutputTargetShaper,
            WristOutputV2Controller,
            load_output_v2_config,
            map_master_relative_deg,
        )
        from openrb_bridge.pc_client.openrb_client import (  # type: ignore[import-not-found]
            OpenRBClient,
        )
        from openrb_bridge.serial_ports import (  # type: ignore[import-not-found]
            resolve_openrb_port,
            resolve_teleop_port,
        )
    finally:
        sys.dont_write_bytecode = previous
    return (
        AngleUnwrapper,
        TeleopSerial,
        field_float_deg,
        parse_line,
        OneEuroFilter,
        OutputTargetShaper,
        WristOutputV2Controller,
        load_output_v2_config,
        map_master_relative_deg,
        OpenRBClient,
        resolve_openrb_port,
        resolve_teleop_port,
    )


def _home_acceptance_tolerance_deg(config: Any) -> float:
    nominal = float(config.settling.tolerance_deg)
    if nominal <= 0.0 or HOME_ACCEPTANCE_TOLERANCE_DEG < nominal:
        raise ValueError("wrist HOME acceptance tolerance must be >= nominal tolerance")
    return HOME_ACCEPTANCE_TOLERANCE_DEG


def _accept_stopped_home_residual(state: Any, config: Any) -> bool:
    tolerance = _home_acceptance_tolerance_deg(config)
    return bool(
        bool(state.active)
        and not bool(state.fault)
        and bool(state.zero_valid)
        and float(state.sensor_age_s) <= float(config.stream.max_state_age_s)
        and abs(float(state.target_fe_deg)) <= 0.01
        and abs(float(state.target_ru_deg)) <= 0.01
        and int(state.j1_rate_raw_s) == 0
        and int(state.j2_rate_raw_s) == 0
        and abs(float(state.fe_deg)) <= tolerance
        and abs(float(state.ru_deg)) <= tolerance
    )


def _wait_home_settled(controller: Any, state_callback: Any | None = None) -> Any:
    if state_callback is None:
        try:
            return controller.wait_settled(0.0, 0.0)
        except TimeoutError:
            state = controller.read_state()
            if _accept_stopped_home_residual(state, controller.config):
                return state
            raise
    setting = controller.config.settling
    deadline = time.monotonic() + float(setting.timeout_s)
    consecutive = 0
    last = None
    while time.monotonic() < deadline:
        last = controller.read_state()
        state_callback(last)
        if (
            abs(float(last.fe_deg)) <= float(setting.tolerance_deg)
            and abs(float(last.ru_deg)) <= float(setting.tolerance_deg)
        ):
            consecutive += 1
            if consecutive >= int(setting.consecutive_samples):
                return last
        else:
            consecutive = 0
        time.sleep(1.0 / float(controller.config.stream.state_hz))
    if last is not None and _accept_stopped_home_residual(last, controller.config):
        return last
    raise TimeoutError(f"output did not settle at FE=0.0, RU=0.0; last={last}")


COLLECTION_HOME_PROFILE_VELOCITY_RAW = 300
COLLECTION_HOME_PROFILE_ACCELERATION_RAW = 250
# Kept as a compatibility constant for collection callers. Policy targets do
# not use this manual-jog speed; the OpenRB tracking profile owns that motion.
SPACEMOUSE_SOFT_HOME_SPEED_DEG_S = 14.0
SPACEMOUSE_SOFT_HOME_MAX_MOTOR_RATE_RAW_S = 80


def prepare_dynamic_home(
    controller: Any, *, state_callback: Any | None = None
) -> tuple[Any, tuple[float, float] | None]:
    """HOME gently and define the measured post-HOME encoders as this run's zero."""
    if not all(
        callable(getattr(controller, name, None))
        for name in ("inspect", "_wait_motion_idle", "wait_settled")
    ) or not hasattr(controller, "client"):
        state = controller.prepare(auto_home_zero=True, force_home_zero=True)
        return state, None
    response = controller.inspect()
    if response.fields.get("active") == "1" or response.fields.get("vmode") == "1":
        stopped = controller.client.set_output_closed_loop_enable(False)
        if not stopped.ok:
            raise RuntimeError(stopped.raw_lines[-1])
    config = controller.config.controller
    configured = controller.client.set_output_direct_config(
        config.fe_kp_milli,
        config.fe_ki_milli,
        config.ru_kp_milli,
        config.ru_ki_milli,
        config.max_motor_rate_raw_s,
        config.deadband_cdeg,
        config.fe_velocity_ff_raw_per_deg,
        config.ru_velocity_ff_raw_per_deg,
        config.fe_moving_feedback_limit_raw,
        config.ru_moving_feedback_limit_raw,
    )
    if not configured.ok:
        raise RuntimeError(configured.raw_lines[-1])
    # HOME_ALL uses the Dynamixel profile already installed on each joint.
    # Normal collection previously inherited the responsive 800/1000 policy
    # profile, making the SpaceMouse R wrist-home unnecessarily aggressive.
    # Install a collection-only gentle profile before HOME; the regular policy
    # profile is restored below after the mechanical zero has been established.
    home_profile = controller.client.set_arm_profile(
        COLLECTION_HOME_PROFILE_VELOCITY_RAW,
        COLLECTION_HOME_PROFILE_ACCELERATION_RAW,
    )
    if not home_profile.ok:
        raise RuntimeError(home_profile.raw_lines[-1])
    home = controller.client.home_all()
    if not home.ok:
        raise RuntimeError(home.raw_lines[-1])
    if state_callback is None:
        controller._wait_motion_idle()
    else:
        deadline = time.monotonic() + float(controller.config.settling.timeout_s)
        while time.monotonic() < deadline:
            status = controller.client.get_motion_status(retries=4)
            try:
                state_callback(controller.read_state())
            except RuntimeError as exc:
                if "output zero became invalid" not in str(exc):
                    raise
            if status.fields.get("active") == "0":
                if status.fields.get("status") == "failed":
                    raise RuntimeError(status.raw_lines[-1])
                break
            time.sleep(1.0 / float(controller.config.stream.state_hz))
        else:
            raise TimeoutError("HOME_ALL timed out")
    fields = controller.client.read_wrist_state().fields
    enc0 = int(fields["enc0_deg"]) / 100.0
    enc1 = int(fields["enc1_deg"]) / 100.0
    zero = controller.client.set_output_zero_absolute_cdeg(
        round(enc0 * 100.0), round(enc1 * 100.0)
    )
    if not zero.ok:
        raise RuntimeError(zero.raw_lines[-1])
    profile = controller.client.set_arm_profile(
        config.motor_profile_velocity_raw,
        config.motor_profile_acceleration_raw,
    )
    if not profile.ok:
        raise RuntimeError(profile.raw_lines[-1])
    enabled = controller.client.set_output_closed_loop_enable(True)
    if not enabled.ok:
        raise RuntimeError(enabled.raw_lines[-1])
    target = controller.client.set_output_target_cdeg(0, 0)
    if not target.ok:
        raise RuntimeError(target.raw_lines[-1])
    return _wait_home_settled(controller, state_callback), (enc0, enc1)


class WristMasterSlaveController:
    """Continuously follow the wrist master without requiring a START button."""

    def __init__(
        self,
        config: str | Path,
        *,
        teleop_port: str = "auto",
        openrb_port: str = "auto",
        baud: int = 115200,
    ) -> None:
        self.config_path = Path(config).expanduser().resolve()
        self.teleop_port = teleop_port
        self.openrb_port = openrb_port
        self.baud = int(baud)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._home_requested = threading.Event()
        self._home_running = threading.Event()
        self._soft_home_requested = threading.Event()
        self._soft_home_running = threading.Event()
        self._park_requested = threading.Event()
        self._park_running = threading.Event()
        self._resume_requested = threading.Event()
        self._parked = False
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._teleop: Any | None = None
        self._client: Any | None = None
        self._controller: Any | None = None
        self._config: Any | None = None
        self._latest: WristTeleopState | None = None
        self._sequence = 0
        self._baseline: tuple[float, float] | None = None
        self._pending_master_line: Any | None = None
        self._spacemouse_velocity = np.zeros(2, dtype=np.float64)  # requested FE/RU deg/s
        self._spacemouse_applied_velocity = np.zeros(2, dtype=np.float64)
        self._spacemouse_override = False
        self._spacemouse_target = np.zeros(2, dtype=np.float64)
        self._spacemouse_last_update_s = 0.0
        self._policy_target_q: np.ndarray | None = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def start(self) -> None:
        if self._thread is not None:
            return
        (
            self._AngleUnwrapper,
            TeleopSerial,
            self._field_float_deg,
            self._parse_line,
            self._OneEuroFilter,
            self._OutputTargetShaper,
            Controller,
            load_config,
            self._map_master_relative_deg,
            OpenRBClient,
            resolve_openrb_port,
            resolve_teleop_port,
        ) = _vendor_imports()
        self._config = load_config(self.config_path)
        teleop = TeleopSerial(resolve_teleop_port(self.teleop_port), self.baud)
        client = OpenRBClient(
            resolve_openrb_port(self.openrb_port), self.baud, timeout=0.6
        ).connect()
        controller = Controller(client, self._config)
        self._teleop, self._client, self._controller = teleop, client, controller
        try:
            self._stop_master_stream()
            teleop.command(
                f"SET_PERIOD {self._config.stream.master_period_ms}",
                {"SET_PERIOD"},
                timeout_s=2.0,
            )
            initial = self._prepare_dynamic_home(controller)
            self._store_state(initial)
            self._next_state_s = time.monotonic()
            self._parked = True
        except BaseException:
            self._close_serials(return_zero=False)
            raise
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_guarded, name="wrist-master-slave", daemon=True
        )
        self._thread.start()

    def _stop_master_stream(self) -> None:
        """Stop ESP32 streaming without accepting stale queued TELE as an ACK."""
        assert self._teleop is not None
        self._teleop.write_command("STOP")
        deadline = time.monotonic() + 2.0
        last_line = None
        while time.monotonic() < deadline:
            line = self._teleop.read_line(0.1)
            if line is None:
                continue
            last_line = line
            parsed = self._parse_line(line)
            if parsed is None or not parsed.ok:
                continue
            if parsed.command == "STOP" or (
                parsed.command == "TELE" and parsed.fields.get("st") == "0"
            ):
                return
        raise TimeoutError(f"Timed out stopping ESP32 master stream; last_line={last_line!r}")

    def _run_soft_home(self, controller: Any, command_period: float) -> Any:
        """Use the OpenRB position loop at a temporarily reduced motor rate."""
        config = controller.config.controller

        def configure(max_rate: int, fe_ff: int, ru_ff: int) -> None:
            disabled = controller.client.set_output_closed_loop_enable(False)
            if not disabled.ok:
                raise RuntimeError(disabled.raw_lines[-1])
            response = controller.client.set_output_direct_config(
                config.fe_kp_milli,
                config.fe_ki_milli,
                config.ru_kp_milli,
                config.ru_ki_milli,
                max_rate,
                config.deadband_cdeg,
                fe_ff,
                ru_ff,
                config.fe_moving_feedback_limit_raw,
                config.ru_moving_feedback_limit_raw,
            )
            if not response.ok:
                raise RuntimeError(response.raw_lines[-1])
            enabled = controller.client.set_output_closed_loop_enable(True)
            if not enabled.ok:
                raise RuntimeError(enabled.raw_lines[-1])

        configure(SPACEMOUSE_SOFT_HOME_MAX_MOTOR_RATE_RAW_S, 0, 0)
        try:
            controller.stream_target_deg(0.0, 0.0)
            output = _wait_home_settled(controller, self._store_state)
        finally:
            # Restore normal collection tracking only after the bounded HOME
            # attempt. Re-enabling initializes the target from measured output,
            # so a failed HOME cannot create a latent position jump.
            configure(
                config.max_motor_rate_raw_s,
                config.fe_velocity_ff_raw_per_deg,
                config.ru_velocity_ff_raw_per_deg,
            )
        with self._lock:
            self._spacemouse_target.fill(0.0)
            self._spacemouse_velocity.fill(0.0)
            self._spacemouse_applied_velocity.fill(0.0)
            self._spacemouse_override = False
        return output

    def _prepare_dynamic_home(self, controller: Any) -> Any:
        """HOME motors and define this run's measured encoders as FE/RU zero.

        The vendor YAML remains read-only. Every process start and explicit
        HOME/park gets a fresh runtime baseline so assembly-dependent absolute
        encoder offsets cannot invalidate an otherwise valid mechanical HOME.
        """
        state, baseline = prepare_dynamic_home(controller, state_callback=self._store_state)
        self._baseline = baseline
        if baseline is not None:
            self._write_runtime_baseline(*baseline)
        return state

    @staticmethod
    def _write_runtime_baseline(enc0: float, enc1: float) -> None:
        path = PROJECT_ROOT / "data" / "wrist_home_baseline.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            "schema_version: 1\n"
            "source: runtime_home_all\n"
            f"captured_at_unix_s: {time.time():.6f}\n"
            f"enc0_abs_deg: {enc0:.6f}\n"
            f"enc1_abs_deg: {enc1:.6f}\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _initialize_master_stream(self, output_state: Any) -> None:
        assert self._teleop is not None and self._controller is not None
        # START is idempotent from the adapter's perspective. The ESP32 can
        # already be streaming TELE after a park/resume transition; in that
        # case command() never sees a separate START acknowledgement because
        # the serial stream is continuously occupied by TELE frames. Accept a
        # fresh TELE frame as proof that the master stream is alive instead of
        # failing the whole collection on a protocol race.
        started = None
        try:
            started = self._teleop.command("START", {"START"}, timeout_s=0.5)
        except TimeoutError:
            # Fall through to the live TELE stream. Some firmware revisions
            # acknowledge START only by continuing their already-live stream.
            self._teleop.write_command("START")
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                line = self._teleop.read_line(0.1)
                if line is None:
                    continue
                parsed = self._parse_line(line)
                if parsed is None or not parsed.ok:
                    continue
                if parsed.command in {"START", "TELE"}:
                    started = parsed
                    if parsed.command == "TELE":
                        self._pending_master_line = parsed
                    break
        if started is None:
            raise TimeoutError("Timed out waiting for ESP32 START/TELE acknowledgement")
        if "enc0_deg" not in started.fields or "enc1_deg" not in started.fields:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                line = self._teleop.read_line(0.1)
                if line is None:
                    continue
                parsed = self._parse_line(line)
                if (
                    parsed is not None
                    and parsed.ok
                    and parsed.command == "TELE"
                    and "enc0_deg" in parsed.fields
                    and "enc1_deg" in parsed.fields
                ):
                    started = parsed
                    self._pending_master_line = parsed
                    break
            else:
                raise TimeoutError("ESP32 wrist stream produced no encoder TELE frame")
        enc0 = self._field_float_deg(started.fields, "enc0_deg")
        enc1 = self._field_float_deg(started.fields, "enc1_deg")
        self._enc0 = self._AngleUnwrapper()
        self._enc1 = self._AngleUnwrapper()
        self._enc0.reset(enc0)
        self._enc1.reset(enc1)
        self._fe_filter = self._OneEuroFilter(
            min_cutoff=self._config.filter.one_euro_min_cutoff,
            beta=self._config.filter.one_euro_beta,
            d_cutoff=self._config.filter.one_euro_d_cutoff,
        )
        self._ru_filter = self._OneEuroFilter(
            min_cutoff=self._config.filter.one_euro_min_cutoff,
            beta=self._config.filter.one_euro_beta,
            d_cutoff=self._config.filter.one_euro_d_cutoff,
        )
        self._fe_filter.reset(0.0)
        self._ru_filter.reset(0.0)
        if self._controller.bounds is None:
            self._controller.inspect()
        self._shaper = self._OutputTargetShaper(self._controller.bounds, self._config)
        self._shaper.reset(0.0, 0.0)
        self._last_command_s = time.monotonic()
        self._next_command_s = self._last_command_s
        self._next_state_s = self._last_command_s
        self._store_state(output_state)

    def _run_guarded(self) -> None:
        try:
            self._run()
        except BaseException as exc:  # noqa: BLE001 - surfaced through check/state
            self._failure = exc
            self._stop.set()

    def _run(self) -> None:
        assert self._teleop is not None and self._controller is not None
        command_period = 1.0 / float(self._config.stream.command_hz)
        state_period = 1.0 / float(self._config.stream.state_hz)
        while not self._stop.is_set():
            if self._park_requested.is_set():
                self._park_running.set()
                self._stop_master_stream()
                output = self._prepare_dynamic_home(self._controller)
                self._store_state(output)
                self._parked = True
                self._park_requested.clear()
                self._park_running.clear()
                continue
            if self._resume_requested.is_set():
                if self._parked:
                    self._initialize_master_stream(self._controller.read_state())
                    with self._lock:
                        self._spacemouse_velocity.fill(0.0)
                        self._spacemouse_applied_velocity.fill(0.0)
                        self._spacemouse_override = False
                    self._parked = False
                self._resume_requested.clear()
                continue
            if self._soft_home_requested.is_set():
                self._soft_home_running.set()
                self._stop_master_stream()
                output = self._run_soft_home(self._controller, command_period)
                self._store_state(output)
                self._parked = True
                self._soft_home_requested.clear()
                self._soft_home_running.clear()
                continue
            if self._home_requested.is_set():
                self._home_running.set()
                self._stop_master_stream()
                output = self._prepare_dynamic_home(self._controller)
                self._store_state(output)
                self._parked = True
                self._home_requested.clear()
                self._home_running.clear()
                continue
            if self._parked:
                now = time.monotonic()
                self._sample_output_if_due(now, state_period)
                self._stop.wait(min(state_period, 0.01))
                continue
            with self._lock:
                external_velocity = self._spacemouse_velocity.copy()
                external_active = self._spacemouse_override
                policy_target = (
                    None if self._policy_target_q is None else self._policy_target_q.copy()
                )
            if policy_target is not None:
                now = time.monotonic()
                if now < self._next_command_s:
                    self._stop.wait(min(self._next_command_s - now, 0.005))
                    continue
                self._last_command_s = now
                # Policy outputs are already absolute FE/RU positions. Do not
                # apply the manual SpaceMouse 14 deg/s jog limiter here: that
                # second limiter can leave a return-to-zero target stranded at
                # the previous large angle while the model has moved on.
                self._spacemouse_target[:] = np.rad2deg(policy_target)
                assert self._controller.bounds is not None
                self._spacemouse_target[0] = np.clip(
                    self._spacemouse_target[0],
                    self._controller.bounds.fe_min_cdeg / 100.0,
                    self._controller.bounds.fe_max_cdeg / 100.0,
                )
                self._spacemouse_target[1] = np.clip(
                    self._spacemouse_target[1],
                    self._controller.bounds.ru_min_cdeg / 100.0,
                    self._controller.bounds.ru_max_cdeg / 100.0,
                )
                self._controller.stream_target_deg(*self._spacemouse_target)
                if now >= self._next_state_s:
                    self._store_state(self._controller.read_state())
                    self._next_state_s = now + state_period
                self._next_command_s = now + command_period
                continue
            if external_active:
                now = time.monotonic()
                if now < self._next_command_s:
                    self._stop.wait(min(self._next_command_s - now, 0.005))
                    continue
                # Never turn a scheduling/serial stall into one large wrist
                # position step. The next iteration will continue from the
                # command target at the normal jog rate.
                dt = float(np.clip(now - self._last_command_s, 0.001, 0.05))
                self._last_command_s = now
                if now - self._spacemouse_last_update_s > SPACEMOUSE_WRIST_COMMAND_LEASE_S:
                    external_velocity.fill(0.0)
                self._spacemouse_applied_velocity = _slew_velocity(
                    self._spacemouse_applied_velocity, external_velocity, dt
                )
                # Integrate from the command target; feedback serial latency must
                # not amplify lag into an oversized position step.
                self._spacemouse_target += self._spacemouse_applied_velocity * dt
                assert self._controller.bounds is not None
                self._spacemouse_target[0] = np.clip(
                    self._spacemouse_target[0],
                    self._controller.bounds.fe_min_cdeg / 100.0,
                    self._controller.bounds.fe_max_cdeg / 100.0,
                )
                self._spacemouse_target[1] = np.clip(
                    self._spacemouse_target[1],
                    self._controller.bounds.ru_min_cdeg / 100.0,
                    self._controller.bounds.ru_max_cdeg / 100.0,
                )
                self._controller.stream_target_deg(*self._spacemouse_target)
                # Do not perform a blocking serial state read on every command.
                # The old read-after-write made the 50 Hz jog loop wait for a
                # second round trip and showed up as visible SpaceMouse lag.
                if now >= self._next_state_s:
                    self._store_state(self._controller.read_state())
                    self._next_state_s = now + state_period
                self._next_command_s = now + command_period
                continue
            if self._pending_master_line is not None:
                parsed = self._pending_master_line
                self._pending_master_line = None
            else:
                line = self._teleop.read_line(0.02)
                if line is None:
                    self._sample_output_if_due(time.monotonic(), state_period)
                    continue
                parsed = self._parse_line(line)
            if parsed is None or not parsed.ok or parsed.command != "TELE":
                self._sample_output_if_due(time.monotonic(), state_period)
                continue
            now = time.monotonic()
            if now < self._next_command_s:
                continue
            dt = max(0.001, now - self._last_command_s)
            self._last_command_s = now
            self._next_command_s = max(self._next_command_s + command_period, now + command_period)
            enc0 = self._enc0.update(self._field_float_deg(parsed.fields, "enc0_deg"))
            enc1 = self._enc1.update(self._field_float_deg(parsed.fields, "enc1_deg"))
            desired_fe, desired_ru = self._map_master_relative_deg(enc0, enc1, self._config)
            filtered_fe = self._fe_filter.filter(desired_fe, dt)
            filtered_ru = self._ru_filter.filter(desired_ru, dt)
            target_fe, target_ru = self._shaper.update(filtered_fe, filtered_ru, dt)
            self._controller.stream_target_deg(target_fe, target_ru)
            self._sample_output_if_due(now, state_period)

    def _sample_output_if_due(self, now: float, state_period: float) -> None:
        """Sample OpenRB feedback independently of ESP32 master traffic."""
        if now < self._next_state_s:
            return
        assert self._controller is not None
        self._store_state(self._controller.read_state())
        self._next_state_s = now + state_period

    def _store_state(self, state: Any) -> None:
        self._sequence += 1
        sample = WristTeleopState(
            actual_q=np.deg2rad(np.asarray([state.fe_deg, state.ru_deg], dtype=np.float32)),
            target_q=np.deg2rad(
                np.asarray([state.target_fe_deg, state.target_ru_deg], dtype=np.float32)
            ),
            host_timestamp_s=time.monotonic(),
            sequence=self._sequence,
            motor_position_raw=np.asarray(
                [state.j1_present_raw, state.j2_present_raw], dtype=np.int64
            ),
            motor_goal_raw=np.asarray(
                [state.j1_goal_raw, state.j2_goal_raw], dtype=np.int64
            ),
            motor_rate_raw_s=np.asarray(
                [state.j1_rate_raw_s, state.j2_rate_raw_s], dtype=np.int64
            ),
        )
        with self._lock:
            self._latest = sample

    def state(self) -> WristTeleopState:
        self.check()
        with self._lock:
            if self._latest is None:
                raise RuntimeError("wrist has not produced state")
            return WristTeleopState(
                self._latest.actual_q.copy(),
                self._latest.target_q.copy(),
                self._latest.host_timestamp_s,
                self._latest.sequence,
                self._latest.motor_position_raw.copy(),
                self._latest.motor_goal_raw.copy(),
                self._latest.motor_rate_raw_s.copy(),
            )

    def check(self) -> None:
        if self._failure is not None:
            raise RuntimeError(f"wrist teleoperation failed: {self._failure}") from self._failure
        if self._thread is not None and not self._thread.is_alive() and not self._stop.is_set():
            raise RuntimeError("wrist teleoperation stopped unexpectedly")

    def set_spacemouse_velocity(self, fe_deg_s: float, ru_deg_s: float) -> None:
        """Override wrist motion from the SpaceMouse while Ctrl is held."""
        self.check()
        with self._lock:
            velocity = np.asarray((fe_deg_s, ru_deg_s), dtype=np.float64)
            if not np.isfinite(velocity).all():
                raise ValueError("SpaceMouse wrist velocity must be finite")
            self._spacemouse_velocity[:] = np.clip(velocity, -45.0, 45.0)
            self._spacemouse_last_update_s = time.monotonic()
            if not self._spacemouse_override:
                # The worker thread exclusively owns OpenRB serial reads. Use its
                # cached target when Ctrl takes over to avoid racing telemetry.
                if self._latest is None:
                    raise RuntimeError("wrist has not produced state")
                self._spacemouse_target[:] = np.rad2deg(self._latest.target_q)
                self._spacemouse_applied_velocity.fill(0.0)
            self._spacemouse_override = True

    def clear_spacemouse_velocity(self) -> None:
        """Stop jogging immediately and hold the last adjusted wrist target."""
        with self._lock:
            self._spacemouse_velocity[:] = 0.0
            self._spacemouse_applied_velocity.fill(0.0)

    def set_policy_target(self, target_q: Any) -> None:
        """Stream a bounded policy wrist target instead of the master pose."""
        self.check()
        target = np.asarray(target_q, dtype=np.float64).reshape(-1)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("policy wrist target must contain two finite values")
        with self._lock:
            self._policy_target_q = target.copy()
            self._spacemouse_override = False

    def clear_policy_target(self) -> None:
        with self._lock:
            self._policy_target_q = None

    def request_home(self) -> None:
        self.check()
        self.clear_spacemouse_velocity()
        self._home_requested.set()

    def request_soft_home(self) -> None:
        """Return FE/RU to runtime zero at the SpaceMouse wrist jog speed."""
        self.check()
        self.clear_spacemouse_velocity()
        self._soft_home_requested.set()

    def request_park(self) -> None:
        """Return FE/RU to zero and ignore master motion until resumed."""
        self.check()
        self._park_requested.set()

    def request_resume(self) -> None:
        """Resume the ESP32 master after an explicit HOME/park operation."""
        self.check()
        self._resume_requested.set()

    def home_status(self) -> dict[str, object]:
        self.check()
        if (
            self._home_requested.is_set()
            or self._home_running.is_set()
            or self._soft_home_requested.is_set()
            or self._soft_home_running.is_set()
            or self._park_requested.is_set()
            or self._park_running.is_set()
        ):
            return {"at_home": False, "detail": "wrist returning to zero"}
        state = self.state()
        error = float(np.max(np.abs(state.actual_q)))
        tolerance = np.deg2rad(_home_acceptance_tolerance_deg(self._config))
        return {
            "at_home": bool(error <= tolerance),
            "detail": (
                f"FE={np.rad2deg(state.actual_q[0]):+.2f} deg "
                f"RU={np.rad2deg(state.actual_q[1]):+.2f} deg"
            ),
            "error_rad": error,
        }

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            timeout = float(getattr(getattr(self._config, "settling", None), "timeout_s", 0.0))
            self._thread.join(timeout=max(2.0, timeout + 2.0))
        alive = self._thread is not None and self._thread.is_alive()
        self._close_serials(return_zero=True)
        self._thread = None
        if alive:
            raise RuntimeError("wrist teleoperation thread did not stop")

    def _close_serials(self, *, return_zero: bool) -> None:
        teleop, client, controller = self._teleop, self._client, self._controller
        self._teleop = self._client = self._controller = None
        if teleop is not None:
            with suppress(OSError, RuntimeError):
                teleop.write_command("STOP")
        if controller is not None:
            controller.shutdown(return_zero=return_zero)
        if return_zero and client is not None:
            hold_all = getattr(client, "hold_all", None)
            if callable(hold_all):
                held = hold_all()
                if not held.ok:
                    raise RuntimeError(held.raw_lines[-1])
        if client is not None:
            client.close()
        if teleop is not None:
            teleop.close()

    def __exit__(self, *_args: object) -> None:
        self.close()
