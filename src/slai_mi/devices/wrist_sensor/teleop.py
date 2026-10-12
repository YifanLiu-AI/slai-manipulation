"""Non-interactive ESP32 master to OpenRB wrist adapter.

The protocol implementation stays in the vendor repository.  This module only
owns lifecycle, automatic master-zero establishment, and radians at the SLAI
boundary.
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
VENDOR_ROOT = PROJECT_ROOT / "third_party" / "02_Python_Client_CLI"
OPEN_LOOP_ROOT = VENDOR_ROOT / "open_loop_record"


@dataclass(frozen=True)
class WristTeleopState:
    actual_q: np.ndarray
    target_q: np.ndarray
    host_timestamp_s: float
    sequence: int


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
        self._spacemouse_velocity = np.zeros(2, dtype=np.float64)  # FE, RU deg/s
        self._spacemouse_override = False
        self._spacemouse_target = np.zeros(2, dtype=np.float64)
        self._spacemouse_last_update_s = 0.0

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
            teleop.command("STOP", {"STOP", "STATUS"}, timeout_s=2.0)
            teleop.command(
                f"SET_PERIOD {self._config.stream.master_period_ms}",
                {"SET_PERIOD"},
                timeout_s=2.0,
            )
            initial = self._prepare_dynamic_home(controller)
            self._store_state(initial)
            self._parked = True
        except BaseException:
            self._close_serials(return_zero=False)
            raise
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_guarded, name="wrist-master-slave", daemon=True
        )
        self._thread.start()

    def _prepare_dynamic_home(self, controller: Any) -> Any:
        """HOME the motors, then establish this assembly's encoder baseline.

        The vendor YAML remains immutable.  The current assembly's absolute
        encoder readings become the runtime output-zero reference, and a copy
        is written only to this project's diagnostic data directory.
        """
        # Lightweight test doubles and non-OpenRB adapters retain their
        # original lifecycle; the real OpenRB controller exposes this client
        # boundary and uses the dynamic assembly baseline below.
        if not all(
            callable(getattr(controller, name, None))
            for name in ("inspect", "_wait_motion_idle")
        ) or not hasattr(controller, "client"):
            return controller.prepare(auto_home_zero=True, force_home_zero=False)
        response = controller.inspect()
        if response.fields.get("active") == "1" or response.fields.get("vmode") == "1":
            stopped = controller.client.set_output_closed_loop_enable(False)
            if not stopped.ok:
                raise RuntimeError(stopped.raw_lines[-1])
        home = controller.client.home_all()
        if not home.ok:
            raise RuntimeError(home.raw_lines[-1])
        controller._wait_motion_idle()
        samples = []
        for _ in range(5):
            fields = controller.client.read_wrist_state(retries=6, retry_delay_s=0.05).fields
            samples.append((int(fields["enc0_deg"]) / 100.0, int(fields["enc1_deg"]) / 100.0))
            time.sleep(0.04)
        enc0 = float(np.mean([item[0] for item in samples]))
        enc1 = float(np.mean([item[1] for item in samples]))
        runtime_zero = replace(
            controller.config.output_zero,
            enc0_abs_deg=enc0,
            enc1_abs_deg=enc1,
        )
        controller.config = replace(controller.config, output_zero=runtime_zero)
        zero = controller.client.set_output_zero_absolute_cdeg(round(enc0 * 100), round(enc1 * 100))
        if not zero.ok:
            raise RuntimeError(zero.raw_lines[-1])
        self._baseline = (enc0, enc1)
        baseline_path = PROJECT_ROOT / "data" / "wrist_home_baseline.yaml"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            f"schema: slai_wrist_home_baseline_v1\nenc0_abs_deg: {enc0:.2f}\nenc1_abs_deg: {enc1:.2f}\n",
            encoding="utf-8",
        )
        return controller.prepare(auto_home_zero=False, force_home_zero=False)

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
            raise RuntimeError("ESP32 wrist master acknowledgement has no encoder fields")
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
        last_external = False
        while not self._stop.is_set():
            if self._park_requested.is_set():
                self._park_running.set()
                self._teleop.command("STOP", {"STOP"}, timeout_s=2.0)
                output = self._prepare_dynamic_home(self._controller)
                self._store_state(output)
                self._parked = True
                self._park_requested.clear()
                self._park_running.clear()
                continue
            if self._resume_requested.is_set():
                if self._parked:
                    self._initialize_master_stream(self._controller.read_state())
                    self._parked = False
                self._resume_requested.clear()
                continue
            if self._home_requested.is_set():
                self._home_running.set()
                self._teleop.command("STOP", {"STOP"}, timeout_s=2.0)
                output = self._prepare_dynamic_home(self._controller)
                self._store_state(output)
                self._parked = True
                self._home_requested.clear()
                self._home_running.clear()
                continue
            if self._parked:
                self._stop.wait(0.01)
                continue
            with self._lock:
                external_velocity = self._spacemouse_velocity.copy()
                external_active = self._spacemouse_override
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
                # Integrate from our own command target, not the feedback target.
                # Reading target_* here adds serial latency to the integrator and
                # can amplify lag into overshoot/oscillation.
                self._spacemouse_target += external_velocity * dt
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
                last_external = True
                continue
            if last_external:
                # Re-zero the master at the current output so releasing Ctrl
                # cannot snap the wrist to the master's old pose.
                self._initialize_master_stream(self._controller.read_state())
                last_external = False
            if self._pending_master_line is not None:
                parsed = self._pending_master_line
                self._pending_master_line = None
            else:
                line = self._teleop.read_line(0.02)
                if line is None:
                    continue
                parsed = self._parse_line(line)
            if parsed is None or not parsed.ok or parsed.command != "TELE":
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
            if now >= self._next_state_s:
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
            self._spacemouse_velocity[:] = np.clip(
                velocity,
                -14.0,
                14.0,
            )
            if not self._spacemouse_override:
                state = self._controller.read_state()
                self._spacemouse_target[:] = (
                    float(state.target_fe_deg), float(state.target_ru_deg)
                )
            self._spacemouse_override = True

    def clear_spacemouse_velocity(self) -> None:
        """Return control to the ESP32 master without a target jump."""
        with self._lock:
            self._spacemouse_velocity[:] = 0.0
            self._spacemouse_override = False
            self._spacemouse_target.fill(0.0)

    def request_home(self) -> None:
        self.check()
        self.clear_spacemouse_velocity()
        self._home_requested.set()

    def request_park(self) -> None:
        """Return FE/RU to zero and ignore master motion until resumed."""
        self.check()
        self._park_requested.set()

    def request_resume(self) -> None:
        """Use the current master pose as zero and resume wrist following."""
        self.check()
        self._resume_requested.set()

    def home_status(self) -> dict[str, object]:
        self.check()
        if (
            self._home_requested.is_set()
            or self._home_running.is_set()
            or self._park_requested.is_set()
            or self._park_running.is_set()
        ):
            return {"at_home": False, "detail": "wrist returning to zero"}
        state = self.state()
        error = float(np.max(np.abs(state.actual_q)))
        tolerance = np.deg2rad(float(self._config.settling.tolerance_deg))
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
        if client is not None:
            client.close()
        if teleop is not None:
            teleop.close()

    def __exit__(self, *_args: object) -> None:
        self.close()
