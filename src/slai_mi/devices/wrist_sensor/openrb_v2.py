"""Optional OpenRB FE/RU output-loop bridge used by real collection.

The protocol and controller remain owned by the original Wrist2Wrist project.
This adapter only exposes the lifecycle needed by ``scx``: connect, HOME to
the calibrated output zero, read FE/RU, and close.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

from .teleop import prepare_dynamic_home

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BRIDGE_ROOT = PROJECT_ROOT / "third_party" / "02_Python_Client_CLI"
DEFAULT_CONFIG = DEFAULT_BRIDGE_ROOT / "closed_loop_record" / "wrist_output_v2.yaml"


class OpenRBWristV2:
    """Thread-safe bridge around the original project's V2 controller."""

    def __init__(self, config: str | Path = DEFAULT_CONFIG, *, port: str = "auto", baud: int = 115200, mode: str = "closed_loop"):
        self.config_path = Path(config).expanduser().resolve()
        self.port = port
        self.baud = int(baud)
        if mode not in {"closed_loop", "hold"}:
            raise ValueError("wrist mode must be closed_loop or hold")
        self.mode = mode
        self._lock = threading.RLock()
        self._client: Any | None = None
        self._controller: Any | None = None
        self._state: Any | None = None
        self._home_thread: threading.Thread | None = None
        self._home_error: BaseException | None = None
        self._home_started = False

    @staticmethod
    def _imports() -> tuple[Any, Any, Any, Any]:
        root = str(DEFAULT_BRIDGE_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from openrb_bridge.output_v2 import WristOutputV2Controller, load_output_v2_config
        from openrb_bridge.pc_client.openrb_client import OpenRBClient
        from openrb_bridge.serial_ports import resolve_openrb_port

        return OpenRBClient, WristOutputV2Controller, load_output_v2_config, resolve_openrb_port

    def start(self, *, home: bool = True) -> None:
        with self._lock:
            if self._controller is not None:
                return
            OpenRBClient, Controller, load_config, resolve_port = self._imports()
            actual_port = resolve_port(self.port)
            client = OpenRBClient(actual_port, self.baud, timeout=2.0).connect()
            try:
                controller = Controller(client, load_config(self.config_path))
                self._client, self._controller = client, controller
            except Exception:
                client.close()
                raise
        if home:
            try:
                self.home_and_hold() if self.mode == "hold" else self.home()
            except Exception:
                self.close()
                raise

    def home(self) -> Any:
        with self._lock:
            if self._controller is None:
                raise RuntimeError("OpenRB wrist is not started")
            state, _baseline = prepare_dynamic_home(self._controller)
            self._state = state
            self._home_started = True
            self._home_error = None
            return state

    def home_and_hold(self) -> Any:
        with self._lock:
            if self._client is None or self._controller is None:
                raise RuntimeError("OpenRB wrist is not started")
            response = self._client.home_all()
            if not response.ok:
                raise RuntimeError(response.raw_lines[-1])
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                status = self._client.get_motion_status(retries=4)
                if status.fields.get("active") == "0":
                    break
                time.sleep(0.05)
            else:
                raise TimeoutError("OpenRB HOME_ALL timed out")
            held = self._client.hold_all()
            if not held.ok:
                raise RuntimeError(held.raw_lines[-1])
            self._state = self._read_hold_state()
            self._home_started = True
            self._home_error = None
            return self._state

    def _read_hold_state(self) -> Any:
        fields = self._client.read_wrist_state().fields
        zero = self._controller.config.output_zero
        enc0 = int(fields["enc0_deg"]) / 100.0
        enc1 = int(fields["enc1_deg"]) / 100.0
        return SimpleNamespace(
            fe_deg=enc0 - zero.enc0_abs_deg,
            ru_deg=enc1 - zero.enc1_abs_deg,
            target_fe_deg=enc0 - zero.enc0_abs_deg,
            target_ru_deg=enc1 - zero.enc1_abs_deg,
            active=False,
            zero_valid=False,
        )

    def request_home(self) -> None:
        with self._lock:
            if self._controller is None:
                raise RuntimeError("OpenRB wrist is not started")
            if self._home_thread is not None and self._home_thread.is_alive():
                return
            self._home_started = False
            self._home_error = None
            self._home_thread = threading.Thread(target=self._home_worker, name="openrb-wrist-home", daemon=True)
            self._home_thread.start()

    def _home_worker(self) -> None:
        try:
            self.home_and_hold() if self.mode == "hold" else self.home()
        except (RuntimeError, TimeoutError) as exc:  # transfer to polling thread
            with self._lock:
                self._home_error = exc

    def state(self) -> Any:
        home_thread = self._home_thread
        if home_thread is not None and home_thread.is_alive():
            with self._lock:
                if self._state is not None:
                    return self._state
        with self._lock:
            if self._controller is None:
                raise RuntimeError("OpenRB wrist is not started")
            self._state = self._read_hold_state() if self.mode == "hold" else self._controller.read_state()
            return self._state

    def home_status(self) -> dict[str, object]:
        with self._lock:
            if self._home_error is not None:
                raise RuntimeError(f"OpenRB FE/RU HOME failed: {self._home_error}") from self._home_error
            state = self._state
            running = self._home_thread is not None and self._home_thread.is_alive()
        if running or not self._home_started:
            return {"at_home": False, "detail": "OpenRB FE/RU 回零中"}
        if self.mode == "hold":
            return {"at_home": True, "detail": "OpenRB 电机已 HOME 并 HOLD"}
        state = self.state()
        tolerance = float(self._controller.config.settling.tolerance_deg)
        error = max(abs(float(state.fe_deg)), abs(float(state.ru_deg)))
        return {
            "at_home": bool(error <= tolerance),
            "detail": f"FE={state.fe_deg:+.2f}° RU={state.ru_deg:+.2f}°",
            "fe_error_deg": abs(float(state.fe_deg)),
            "ru_error_deg": abs(float(state.ru_deg)),
        }

    def close(self) -> None:
        with self._lock:
            client, controller = self._client, self._controller
            self._client = self._controller = None
        if controller is not None and self.mode == "hold" and client is not None:
            # Collection shutdown uses the original project's verified
            # power-off protocol before releasing the serial connection.
            self._client, self._controller = client, controller
            try:
                self.home_and_hold()
            finally:
                with self._lock:
                    self._client = self._controller = None
        elif controller is not None:
            controller.shutdown(return_zero=True)
        if client is not None:
            client.close()

    def __enter__(self) -> Self:
        self.start(home=True)
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
