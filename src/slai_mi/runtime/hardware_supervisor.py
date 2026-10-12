"""Fail-closed supervisor for independently isolated hardware processes."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import TypeVar

from slai_mi.devices.driver_process import DriverProcess

T = TypeVar("T")


class HardwareProcessSupervisor:
    """Start all drivers, require explicit arm, and stop all after any failure."""

    def __init__(
        self,
        drivers: Mapping[str, DriverProcess],
        *,
        heartbeat_interval_s: float = 0.1,
    ):
        if not drivers:
            raise ValueError("at least one hardware driver is required")
        if heartbeat_interval_s <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.drivers = dict(drivers)
        self.heartbeat_interval_s = heartbeat_interval_s
        self.armed = False
        self.failure: BaseException | None = None
        self.stop_event = threading.Event()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_threads: list[threading.Thread] = []
        self._failure_lock = threading.Lock()

    def start(self) -> None:
        started: list[DriverProcess] = []
        try:
            for driver in self.drivers.values():
                driver.start()
                driver.heartbeat()
                started.append(driver)
            self._start_heartbeat_threads()
        except BaseException:
            self._stop_heartbeat_threads()
            for driver in reversed(started):
                driver.disable()
                driver.stop()
            raise

    def _start_heartbeat_threads(self) -> None:
        self._heartbeat_stop.clear()

        def heartbeat_driver(name: str, driver: DriverProcess) -> None:
            while not self._heartbeat_stop.wait(self.heartbeat_interval_s):
                try:
                    driver.heartbeat()
                except BaseException as exc:  # noqa: BLE001 - hardware faults fail closed
                    if not self._heartbeat_stop.is_set():
                        self.fail_closed(
                            RuntimeError(f"{name} background heartbeat failed: {exc}")
                        )
                    return

        self._heartbeat_threads = [
            threading.Thread(
                target=heartbeat_driver,
                args=(name, driver),
                name=f"{name}-heartbeat",
            )
            for name, driver in self.drivers.items()
        ]
        for thread in self._heartbeat_threads:
            thread.start()

    def _stop_heartbeat_threads(self) -> None:
        self._heartbeat_stop.set()
        current = threading.current_thread()
        for thread in self._heartbeat_threads:
            if thread is not current:
                thread.join()
        self._heartbeat_threads.clear()

    def arm(self) -> None:
        if self.failure is not None or self.stop_event.is_set():
            raise RuntimeError("failed supervisor cannot be re-armed; create a new session")
        armed: list[DriverProcess] = []
        try:
            for driver in self.drivers.values():
                driver.arm()
                armed.append(driver)
            self.armed = True
        except BaseException:
            for driver in reversed(armed):
                driver.disable()
            raise

    def check(self) -> None:
        if self.failure is not None:
            raise RuntimeError(f"hardware supervisor failed closed: {self.failure}") from self.failure
        try:
            for driver in self.drivers.values():
                driver.heartbeat()
        except BaseException as exc:
            self.fail_closed(exc)
            raise RuntimeError(f"hardware supervisor failed closed: {exc}") from exc

    def call_with_peer_heartbeats(
        self,
        device_id: str,
        operation: Callable[[], T],
        *,
        interval_s: float = 0.1,
    ) -> T:
        """Run a blocking device call while heartbeating every other driver."""
        if device_id not in self.drivers:
            raise KeyError(device_id)
        if interval_s <= 0:
            raise ValueError("heartbeat interval must be positive")

        stopped = threading.Event()
        heartbeat_failures: list[BaseException] = []

        def keep_peers_alive() -> None:
            try:
                while not stopped.is_set():
                    for name, driver in self.drivers.items():
                        if name != device_id:
                            driver.heartbeat()
                    stopped.wait(interval_s)
            except Exception as exc:  # noqa: BLE001 - driver faults must fail closed
                heartbeat_failures.append(exc)
                stopped.set()

        thread = threading.Thread(
            target=keep_peers_alive,
            name=f"{device_id}-peer-heartbeats",
        )
        thread.start()
        operation_failure: Exception | None = None
        result: T | None = None
        try:
            result = operation()
        except Exception as exc:  # noqa: BLE001 - operation faults must fail closed
            operation_failure = exc
        finally:
            stopped.set()
            thread.join()

        failure = heartbeat_failures[0] if heartbeat_failures else operation_failure
        if failure is not None:
            self.fail_closed(failure)
            raise RuntimeError(f"hardware supervisor failed closed: {failure}") from failure
        self.check()
        return result  # type: ignore[return-value]

    def fail_closed(self, failure: BaseException) -> None:
        with self._failure_lock:
            if self.failure is not None:
                return
            self.failure = failure
            self.stop_event.set()
            self._heartbeat_stop.set()
            self.armed = False
            for driver in self.drivers.values():
                driver.disable()

    def stop(self) -> None:
        self.stop_event.set()
        self.armed = False
        self._stop_heartbeat_threads()
        for driver in self.drivers.values():
            driver.disable()
        for driver in reversed(tuple(self.drivers.values())):
            driver.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()
