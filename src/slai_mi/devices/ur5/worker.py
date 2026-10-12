"""Python 3.11 worker owning UR RTDE connections and the control loop boundary."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from slai_mi.devices.worker_server import serve

RUNNING_ROBOT_MODE = 7
NORMAL_SAFETY_MODE = 1


class FakeRTDE:
    def __init__(self) -> None:
        self.twist = [0.0] * 6

    def state(self) -> dict[str, Any]:
        return {"tcp_pose": [0.0] * 6, "tcp_speed": self.twist, "joints": [0.0] * 6, "robot_mode": 7, "safety_mode": 1}

    def speed(self, twist: list[float], _acceleration: float, _duration_s: float) -> None:
        self.twist = twist

    def speed_joint(
        self, velocity: list[float], _acceleration: float, _duration_s: float
    ) -> None:
        self.twist = [0.0] * 6

    def joints_safe(self, _joints: list[float]) -> bool:
        return True

    def prepare(self, _tcp_pose: list[float]) -> None:
        return

    def stop(self) -> None:
        self.twist = [0.0] * 6

    def close(self) -> None:
        self.stop()


class VendorRTDE:
    def __init__(self, host: str) -> None:
        # ABI-bound imports and hardware connections live only in this worker.
        import rtde_receive

        from slai_mi.devices.ur5.runtime import exclusive_controller_lock

        self.host = host
        self._control_lock = exclusive_controller_lock()
        self._control_lock.__enter__()
        try:
            self.receiver = rtde_receive.RTDEReceiveInterface(host)
        except BaseException:
            self._control_lock.__exit__(None, None, None)
            raise
        self.control = None
        if not self.receiver.isConnected():
            raise RuntimeError("UR5 RTDE receive interface did not connect")

    def _start_control(self) -> None:
        import rtde_control

        self.control = rtde_control.RTDEControlInterface(self.host)
        if not self.control.isConnected():
            raise RuntimeError("UR5 RTDE control interface did not connect")
        # A prior process can disconnect while leaving its uploaded script alive.
        # Always replace it so this client's registers are the ones being consumed.
        self.control.stopScript()
        if not self.control.reuploadScript():
            raise RuntimeError("UR5 RTDE control script reupload failed")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.control.isProgramRunning():
                return
            time.sleep(0.05)
        raise RuntimeError("UR5 RTDE control script did not start after reupload")

    def state(self) -> dict[str, Any]:
        if self.receiver.isEmergencyStopped():
            raise RuntimeError("UR5 emergency stop is active")
        if self.receiver.isProtectiveStopped():
            raise RuntimeError("UR5 protective stop is active")
        return {
            "tcp_pose": list(self.receiver.getActualTCPPose()),
            "tcp_speed": list(self.receiver.getActualTCPSpeed()),
            "joints": list(self.receiver.getActualQ()),
            "robot_mode": int(self.receiver.getRobotMode()),
            "safety_mode": int(self.receiver.getSafetyMode()),
        }

    def speed(self, twist: list[float], acceleration: float, duration_s: float) -> None:
        if self.control is None:
            self._start_control()
        assert self.control is not None
        if not self.control.isProgramRunning():
            raise RuntimeError("UR5 RTDE control script is not running")
        if not self.control.isPoseWithinSafetyLimits(self.receiver.getActualTCPPose()):
            raise RuntimeError("current UR5 TCP pose is outside configured safety limits")
        if not self.control.speedL(twist, acceleration, duration_s):
            raise RuntimeError("UR5 speedL command returned false")

    def prepare(self, tcp_pose: list[float]) -> None:
        """Complete the legacy RTDE control preflight without sending motion."""
        if self.control is None:
            self._start_control()
        assert self.control is not None
        if not self.control.isProgramRunning():
            raise RuntimeError("UR5 RTDE control script is not running")
        if not self.control.isPoseWithinSafetyLimits(tcp_pose):
            raise RuntimeError("current UR5 TCP pose is outside configured safety limits")

    def speed_joint(
        self, velocity: list[float], acceleration: float, duration_s: float
    ) -> None:
        if self.control is None:
            self._start_control()
        assert self.control is not None
        if not self.control.isProgramRunning():
            raise RuntimeError("UR5 RTDE control script is not running")
        if not self.control.speedJ(velocity, acceleration, duration_s):
            raise RuntimeError("UR5 speedJ command returned false")

    def joints_safe(self, joints: list[float]) -> bool:
        if self.control is None:
            self._start_control()
        assert self.control is not None
        return bool(self.control.isJointsWithinSafetyLimits(joints))

    def stop(self) -> None:
        control, self.control = self.control, None
        if control is None:
            return
        try:
            if not control.speedL([0.0] * 6, 0.25, 0.1):
                control.stopScript()
            else:
                if not control.speedStop(1.0):
                    control.stopScript()
                else:
                    control.stopL(1.0)
        except Exception:  # noqa: BLE001 - best-effort vendor emergency stop fallback
            control.stopScript()
        finally:
            try:
                control.stopScript()
            finally:
                control.disconnect()

    def close(self) -> None:
        try:
            self.stop()
            self.receiver.disconnect()
        finally:
            self._control_lock.__exit__(None, None, None)


class UR5WorkerBackend:
    def __init__(self, rtde: Any, *, max_linear: float, max_angular: float):
        self.rtde = rtde
        self.max_linear = max_linear
        self.max_angular = max_angular

    def _state(self) -> dict[str, Any]:
        state = self.rtde.state()
        for name in ("tcp_pose", "tcp_speed", "joints"):
            values = [float(value) for value in state[name]]
            if len(values) != 6 or not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"invalid UR5 {name} feedback")
            state[name] = values
        if state["robot_mode"] != RUNNING_ROBOT_MODE:
            raise RuntimeError(f"UR5 robot mode is {state['robot_mode']}, expected RUNNING")
        if state["safety_mode"] != NORMAL_SAFETY_MODE:
            raise RuntimeError(f"UR5 safety mode is {state['safety_mode']}, expected NORMAL")
        return state

    def handle(self, message_type: str, payload: Mapping[str, Any], armed: bool) -> Mapping[str, Any]:
        if message_type == "read_state":
            return {"type": "state", "state": self._state()}
        if message_type == "prepare_control":
            state = self._state()
            self.rtde.prepare(state["tcp_pose"])
            return {"type": "control_ready"}
        if message_type == "stop_motion":
            self.rtde.stop()
            return {"type": "stopped"}
        if message_type == "write_twist":
            if not armed:
                raise RuntimeError("UR5 worker is not armed")
            self._state()
            twist = [float(value) for value in payload.get("twist", [])]
            if len(twist) != 6 or not all(math.isfinite(value) for value in twist):
                raise ValueError("UR5 twist must contain six finite values")
            if any(abs(value) > self.max_linear for value in twist[:3]):
                raise ValueError("UR5 linear speed exceeds worker limit")
            if any(abs(value) > self.max_angular for value in twist[3:]):
                raise ValueError("UR5 angular speed exceeds worker limit")
            acceleration = float(payload.get("acceleration", 0.0))
            duration_s = float(payload.get("duration_s", 0.0))
            if not 0.0 < acceleration <= 0.5 or not 0.0 < duration_s <= 0.1:
                raise ValueError("UR5 acceleration or command duration exceeds worker limit")
            self.rtde.speed(twist, acceleration, duration_s)
            return {"type": "command_ack"}
        if message_type == "write_joint_velocity":
            if not armed:
                raise RuntimeError("UR5 worker is not armed")
            state = self._state()
            velocity = [float(value) for value in payload.get("velocity", [])]
            if len(velocity) != 6 or not all(math.isfinite(value) for value in velocity):
                raise ValueError("UR5 joint velocity must contain six finite values")
            if any(abs(value) > self.max_angular for value in velocity):
                raise ValueError("UR5 joint speed exceeds worker limit")
            acceleration = float(payload.get("acceleration", 0.0))
            duration_s = float(payload.get("duration_s", 0.0))
            if not 0.0 < acceleration <= 0.5 or not 0.0 < duration_s <= 0.1:
                raise ValueError("UR5 acceleration or command duration exceeds worker limit")
            projected = [
                position + speed * max(0.25, 2.0 * duration_s)
                for position, speed in zip(state["joints"], velocity, strict=True)
            ]
            if not self.rtde.joints_safe(projected):
                raise RuntimeError("UR5 predicted joint command is outside safety limits")
            self.rtde.speed_joint(velocity, acceleration, duration_s)
            return {"type": "command_ack"}
        raise ValueError(f"unsupported UR5 command: {message_type}")

    def disable(self) -> None:
        self.rtde.stop()

    def close(self) -> None:
        self.rtde.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--watchdog-s", type=float, default=0.25)
    parser.add_argument("--max-linear", type=float, default=0.02)
    parser.add_argument("--max-angular", type=float, default=0.1)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()

    def factory() -> UR5WorkerBackend:
        rtde = FakeRTDE() if args.fake else VendorRTDE(args.host)
        return UR5WorkerBackend(rtde, max_linear=args.max_linear, max_angular=args.max_angular)

    serve(socket_path=args.socket, device_id=args.device_id, backend_factory=factory, watchdog_s=args.watchdog_s)


if __name__ == "__main__":
    main()
