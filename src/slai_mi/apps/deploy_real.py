"""Run a trained PI0.5 checkpoint through the supervised physical station."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from collections.abc import Iterator

import numpy as np

from slai_mi.apps._common import (
    load_yaml,
    print_plan,
    project_path,
    reexec_with_python,
    require_real_robot_confirmation,
)
from slai_mi.datasets.pi05 import policy_rgb
from slai_mi.input_schema import (
    capture_vector_names,
    enabled_cameras,
    load_input_schema,
    select_transformed_vector,
    split_capture_vector,
)
from slai_mi.runtime.real_workflows import _SignalStop, validate_real_hardware_config
from slai_mi.site_adapter import UR5OnlySession, _cameras
from slai_mi.devices.gripper import HiwonderGripper
from slai_mi.devices.wrist_sensor.teleop import WristMasterSlaveController
from slai_mi.devices.ur5.geometry import apply_relative_workspace_guard

CAMERA_OCCUPANCY_SERVICE = "slai-collection-ui.service"


@contextmanager
def _pause_camera_occupancy_service() -> Iterator[None]:
    """Release the shared RealSense devices and restore the prior UI state."""
    probe = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", CAMERA_OCCUPANCY_SERVICE],
        check=False,
    )
    if probe.returncode not in (0, 3):
        raise RuntimeError(
            f"could not inspect {CAMERA_OCCUPANCY_SERVICE}: exit {probe.returncode}"
        )
    was_active = probe.returncode == 0
    if was_active:
        subprocess.run(
            ["systemctl", "--user", "stop", CAMERA_OCCUPANCY_SERVICE],
            check=True,
        )
        # systemctl stop is synchronous, but give camera backends a short time
        # to close their V4L/RealSense handles before the next pipeline.start().
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            state = subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", CAMERA_OCCUPANCY_SERVICE],
                check=False,
            )
            if state.returncode == 3:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(f"{CAMERA_OCCUPANCY_SERVICE} did not stop")
    try:
        yield
    finally:
        if was_active:
            subprocess.run(
                ["systemctl", "--user", "start", CAMERA_OCCUPANCY_SERVICE],
                check=True,
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Inference YAML produced by slai-pi05 all")
    parser.add_argument("--hardware-config", default="configs/hardware.yaml")
    parser.add_argument("--task", default="configs/tasks/remove_objects_from_box_20mm.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--confirm")
    return parser


def _observation(schema, frames, ur5, hand, prompt, torch):
    host_times = [frame.host_timestamp_s for frame in frames.values()]
    max_skew = float(schema["synchronization"]["max_camera_skew_ms"]) / 1000.0
    if max(host_times) - min(host_times) > max_skew:
        raise RuntimeError("live camera skew exceeds the input schema")
    max_age = float(schema["synchronization"]["max_camera_age_ms"]) / 1000.0
    if time.monotonic() - min(host_times) > max_age:
        raise RuntimeError("live camera observation is stale")
    batch = {}
    for camera in enabled_cameras(schema):
        image = policy_rgb(frames[str(camera["role"])].color)
        batch[str(camera["policy_key"])] = (
            torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        )
    values = {
        str(schema["capture"]["tcp_pose"]["key"]): ur5["tcp_pose"],
        str(schema["capture"]["state"]["key"]): np.concatenate((ur5["joints"], hand)),
    }
    parts = [
        select_transformed_vector(values[source["key"]], source, f"pi05.state.sources[{index}]")
        for index, source in enumerate(schema["pi05"]["state"]["sources"])
    ]
    state = np.concatenate(parts, dtype=np.float32)
    state = np.pad(state, (0, int(schema["pi05"]["state"]["model_pad_to"]) - len(state)))
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0)
    batch["task"] = [prompt]
    return batch


def _at_home(session: UR5OnlySession, wrist, gripper) -> bool:
    arm = np.asarray(session.read_ur5_state()["joints"], dtype=float)
    wrist_status = wrist.home_status()
    gripper_state = gripper.read_state()
    return bool(
        np.max(np.abs(arm - session.ur5_home_joints)) <= 0.010
        and bool(wrist_status.get("at_home"))
        and gripper_state.binary_state == 0
    )


class Task2PolicyBridge:
    """Apply the schema's 9D action to UR5, OpenRB wrist, and gripper."""

    def __init__(self, session: UR5OnlySession, wrist, gripper, schema) -> None:
        self.session = session
        self.wrist = wrist
        self.gripper = gripper
        self.schema = schema
        self.start_pose: np.ndarray | None = None
        self._last_gripper: int | None = None

    def apply(self, action: object) -> None:
        components = split_capture_vector(self.schema, "action", action)
        try:
            twist = np.asarray(components["ur5"]["target_tcp_speed"], dtype=float)
            wrist_target = np.asarray(components["wrist"]["target_q"], dtype=float)
            gripper_target = float(components["gripper"]["target_q"][0])
        except KeyError as exc:
            raise ValueError(f"TASK2 action schema is missing component: {exc}") from exc
        if twist.shape != (6,) or wrist_target.shape != (2,):
            raise ValueError("TASK2 action must contain UR5 6D and wrist 2D actions")
        if not np.isfinite(twist).all() or not np.isfinite(wrist_target).all():
            raise ValueError("TASK2 action contains non-finite values")
        for values, limit in (
            (twist[:3], self.session.speed_settings.translation),
            (twist[3:], self.session.speed_settings.rotation),
        ):
            magnitude = float(np.linalg.norm(values))
            if magnitude > limit:
                values *= limit / magnitude
        state = self.session.read_ur5_state()
        current_pose = np.asarray(state["tcp_pose"], dtype=float)
        if self.start_pose is None:
            self.start_pose = current_pose.copy()
        twist, _ = apply_relative_workspace_guard(
            twist,
            current_pose,
            self.start_pose,
            self.session.max_offset_m,
            self.session.max_rotation_rad,
            0.25,
        )
        self.session.write_ur5_twist(
            twist,
            acceleration=self.session.ur5_acceleration,
            duration_s=self.session.control_period_s,
        )
        self.wrist.set_policy_target(wrist_target)
        command = int(float(np.clip(gripper_target, 0.0, 1.0)) >= 0.5)
        command_changed = command != self._last_gripper
        if command_changed:
            self.gripper.command(command)
            self._last_gripper = command
        self.session.check()
        return {
            "gripper_action": gripper_target,
            "gripper_command": command,
            "gripper_command_changed": command_changed,
        }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_yaml(args.config)
    hardware, task = load_yaml(args.hardware_config), load_yaml(args.task)
    checkpoint = project_path(config["checkpoint"])
    steps = args.max_steps
    if steps is None:
        steps = int(config.get("deployment", {}).get("max_steps", 0))
    if steps < 0:
        raise SystemExit("--max-steps must be non-negative")
    print_plan(
        {
            "app": "deploy_real",
            "mode": "execute" if args.execute_real else "dry-run",
            "checkpoint": checkpoint,
            "task": task.get("task", {}).get("id"),
            "max_steps": steps,
        }
    )
    require_real_robot_confirmation(args.execute_real, args.confirm)
    if not args.execute_real:
        return 0
    model_python = config.get("model_python", ".venv-lerobot-v3/bin/python")
    result = reexec_with_python(
        str(model_python),
        "slai_mi.apps.deploy_real",
        list(sys.argv[1:] if argv is None else argv),
        "SLAI_DEPLOY_REEXEC",
    )
    if result is not None:
        return result

    from slai_mi.policies.pi05_lerobot import PI05Policy

    validate_real_hardware_config(
        hardware, required=("ur5", "wrist_sensor", "gripper", "cameras")
    )
    session = UR5OnlySession(hardware, task)
    wrist_config = hardware["wrist_sensor"]
    wrist = WristMasterSlaveController(
        project_path(wrist_config["config"]),
        teleop_port=str(wrist_config.get("teleop_port", "auto")),
        openrb_port=str(wrist_config.get("openrb_port", "auto")),
        baud=int(wrist_config.get("baud", 115200)),
    )
    gripper_config = hardware["gripper"]
    gripper = HiwonderGripper(
        str(gripper_config["port"]),
        servo_id=int(gripper_config.get("servo_id", 1)),
        baud=int(gripper_config.get("baud", 115200)),
        move_time_ms=int(gripper_config.get("move_time_ms", 1000)),
        max_temperature_c=int(gripper_config.get("max_temperature_c", 60)),
        binary_threshold=int(gripper_config.get("binary_threshold", 500)),
        open_unload_delay_s=float(gripper_config.get("open_unload_delay_s", 1.5)),
    )
    schema = load_input_schema(config.get("input_schema"))
    dataset = config["dataset"]
    dataset["root"] = str(project_path(dataset["root"]))
    config["dataset"] = dataset
    policy = PI05Policy(config, checkpoint)
    physical_dim = len(capture_vector_names(schema, "action"))
    prompt = str(config.get("deployment", {}).get("task_prompt") or task["task"]["instruction"])
    timeout = float(config.get("deployment", {}).get("inference_timeout_s", 5.0))
    period = 1.0 / int(schema["pi05"]["fps"])
    stop = threading.Event()
    completed = 0
    with (
        _pause_camera_occupancy_service(),
        _cameras(hardware) as cameras,
        session.lease(arm=False),
        wrist,
        gripper,
        _SignalStop(stop),
    ):
        gripper.command(0)
        if not _at_home(session, wrist, gripper):
            raise RuntimeError(
                "TASK2 robot is not at the commissioned start pose; hardware stayed unarmed"
            )
        wrist.request_resume()
        bridge = Task2PolicyBridge(session, wrist, gripper, schema)
        while not stop.is_set() and (steps == 0 or completed < steps):
            started = time.monotonic()
            frames = cameras.read(timeout_s=1.0)
            batch = _observation(
                schema,
                frames,
                session.read_ur5_state(),
                np.concatenate(
                    (
                        wrist.state().actual_q,
                        np.asarray([gripper.read_state().actual_q[0]], dtype=np.float32),
                    )
                ),
                prompt,
                policy.torch,
            )
            if session.supervisor.armed:
                session.write_ur5_twist(
                    np.zeros(6), acceleration=session.ur5_acceleration, duration_s=period
                )
            inferred = time.monotonic()
            action = policy.infer(batch)[0, :physical_dim]
            if time.monotonic() - inferred > timeout:
                raise RuntimeError("PI0.5 inference exceeded the fail-closed timeout")
            if not session.supervisor.armed:
                session.prepare_ur5_control()
                session.arm()
            gripper_event = bridge.apply(action)
            if gripper_event["gripper_command_changed"]:
                gripper_state = gripper.read_state()
                print(
                    f"step={completed} gripper_action={gripper_event['gripper_action']:.3f} "
                    f"command={gripper_event['gripper_command']} "
                    f"position={gripper_state.position} "
                    f"actual={gripper_state.binary_state}"
                , flush=True)
            completed += 1
            stop.wait(max(0.0, period - (time.monotonic() - started)))
        if session.supervisor.armed:
            session.write_ur5_twist(
                np.zeros(6), acceleration=session.ur5_acceleration, duration_s=period
            )
        wrist.clear_policy_target()
        session.stop_ur5_motion()
    print_plan({"status": "complete", "steps": completed})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
