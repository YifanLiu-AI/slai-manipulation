"""Run the wrist8d JAX PI0.5 checkpoint on the supervised real station."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from slai_mi.apps._common import (
    load_yaml,
    print_plan,
    project_path,
    require_real_robot_confirmation,
)
from slai_mi.devices.ur5.geometry import (
    project_pose,
    rotation_offset_rad,
)
from slai_mi.devices.ur5.process import UR5DriverProcess
from slai_mi.devices.wrist_sensor.limits import CONTROL_LIMITS_FE_RU
from slai_mi.runtime.hardware_supervisor import HardwareProcessSupervisor
from slai_mi.runtime.real_workflows import _SignalStop

RESULT_PREFIX = "SLAI_JAX_RESULT "


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--hardware-config", default="configs/hardware.yaml")
    parser.add_argument("--task", default="configs/tasks/block_into_box_pi05_wrist8d_ultraslow.yaml")
    parser.add_argument("--camera-base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--observe-live", action="store_true")
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--_jax-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _limit_norm(values: np.ndarray, limit: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    magnitude = float(np.linalg.norm(result))
    if magnitude > limit:
        result *= limit / magnitude
    return result


def _scale_to_workspace(
    twist: np.ndarray,
    current_pose: np.ndarray,
    start_pose: np.ndarray,
    max_offset_m: float,
    max_rotation_rad: float,
    horizon_s: float,
) -> np.ndarray:
    guarded = np.asarray(twist, dtype=np.float64).copy()
    projected = project_pose(current_pose, guarded, horizon_s)
    current_distance = float(np.linalg.norm(current_pose[:3] - start_pose[:3]))
    projected_distance = float(np.linalg.norm(projected[:3] - start_pose[:3]))
    if projected_distance > max_offset_m and projected_distance >= current_distance:
        low, high = 0.0, 1.0
        for _ in range(24):
            scale = (low + high) / 2.0
            candidate = current_pose[:3] + guarded[:3] * horizon_s * scale
            if float(np.linalg.norm(candidate - start_pose[:3])) <= max_offset_m:
                low = scale
            else:
                high = scale
        guarded[:3] *= low
    current_angle = rotation_offset_rad(start_pose, current_pose)
    projected_angle = rotation_offset_rad(start_pose, projected)
    if projected_angle > max_rotation_rad and projected_angle >= current_angle:
        low, high = 0.0, 1.0
        angular = guarded.copy()
        angular[:3] = 0.0
        for _ in range(24):
            scale = (low + high) / 2.0
            candidate = angular.copy()
            candidate[3:] *= scale
            angle = rotation_offset_rad(
                start_pose, project_pose(current_pose, candidate, horizon_s)
            )
            if angle <= max_rotation_rad:
                low = scale
            else:
                high = scale
        guarded[3:] *= low
    return guarded


def _run_jax_worker(args: argparse.Namespace) -> int:
    import io
    import urllib.request

    from openpi.policies import policy_config
    from PIL import Image

    from slai_mi.apps.pi05 import _settings
    from slai_mi.input_schema import enabled_cameras, load_input_schema, select_transformed_vector
    from slai_mi.training.pi05 import make_train_config

    inference = load_yaml(args.config)
    training_path = inference.get("training_config", "configs/pi05_wrist_8dof_h100_jax.yaml")
    training = load_yaml(training_path)
    settings = _settings(training)
    train_config = make_train_config(settings)
    checkpoint = project_path(inference["checkpoint"])
    load_started = time.monotonic()
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint,
        default_prompt=str(inference["deployment"]["task_prompt"]),
    )
    load_s = time.monotonic() - load_started

    def collect() -> tuple[dict[str, object], dict[str, object]]:
        hardware = load_yaml(args.hardware_config)
        schema = load_input_schema(project_path(inference["input_schema"]))
        ur5 = hardware["ur5"]
        wrist = hardware["wrist_sensor"]
        ur_code = """
import json, sys, rtde_receive
r = rtde_receive.RTDEReceiveInterface(sys.argv[1])
try:
 print(json.dumps({"q": r.getActualQ(), "tcp": r.getActualTCPPose(), "speed": r.getActualTCPSpeed(), "robot_mode": r.getRobotMode(), "safety_mode": r.getSafetyMode()}))
finally:
 r.disconnect()
"""
        ur = json.loads(
            subprocess.check_output(
                [str(ur5["driver_python"]), "-c", ur_code, str(ur5["host"])], text=True
            )
        )
        wrist_code = """
import json, math, sys
from openrb_bridge.output_v2 import load_output_v2_config
from openrb_bridge.pc_client.openrb_client import OpenRBClient
from openrb_bridge.serial_ports import resolve_openrb_port
c = load_output_v2_config(sys.argv[1])
client = OpenRBClient(resolve_openrb_port(sys.argv[2]), int(sys.argv[3]), timeout=2.0).connect()
try:
 r = client.read_wrist_state(retries=4)
finally:
 client.close()
if not r.ok:
 raise RuntimeError(r.raw_lines)
f = r.fields
print(json.dumps({"q": [math.radians(int(f["enc0_deg"])/100-c.output_zero.enc0_abs_deg), math.radians(int(f["enc1_deg"])/100-c.output_zero.enc1_abs_deg)], "fields": f}))
"""
        wrist_environment = os.environ.copy()
        vendor_root = str(project_path("third_party/02_Python_Client_CLI"))
        wrist_environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (vendor_root, wrist_environment.get("PYTHONPATH")))
        )
        wrist_sample = json.loads(
            subprocess.check_output(
                [
                    str(project_path(".venv/bin/python")),
                    "-c",
                    wrist_code,
                    str(project_path(wrist["config"])),
                    str(wrist.get("openrb_port", "auto")),
                    str(wrist.get("baud", 115200)),
                ],
                env=wrist_environment,
                text=True,
            )
        )
        images: dict[str, np.ndarray] = {}
        image_shapes: dict[str, list[int]] = {}
        for camera in enabled_cameras(schema):
            role, key = str(camera["role"]), str(camera["openpi_key"])
            url = f"{args.camera_base_url.rstrip('/')}/api/cameras/{role}/frame.jpg"
            with urllib.request.urlopen(url, timeout=2.0) as response:
                image = np.asarray(Image.open(io.BytesIO(response.read())).convert("RGB"))
            images[key] = image
            image_shapes[key] = list(image.shape)
        values = {
            str(schema["capture"]["tcp_pose"]["key"]): np.asarray(ur["tcp"], dtype=np.float32),
            str(schema["capture"]["state"]["key"]): np.concatenate(
                (np.asarray(ur["q"], dtype=np.float32), np.asarray(wrist_sample["q"], dtype=np.float32))
            ),
        }
        state = np.concatenate(
            [
                select_transformed_vector(
                    values[source["key"]], source, f"pi05.state.sources[{index}]"
                )
                for index, source in enumerate(schema["pi05"]["state"]["sources"])
            ],
            dtype=np.float32,
        )
        observation = {
            "state": state,
            **images,
            "prompt": str(inference["deployment"]["task_prompt"]),
        }
        metadata = {
            "ur": ur,
            "wrist_q_rad": wrist_sample["q"],
            "state": state.tolist(),
            "image_shapes": image_shapes,
        }
        return observation, metadata

    warmup_observation, _ = collect()
    warmup_started = time.monotonic()
    policy.infer(warmup_observation)
    warmup_s = time.monotonic() - warmup_started
    observation, metadata = collect()
    infer_started = time.monotonic()
    result = policy.infer(observation)
    infer_s = time.monotonic() - infer_started
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
        raise RuntimeError(f"JAX policy returned invalid wrist8d actions: {actions.shape}")
    print(
        RESULT_PREFIX
        + json.dumps(
            {
                "checkpoint_load_s": load_s,
                "warmup_s": warmup_s,
                "inference_s": infer_s,
                "policy_timing": result["policy_timing"],
                "observation": metadata,
                "actions": actions.tolist(),
            },
            separators=(",", ":"),
        )
    )
    return 0


def _live_inference(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    model_python = project_path(config.get("model_python", "third_party/openpi/.venv/bin/python"))
    environment = os.environ.copy()
    source_paths = (
        project_path("src"),
        project_path("third_party/openpi/src"),
        project_path("third_party/openpi/packages/openpi-client/src"),
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in source_paths), environment.get("PYTHONPATH", "")]
    )
    command = [
        str(model_python),
        "-m",
        "slai_mi.apps.deploy_real_jax",
        "--config",
        str(project_path(args.config)),
        "--hardware-config",
        str(project_path(args.hardware_config)),
        "--camera-base-url",
        args.camera_base_url,
        "--_jax-worker",
    ]
    completed = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            "JAX live inference failed:\n" + (completed.stderr or completed.stdout).strip()
        )
    line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith(RESULT_PREFIX)),
        None,
    )
    if line is None:
        raise RuntimeError("JAX worker returned no structured result")
    return json.loads(line.removeprefix(RESULT_PREFIX))


class _WristOutput:
    def __init__(self, hardware: dict[str, Any], max_speed_rad_s: float) -> None:
        self.hardware = hardware["wrist_sensor"]
        self.max_speed_rad_s = max_speed_rad_s
        self.client: Any | None = None
        self.controller: Any | None = None
        self.target_q: np.ndarray | None = None
        self.lower: np.ndarray | None = None
        self.upper: np.ndarray | None = None

    def start(self) -> np.ndarray:
        vendor = str(project_path("third_party/02_Python_Client_CLI"))
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        from openrb_bridge.output_v2 import WristOutputV2Controller, load_output_v2_config
        from openrb_bridge.pc_client.openrb_client import OpenRBClient
        from openrb_bridge.serial_ports import resolve_openrb_port

        client = OpenRBClient(
            resolve_openrb_port(str(self.hardware.get("openrb_port", "auto"))),
            int(self.hardware.get("baud", 115200)),
            timeout=1.0,
        ).connect()
        controller = WristOutputV2Controller(
            client, load_output_v2_config(project_path(self.hardware["config"]))
        )
        try:
            response = controller.inspect()
            if not controller.output_zero_matches_config(response):
                raise RuntimeError("OpenRB output zero does not match the calibrated profile")
            state = controller.activate_at_current_output()
        except BaseException:
            client.close()
            raise
        firmware_lower = np.deg2rad(
            [controller.bounds.fe_min_cdeg / 100.0, controller.bounds.ru_min_cdeg / 100.0]
        )
        firmware_upper = np.deg2rad(
            [controller.bounds.fe_max_cdeg / 100.0, controller.bounds.ru_max_cdeg / 100.0]
        )
        mechanical_lower = np.asarray([limit.radians[0] for limit in CONTROL_LIMITS_FE_RU])
        mechanical_upper = np.asarray([limit.radians[1] for limit in CONTROL_LIMITS_FE_RU])
        self.client, self.controller = client, controller
        self.lower = np.maximum(firmware_lower, mechanical_lower)
        self.upper = np.minimum(firmware_upper, mechanical_upper)
        self.target_q = np.deg2rad([state.fe_deg, state.ru_deg])
        return self.target_q.copy()

    def write(self, desired_q: np.ndarray, period_s: float) -> np.ndarray:
        if self.controller is None or self.target_q is None:
            raise RuntimeError("wrist output is not active")
        desired = np.clip(np.asarray(desired_q, dtype=np.float64), self.lower, self.upper)
        delta = _limit_norm(desired - self.target_q, self.max_speed_rad_s * period_s)
        self.target_q = self.target_q + delta
        self.controller.stream_target_deg(*np.rad2deg(self.target_q))
        state = self.controller.read_state()
        return np.deg2rad([state.fe_deg, state.ru_deg])

    def close(self) -> None:
        client, controller = self.client, self.controller
        self.client = self.controller = None
        if controller is not None:
            with suppress(Exception):
                controller.shutdown(return_zero=False)
        if client is not None:
            with suppress(Exception):
                client.hold_all()
            client.close()


def _execute(args: argparse.Namespace, config: dict[str, Any], result: dict[str, Any], steps: int) -> None:
    hardware, task = load_yaml(args.hardware_config), load_yaml(args.task)
    profile = load_yaml(project_path(args.task).parent / task["control_profile_ref"])
    motion = profile["motion"]
    period_s = 1.0 / float(motion["control_hz"])
    linear_limit = float(motion["translation_speed"])
    angular_limit = float(motion["rotation_speed"])
    wrist_limit = min(float(motion["wrist_3_jog_speed"]), angular_limit)
    actions = np.asarray(result["actions"], dtype=np.float64)[:steps]
    actions[:, :3] = np.stack([_limit_norm(action[:3], linear_limit) for action in actions])
    actions[:, 3:6] = np.stack([_limit_norm(action[3:6], angular_limit) for action in actions])

    ur5_config = hardware["ur5"]
    ur5 = UR5DriverProcess(
        python=Path(ur5_config["driver_python"]),
        host=str(ur5_config["host"]),
        watchdog_s=float(ur5_config.get("driver_watchdog_s", 0.25)),
        max_linear_m_s=linear_limit,
        max_angular_rad_s=angular_limit,
    )
    supervisor = HardwareProcessSupervisor({"ur5": ur5})
    wrist = _WristOutput(hardware, wrist_limit)
    stop = threading.Event()
    completed = 0
    start_pose: np.ndarray | None = None
    final_pose: np.ndarray | None = None
    final_wrist: np.ndarray | None = None
    try:
        supervisor.start()
        initial = ur5.read_state()
        if float(np.linalg.norm(initial["tcp_speed"])) > 0.005:
            raise RuntimeError("UR5 is already moving; policy execution stayed disabled")
        ur5.prepare_control()
        start_pose = np.asarray(ur5.read_state()["tcp_pose"], dtype=np.float64)
        observed_pose = np.asarray(result["observation"]["ur"]["tcp"], dtype=np.float64)
        if (
            float(np.linalg.norm(start_pose[:3] - observed_pose[:3])) > float(motion["max_offset_mm"]) / 1000.0
            or rotation_offset_rad(observed_pose, start_pose) > math.radians(float(motion["max_rotation_deg"]))
        ):
            raise RuntimeError("UR5 moved outside the inference freshness envelope")
        wrist.start()
        supervisor.arm()
        started = time.monotonic()
        with _SignalStop(stop):
            for index, action in enumerate(actions):
                if stop.is_set():
                    break
                state = ur5.read_state()
                current_pose = np.asarray(state["tcp_pose"], dtype=np.float64)
                twist = _scale_to_workspace(
                    action[:6],
                    current_pose,
                    start_pose,
                    float(motion["max_offset_mm"]) / 1000.0,
                    math.radians(float(motion["max_rotation_deg"])),
                    0.25,
                )
                supervisor.call_with_peer_heartbeats(
                    "ur5",
                    lambda command=twist: ur5.write_twist(
                        command,
                        acceleration=float(motion["acceleration"]),
                        duration_s=period_s,
                    ),
                    check_after=False,
                )
                final_wrist = wrist.write(action[6:8], period_s)
                completed += 1
                stop.wait(max(0.0, started + (index + 1) * period_s - time.monotonic()))
        if supervisor.armed:
            supervisor.call_with_peer_heartbeats(
                "ur5",
                lambda: ur5.write_twist(
                    np.zeros(6), acceleration=float(motion["acceleration"]), duration_s=period_s
                ),
                check_after=False,
            )
        final_pose = np.asarray(ur5.read_state()["tcp_pose"], dtype=np.float64)
    finally:
        wrist.close()
        supervisor.stop()
    print_plan(
        {
            "status": "complete",
            "steps": completed,
            "duration_s": completed * period_s,
            "tcp_start": start_pose,
            "tcp_final": final_pose,
            "translation_mm": None
            if start_pose is None or final_pose is None
            else float(np.linalg.norm(final_pose[:3] - start_pose[:3]) * 1000.0),
            "rotation_deg": None
            if start_pose is None or final_pose is None
            else math.degrees(rotation_offset_rad(start_pose, final_pose)),
            "wrist_final_rad": final_wrist,
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._jax_worker:
        return _run_jax_worker(args)
    if args.observe_live and args.execute_real:
        raise SystemExit("--observe-live and --execute-real are mutually exclusive")
    config = load_yaml(args.config)
    steps = args.max_steps
    if steps is None:
        steps = int(config.get("deployment", {}).get("max_steps", 15))
    if not 1 <= steps <= 15:
        raise SystemExit("--max-steps must be between 1 and 15 for the smoke checkpoint")
    print_plan(
        {
            "app": "deploy_real_jax",
            "mode": "execute" if args.execute_real else "observe" if args.observe_live else "dry-run",
            "checkpoint": project_path(config["checkpoint"]),
            "steps": steps,
            "duration_s": steps / 15.0,
            "camera_base_url": args.camera_base_url,
        }
    )
    require_real_robot_confirmation(args.execute_real, args.confirm)
    if not args.execute_real and not args.observe_live:
        return 0
    result = _live_inference(args, config)
    actions = np.asarray(result["actions"], dtype=np.float32)
    print_plan(
        {
            "status": "live_inference_complete",
            "checkpoint_load_s": result["checkpoint_load_s"],
            "warmup_s": result["warmup_s"],
            "inference_s": result["inference_s"],
            "actions_shape": actions.shape,
            "first_action": actions[0],
        }
    )
    if args.observe_live:
        return 0
    _execute(args, config, result, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
