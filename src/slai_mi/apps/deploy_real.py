"""Run a trained PI0.5 checkpoint through the supervised physical station."""

from __future__ import annotations

import argparse
import sys
import threading
import time

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
)
from slai_mi.runtime.real_workflows import _SignalStop, validate_real_hardware_config
from slai_mi.site_adapter import RealPolicyBridge, StationSession, _cameras


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


def _at_home(session: StationSession) -> bool:
    arm = np.asarray(session.read_ur5_state()["joints"], dtype=float)
    hand = np.asarray(session.read_wuji_positions(), dtype=float)
    return bool(
        np.max(np.abs(arm - session.ur5_home_joints)) <= 0.010
        and np.max(np.abs(hand - session.wuji_home_joints)) <= 0.100
    )


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

    validate_real_hardware_config(hardware, required=("ur5", "wujihand", "cameras"))
    session = StationSession(hardware, task)
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
    with _cameras(hardware) as cameras, session.lease(arm=False), _SignalStop(stop):
        if not _at_home(session):
            raise RuntimeError("robot is not at the commissioned task start pose; hardware stayed unarmed")
        bridge = RealPolicyBridge(session, config.get("input_schema"))
        while not stop.is_set() and (steps == 0 or completed < steps):
            started = time.monotonic()
            frames = cameras.read(timeout_s=1.0)
            batch = _observation(
                schema,
                frames,
                session.read_ur5_state(),
                session.read_wuji_positions(),
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
            bridge.apply(action)
            completed += 1
            stop.wait(max(0.0, period - (time.monotonic() - started)))
        if session.supervisor.armed:
            session.write_ur5_twist(
                np.zeros(6), acceleration=session.ur5_acceleration, duration_s=period
            )
    print_plan({"status": "complete", "steps": completed})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
