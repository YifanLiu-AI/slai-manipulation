"""Collect demonstrations from the physical robot system."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Sequence
from pathlib import Path

from slai_mi.devices.gripper import resolve_gripper_config
from slai_mi.runtime import (
    CollectionDependencies,
    RealCollectionWorkflow,
    StrategyProfileError,
    load_strategy_profile,
)
from slai_mi.runtime.adapters import (
    AdapterPluginError,
    adapter_plugin_spec,
    build_adapter_dependencies,
)

from ._common import (
    enabled_devices,
    load_yaml,
    print_plan,
    project_path,
    require_real_robot_confirmation,
)

_dependencies_factory = None
DEFAULT_STRATEGY = "ur5e_wrist_gripper_9dof_collection"
DEFAULT_TASK = "configs/tasks/task2_continuous.yaml"
HOME_PRESETS = {
    "parent": "../poses/tasks/task2_continuous_start.yaml",
    "last-point": "../poses/tasks/last-point.yaml",
}


def apply_home_preset(task: dict, preset: str | None) -> dict:
    """Return a per-run task config with an explicitly selected HOME pose."""
    selected = copy.deepcopy(task)
    if preset is None:
        return selected
    if preset not in HOME_PRESETS:
        raise ValueError(f"unknown HOME preset: {preset}")
    selected["start_pose_ref"] = HOME_PRESETS[preset]
    selected["home_preset"] = preset
    return selected


def require_explicit_legacy_binary_gripper(
    *, state_schema: str | None, gripper_enabled: bool, allowed: bool
) -> None:
    if gripper_enabled and state_schema != "task2_9dof_continuous" and not allowed:
        raise SystemExit(
            "binary gripper collection is retired; use the continuous TASK2 strategy, "
            "or pass --allow-legacy-binary-gripper for an intentional legacy run"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config", default="configs/hardware.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help="Strategy id from configs/strategies or an explicit YAML path",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--home-preset",
        choices=tuple(HOME_PRESETS),
        help="Select the run HOME pose without editing the task YAML",
    )
    episode_mode = parser.add_mutually_exclusive_group()
    episode_mode.add_argument("--episodes", type=int, default=1)
    episode_mode.add_argument(
        "--continuous",
        action="store_true",
        help="Keep accepting episodes until interrupted",
    )
    parser.add_argument("--execute-real", action="store_true", help="Allow real robot commands")
    parser.add_argument("--confirm", help="Required physical-motion confirmation phrase")
    parser.add_argument(
        "--allow-legacy-binary-gripper",
        action="store_true",
        help="Explicitly allow the retired endpoint-only 0/1 gripper collection contract",
    )
    parser.add_argument("--adapter-plugin", help="Site adapter factory as module:factory")
    parser.add_argument(
        "--dashboard-host",
        default="0.0.0.0",
        help="Dashboard listen address; defaults to all interfaces for LAN access",
    )
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-open-dashboard", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    if not 1 <= args.dashboard_port <= 65535:
        raise SystemExit("--dashboard-port must be between 1 and 65535")
    hardware = load_yaml(args.hardware_config)
    dataset = load_yaml(args.dataset_config)
    task = apply_home_preset(load_yaml(args.task), args.home_preset)
    selected_pose = Path(args.task).parent / str(task.get("start_pose_ref", ""))
    if args.home_preset is not None and not project_path(selected_pose).is_file():
        raise SystemExit(f"HOME preset pose does not exist: {selected_pose}")
    require_real_robot_confirmation(args.execute_real, args.confirm)
    try:
        strategy = load_strategy_profile(args.strategy)
        strategy.validate_for(
            "collect_real", hardware=hardware, task=task, execute=args.execute_real
        )
        hardware = strategy.configure_hardware(hardware)
    except StrategyProfileError as exc:
        raise SystemExit(f"Real collection strategy failed: {exc}") from exc
    sidecar = dataset.get("pi05_v21_sidecar", {})
    if not isinstance(sidecar, dict):
        raise SystemExit("dataset.pi05_v21_sidecar must be a mapping")
    if sidecar.get("enabled", False) and sidecar.get("backend") != "jax":
        raise SystemExit("enabled PI0.5 v2.1 sidecar requires explicit backend=jax")
    gripper_config = hardware.get("gripper", {})
    require_explicit_legacy_binary_gripper(
        state_schema=strategy.state_schema,
        gripper_enabled=isinstance(gripper_config, dict)
        and bool(gripper_config.get("enabled", False)),
        allowed=args.allow_legacy_binary_gripper,
    )
    gripper_driver = (
        resolve_gripper_config(gripper_config)["driver"]
        if isinstance(gripper_config, dict) and gripper_config.get("enabled", False)
        else "disabled"
    )
    print_plan(
        {
            "app": "collect_real",
            "mode": "execute" if args.execute_real else "dry-run",
            "task": task.get("task", {}).get("id"),
            "home_preset": args.home_preset or "task-default",
            "home_pose_ref": task.get("start_pose_ref"),
            "episodes": "continuous" if args.continuous else args.episodes,
            "dataset_format": dataset.get("format"),
            "dataset_root": str(project_path(dataset.get("root", "data/lerobot"))),
            "pi05_v21_sidecar": (
                {
                    "enabled": True,
                    "backend": "jax",
                    "root": str(project_path(sidecar["root"])),
                    "lerobot_v21_python": str(
                        project_path(sidecar["lerobot_v21_python"])
                    ),
                }
                if sidecar.get("enabled", False)
                else {"enabled": False}
            ),
            "enabled_devices": enabled_devices(hardware),
            "gripper_driver": gripper_driver,
            **strategy.plan_fields(),
            "dashboard": (
                "disabled"
                if args.no_dashboard
                else f"http://{args.dashboard_host}:{args.dashboard_port}"
            ),
        }
    )
    if args.execute_real:
        spec = adapter_plugin_spec(args.adapter_plugin, hardware)
        if spec is None and _dependencies_factory is None:
            raise SystemExit(
                "Real collection adapters are unavailable in this installation; no motion was sent."
            )
        try:
            dependencies: CollectionDependencies = (
                build_adapter_dependencies(
                    spec, CollectionDependencies, hardware, dataset, task
                )
                if spec is not None
                else _dependencies_factory(hardware, dataset, task)
            )
        except AdapterPluginError as exc:
            raise SystemExit(f"Real adapter plugin failed: {exc}") from exc
        try:
            RealCollectionWorkflow(
                hardware,
                dataset,
                task,
                dependencies,
                episode_limit=None if args.continuous else args.episodes,
                dashboard_enabled=not args.no_dashboard,
                dashboard_host=args.dashboard_host,
                dashboard_port=args.dashboard_port,
                dashboard_open_browser=not args.no_open_dashboard,
            ).run()
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"Real collection failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
