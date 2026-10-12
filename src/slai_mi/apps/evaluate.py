"""Command-line entry points for offline model, dataset, and sim-real evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from slai_mi.evaluation import action_metrics, compare_summaries, fit_camera_pose, summarize_images


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        output.expanduser().resolve().write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def _action(args: argparse.Namespace) -> dict[str, Any]:
    payload = np.load(args.input)
    return action_metrics(
        payload["predicted"], payload["target"], payload["ranges"], arm_dim=args.arm_dim
    )


def _domain_gap(args: argparse.Namespace) -> dict[str, Any]:
    reference = np.load(args.reference)["images"]
    candidate = np.load(args.candidate)["images"]
    reference_summary = summarize_images(reference, bins=args.bins)
    candidate_summary = summarize_images(candidate, bins=args.bins)
    return {
        "reference": reference_summary,
        "candidate": candidate_summary,
        "gap": compare_summaries(reference_summary, candidate_summary),
    }


def _camera(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = fit_camera_pose(
        payload["intrinsics"],
        payload["world_points"],
        payload["image_points"],
        weights=payload.get("weights"),
    )
    return {
        "eye": result.eye.tolist(),
        "rotation_vector": result.rotation_vector.tolist(),
        "translation_vector": result.translation_vector.tolist(),
        "pixel_errors": result.pixel_errors.tolist(),
        "weighted_rmse_px": result.weighted_rmse_px,
    }


def _dataset(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.root / "meta" / "robot_teleoperation_contract.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from slai_mi.datasets.lerobot_v3.configured import (
        GRIPPER_9DOF_CONTRACT_ID,
        WRIST_8DOF_CONTRACT_ID,
        ConfiguredDatasetContract,
    )

    if manifest.get("contract_id") == WRIST_8DOF_CONTRACT_ID:
        from slai_mi.input_schema import load_input_schema

        schema_path = args.schema or Path("configs/input_schemas/ur5e_wrist_8dof.yaml")
        return ConfiguredDatasetContract(load_input_schema(schema_path)).validate_root(args.root)
    if manifest.get("contract_id") == GRIPPER_9DOF_CONTRACT_ID:
        from slai_mi.input_schema import load_input_schema

        schema_path = args.schema or Path(
            "configs/input_schemas/ur5e_wrist_gripper_9dof.yaml"
        )
        return ConfiguredDatasetContract(load_input_schema(schema_path)).validate_root(args.root)
    from slai_mi.datasets.lerobot_v3 import validate_dataset_root

    return validate_dataset_root(args.root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    action = subparsers.add_parser("actions", help="score predicted actions from an NPZ file")
    action.add_argument("input", type=Path, help="NPZ containing predicted, target, and ranges")
    action.add_argument("--arm-dim", type=int, default=6)
    action.add_argument("--output", type=Path)
    action.set_defaults(handler=_action)

    domain = subparsers.add_parser("domain-gap", help="compare two RGB image batches")
    domain.add_argument("reference", type=Path, help="NPZ containing an images array")
    domain.add_argument("candidate", type=Path, help="NPZ containing an images array")
    domain.add_argument("--bins", type=int, default=32)
    domain.add_argument("--output", type=Path)
    domain.set_defaults(handler=_domain_gap)

    camera = subparsers.add_parser("camera", help="fit camera extrinsics from landmark JSON")
    camera.add_argument("input", type=Path)
    camera.add_argument("--output", type=Path)
    camera.set_defaults(handler=_camera)

    dataset = subparsers.add_parser("dataset", help="validate a canonical LeRobot v3 root")
    dataset.add_argument("root", type=Path)
    dataset.add_argument("--schema", type=Path)
    dataset.add_argument("--output", type=Path)
    dataset.set_defaults(handler=_dataset)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = args.handler(args)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"evaluation failed: {exc}") from exc
    _write_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
