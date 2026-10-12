"""Safe, backend-neutral policy inference command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from slai_mi.apps._common import reexec_with_python
from slai_mi.policies.runtime import build_inference_plan, execute_inference


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--checkpoint", help="Override the checkpoint from the config")
    parser.add_argument(
        "--execute", action="store_true", help="Invoke the configured offline/simulation backend"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_inference_plan(Path(args.config), checkpoint_override=args.checkpoint)
        if args.execute:
            configured_python = plan.config.get("model_python")
            if configured_python is None and plan.backend == "slai_mi.policies.pi05_lerobot:factory":
                configured_python = ".venv-lerobot-v3/bin/python"
            if configured_python is not None:
                result = reexec_with_python(
                    str(configured_python),
                    "slai_mi.inference",
                    list(sys.argv[1:] if argv is None else argv),
                    "SLAI_INFERENCE_REEXEC",
                )
                if result is not None:
                    return result
        result = execute_inference(plan) if args.execute else plan.as_dict(mode="dry-run")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
