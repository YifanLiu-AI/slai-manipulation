"""LeRobot v3-side entry point for the PI0.5 converter."""

from __future__ import annotations

import argparse
from pathlib import Path

from slai_mi.input_schema import load_input_schema

from .pi05 import convert_v3_to_v21


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--v21-python", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args()
    schema = load_input_schema(args.schema)
    convert_v3_to_v21(
        args.source,
        args.target,
        repo_id=args.repo_id,
        source_fps=int(schema["capture"]["fps"]),
        policy_fps=int(schema["pi05"]["fps"]),
        v21_python=args.v21_python,
        schema_path=args.schema,
    )


if __name__ == "__main__":
    main()
