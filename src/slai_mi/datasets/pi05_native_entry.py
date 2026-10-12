"""Environment-specific entry points for schema-driven PI0.5 training views."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pi05_native import build_native_v21, upgrade_native_v21_to_v30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    v21 = subparsers.add_parser("build-v21")
    v21.add_argument("source", type=Path)
    v21.add_argument("target", type=Path)
    v21.add_argument("--source-repo-id", required=True)
    v21.add_argument("--target-repo-id", required=True)
    v21.add_argument("--schema", type=Path, required=True)

    v30 = subparsers.add_parser("upgrade-v30")
    v30.add_argument("source", type=Path)
    v30.add_argument("target", type=Path)
    v30.add_argument("--repo-id", required=True)

    args = parser.parse_args()
    if args.command == "build-v21":
        result = build_native_v21(
            args.source,
            args.target,
            source_repo_id=args.source_repo_id,
            target_repo_id=args.target_repo_id,
            schema_path=args.schema,
        )
    else:
        result = upgrade_native_v21_to_v30(args.source, args.target, repo_id=args.repo_id)
    print(result.resolve())


if __name__ == "__main__":
    main()
