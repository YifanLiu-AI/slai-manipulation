"""Recover a PI0.5 v2.1 sidecar from completed incremental episode staging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pi05_incremental import recover_incremental_v21


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging", type=Path)
    args = parser.parse_args()
    report = recover_incremental_v21(args.staging)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
