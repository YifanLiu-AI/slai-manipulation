"""Shared command-line safety and configuration helpers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_path(value: str | Path) -> Path:
    """Resolve a user path relative to the project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(value: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and fail with a useful error for invalid files."""
    path = project_path(value)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise SystemExit(f"Configuration file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"Invalid YAML configuration in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Configuration must contain a mapping: {path}")
    return payload


def require_real_robot_confirmation(enabled: bool, confirmation: str | None) -> None:
    """Require an unambiguous phrase before a process may command real hardware."""
    expected = "I_UNDERSTAND_REAL_ROBOT_MOTION"
    if enabled and confirmation != expected:
        raise SystemExit(
            "Real hardware remains disabled. Pass --execute-real "
            f"--confirm {expected} after completing the physical safety check."
        )


def print_plan(plan: dict[str, Any]) -> None:
    """Emit a stable, machine-readable dry-run plan."""
    print(json.dumps(plan, indent=2, ensure_ascii=False, default=str))


def enabled_devices(config: dict[str, Any]) -> list[str]:
    """Return enabled device sections while ignoring top-level metadata."""
    return sorted(
        name
        for name, section in config.items()
        if isinstance(section, dict) and section.get("enabled", False)
    )


def reexec_with_python(python: str | Path, module: str, argv: list[str], marker: str) -> int | None:
    """Re-enter a CLI in its pinned environment while preserving local source imports."""
    executable = project_path(python).absolute()
    if os.environ.get(marker) == "1":
        return None
    if not executable.is_file():
        raise ValueError(f"required Python does not exist: {executable}")
    environment = os.environ.copy()
    environment[marker] = "1"
    source = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(
        [str(executable), "-m", module, *argv], env=environment, check=False
    ).returncode
