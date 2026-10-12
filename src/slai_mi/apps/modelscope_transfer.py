"""Safe ModelScope dataset/checkpoint transfer commands for the CCI/H100 pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ._common import print_plan, project_path

ENDPOINT = "https://modelscope.cn"
REVISION = "master"
MANIFEST = ".slai-modelscope-manifest.json"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.strip().lower().replace("_", "-"))
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        raise ValueError("task/repository name must contain letters or digits")
    return slug


def dataset_repo(task: str) -> str:
    return f"tmp-{_slug(task)}-dataset"


def checkpoint_repo(task: str) -> str:
    return f"{_slug(task)}-checkpoint"


def _repo_id(namespace: str | None, name: str) -> str:
    owner = (namespace or os.environ.get("MODELSCOPE_NAMESPACE") or "").strip().strip("/")
    if not owner or "/" in owner:
        raise ValueError("set --namespace or MODELSCOPE_NAMESPACE to one ModelScope username")
    return f"{owner}/{name}"


def _files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != MANIFEST:
            yield path


def write_manifest(root: Path, *, kind: str, repo_id: str) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"transfer source is not a directory: {root}")
    entries = []
    total = 0
    for path in _files(root):
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        total += size
        entries.append({"path": path.relative_to(root).as_posix(), "size": size, "sha256": digest.hexdigest()})
    payload = {
        "schema_version": 1,
        "kind": kind,
        "repo_id": repo_id,
        "revision": REVISION,
        "file_count": len(entries),
        "total_bytes": total,
        "files": entries,
    }
    output = root / MANIFEST
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = root / MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(f"downloaded transfer manifest is missing: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    missing, changed = [], []
    for entry in payload.get("files", []):
        path = root / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != int(entry["size"]) or digest.hexdigest() != entry["sha256"]:
            changed.append(entry["path"])
    if missing or changed:
        raise RuntimeError(f"manifest verification failed: missing={missing}, changed={changed}")
    return payload


def _repo_type(kind: str) -> str:
    return "dataset" if kind == "dataset" else "model"


def _run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["MODELSCOPE_ENDPOINT"] = ENDPOINT
    subprocess.run(command, env=environment, check=True)


def _ensure_client() -> None:
    if shutil.which("ms-hub") is None:
        raise RuntimeError('ms-hub is missing; install "modelscope-hub==0.2.0"')


def _validate_pi05_dataset(root: Path) -> None:
    info = root / "meta" / "info.json"
    stats = root / "meta" / "stats.json"
    if not info.is_file() or not stats.is_file():
        return
    metadata = json.loads(info.read_text(encoding="utf-8"))
    features = metadata.get("features", {})
    if "observation.state" not in features or "action" not in features:
        return
    from slai_mi.datasets.pi05_native import validate_native_v30_stats

    validate_native_v30_stats(root)
    print(f"NORMALIZATION_CHECK_OK dataset={root}", flush=True)


def _upload(args: argparse.Namespace, *, kind: str) -> int:
    name = args.repo_name or (dataset_repo(args.task) if kind == "dataset" else checkpoint_repo(args.task))
    repo_id = _repo_id(args.namespace, name)
    source = project_path(args.source).resolve()
    if kind == "dataset":
        _validate_pi05_dataset(source)
    repo_type = _repo_type(kind)
    plan = {
        "app": "modelscope_transfer", "command": f"{kind}-upload", "mode": "execute" if args.execute else "dry-run",
        "repo_id": repo_id, "repo_type": repo_type, "visibility": "private", "revision": REVISION,
        "source": source, "max_workers": args.max_workers,
    }
    print_plan(plan)
    if not args.execute:
        return 0
    _ensure_client()
    write_manifest(source, kind=kind, repo_id=repo_id)
    if not args.skip_create:
        _run(["ms-hub", "create", repo_id, "--repo-type", repo_type, "--visibility", "private"])
    _run(["ms-hub", "upload", repo_id, str(source), "--repo-type", repo_type, "--max-workers", str(args.max_workers), "--commit-message", args.commit_message])
    return 0


def _download(args: argparse.Namespace, *, kind: str) -> int:
    name = args.repo_name or (dataset_repo(args.task) if kind == "dataset" else checkpoint_repo(args.task))
    repo_id = _repo_id(args.namespace, name)
    target = project_path(args.target).resolve()
    repo_type = _repo_type(kind)
    plan = {
        "app": "modelscope_transfer", "command": f"{kind}-download", "mode": "execute" if args.execute else "dry-run",
        "repo_id": repo_id, "repo_type": repo_type, "revision": args.revision, "target": target,
        "max_workers": args.max_workers, "range_workers": args.range_workers,
    }
    print_plan(plan)
    if not args.execute:
        return 0
    _ensure_client()
    target.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MODELSCOPE_ENDPOINT"] = ENDPOINT
    environment["MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS"] = str(args.range_workers)
    subprocess.run(["ms-hub", "download", repo_id, "--repo-type", repo_type, "--revision", args.revision, "--local-dir", str(target), "--max-workers", str(args.max_workers)], env=environment, check=True)
    _run(["ms-hub", "cache", "verify", repo_id, "--repo-type", repo_type, "--revision", args.revision, "--local-dir", str(target), "--fail-on-missing-files", "--fail-on-extra-files"])
    verified = verify_manifest(target)
    if kind == "dataset":
        _validate_pi05_dataset(target)
    result: dict[str, Any] = {
        "status": "verified",
        "repo_id": repo_id,
        "file_count": verified["file_count"],
        "total_bytes": verified["total_bytes"],
    }
    if kind == "checkpoint" and args.inference_template is not None:
        template_path = project_path(args.inference_template).resolve()
        output_path = project_path(args.inference_output).resolve()
        inference = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
        if not isinstance(inference, dict) or "dataset" not in inference:
            raise ValueError(f"invalid inference template: {template_path}")
        checkpoint = target
        if not (checkpoint / "config.json").is_file():
            candidates = list(target.rglob("config.json"))
            candidates = [path.parent for path in candidates if (path.parent / "policy_preprocessor.json").is_file()]
            if len(candidates) != 1:
                raise ValueError(f"could not identify one pretrained_model under {target}")
            checkpoint = candidates[0]
        inference["checkpoint"] = str(checkpoint)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml.safe_dump(inference, sort_keys=False), encoding="utf-8")
        result["inference_config"] = str(output_path)
        result["checkpoint"] = str(checkpoint)
    print_plan(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for kind in ("dataset", "checkpoint"):
        upload = sub.add_parser(f"{kind}-upload")
        upload.add_argument("--task", required=True)
        upload.add_argument("--source", type=Path, required=True)
        upload.add_argument("--namespace")
        upload.add_argument("--repo-name")
        upload.add_argument("--max-workers", type=int, default=8)
        upload.add_argument("--commit-message", default="SLAI temporary transfer")
        upload.add_argument("--skip-create", action="store_true", help="upload to an existing private repository")
        upload.add_argument("--execute", action="store_true")
        download = sub.add_parser(f"{kind}-download")
        download.add_argument("--task", required=True)
        download.add_argument("--target", type=Path, required=True)
        download.add_argument("--namespace")
        download.add_argument("--repo-name")
        download.add_argument("--revision", default=REVISION)
        download.add_argument("--max-workers", type=int, default=8)
        download.add_argument("--range-workers", type=int, default=4)
        if kind == "checkpoint":
            download.add_argument("--inference-template", type=Path)
            download.add_argument(
                "--inference-output",
                type=Path,
                default=Path("outputs/modelscope/inference.yaml"),
            )
        download.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_workers < 1 or getattr(args, "range_workers", 1) < 1:
        raise SystemExit("worker counts must be positive")
    kind, operation = args.command.split("-", 1)
    try:
        return _upload(args, kind=kind) if operation == "upload" else _download(args, kind=kind)
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"ModelScope transfer failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
