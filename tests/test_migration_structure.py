import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_migration_keeps_capability_boundaries() -> None:
    required = (
        "assets/robots/ur5_wrist_wujihand/ur5_wrist_wuji_right.urdf",
        "clients/ios/IPhonePoseStreamer/README.md",
        "configs/dataset.yaml",
        "configs/hardware.yaml",
        "configs/inference.yaml",
        "configs/training.yaml",
        "src/slai_mi/apps/collect_real.py",
        "src/slai_mi/apps/collect_sim.py",
        "src/slai_mi/apps/teleop_real.py",
        "src/slai_mi/apps/teleop_sim.py",
        "src/slai_mi/collection/episode.py",
        "src/slai_mi/datasets/lerobot_v3/contract.py",
        "src/slai_mi/devices/spacemouse/client.py",
        "src/slai_mi/devices/iphone/pose_hub.py",
        "src/slai_mi/devices/ur5/runtime.py",
        "src/slai_mi/devices/wujihand/safety.py",
        "src/slai_mi/observability/session.py",
        "src/slai_mi/policies/runtime.py",
        "src/slai_mi/retargeting/multiview.py",
        "src/slai_mi/runtime/real_workflows.py",
        "src/slai_mi/simulation/isaac/runtime.py",
        "src/slai_mi/simulation/isaac/robot_cfg.py",
        "src/slai_mi/simulation/runtime.py",
        "src/slai_mi/simulation/writers.py",
        "src/slai_mi/training/runtime.py",
        "src/slai_mi/ui/collection_frontend.py",
        "src/slai_mi/ui/pose_hub/server.py",
        "third_party/wuji-retargeting/LICENSE",
    )

    missing = [path for path in required if not (PROJECT_ROOT / path).is_file()]
    assert not missing, f"missing migration capability boundaries: {missing}"


def test_generated_artifacts_are_not_source_directories() -> None:
    forbidden = ("logs", "outputs", "checkpoints", "swanlog", "tmp")
    tracked = subprocess.run(
        ["git", "ls-files", "--", *forbidden],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert not tracked, f"generated artifact directories must stay untracked: {tracked}"
