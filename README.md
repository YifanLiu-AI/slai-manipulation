# SLaI Manipulation Intelligence

机器人操作智能项目，面向 Wujihand + UR5 的数据采集、模型训练与推理。

## 项目结构

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── assets/
│   └── robots/
│       └── ur5_wrist_wujihand/    # 28-DoF URDF and meshes
├── configs/
│   ├── hardware.yaml              # Physical-device setup
│   ├── input_schema.yaml          # Camera/DoF/channel mapping shared by all paths
│   ├── dataset.yaml               # LeRobot v3 contract
│   ├── tasks/                     # Task definitions
│   └── poses/                     # Named robot poses
├── data/
│   ├── raw/               # 原始采集数据
│   ├── training/pi05/     # PI0.5 LeRobot v2.1 数据 (not committed)
│   └── normalization/pi05/ # PI0.5 norm stats (not committed)
├── src/slai_mi/
│   ├── apps/              # Real/sim teleoperation and collection CLIs
│   ├── devices/           # SpaceMouse, cameras, UR5, Wujihand, iPhone, wrist
│   ├── collection/        # Synchronization, episodes, telemetry, recording
│   ├── datasets/          # LeRobot v3 schema, writer, merge, validation
│   ├── simulation/        # Isaac Lab scene and deterministic episode runtime
│   ├── retargeting/       # Hand geometry, calibration, MANO process boundary
│   ├── policies/          # Policy adapters and action chunk execution
│   ├── training/          # Dataset filtering and training runtime
│   ├── evaluation/        # Policy and sim-real metrics
│   ├── observability/     # Session manifests and incident classification
│   └── ui/                # Collection dashboard and public Pose Hub
├── clients/ios/           # iPhone pose-streaming client
├── third_party/           # Pinned Wuji retargeting subset and license
└── tests/
```

## 快速开始

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Pipeline

```text
teleop_real.py / teleop_sim.py
              ↓
collect_real.py / collect_sim.py -> data/raw -> data/processed
                                              ↓
                                         train.py -> inference.py
```

所有命令默认只输出 dry-run 计划，不控制真实机器人：

```bash
uv run slai-teleop-real
uv run slai-teleop-sim
uv run slai-collect-real
uv run slai-collect-sim
uv run slai-train
uv run slai-mi-infer
uv run slai-pi05 train
uv run slai-pi05 all --run-id RUN_NAME --execute --smoke
uv run slai-deploy-real --config outputs/pi05/RUN_NAME/inference.yaml
```

启动只读采集前端：

```bash
uv run slai-collection-ui --host 127.0.0.1 --port 8080
uv run slai-collection-ui --live --host 127.0.0.1 --port 8080
```

iPhone Pose Hub and the 4090 handoff bridge use package entry points and default
to dry-run plans:

```bash
uv run --extra pose-hub slai-pose-hub
uv run --extra pose-hub slai-pose-hub-bridge --session SESSION_ID
```

Production services pass `--run`; credentials remain in machine-local systemd
environment files. The local robot contract is `5005` for pose output and
`5006` for measured joint-state input. See [`docs/iphone_pose.md`](docs/iphone_pose.md).

真机执行需要完整硬件配置、生产 adapter、`--execute-real` 和命令提示的确认短语。
仿真执行需要在 `robot_teaching_isaaclab` 环境中使用 `--run`。

### 当前真机状态（2026-08-16）

第九轮已在现场真实验证 UR5、Wuji、三台 RealSense、`teleop_real --execute-real` 的
commissioning 控制路径，以及 `collect_real --episodes 1 --execute-real` 的 LeRobot v3
原子 episode 提交。验证数据位于
`data/lerobot/block_into_box-20260816T133658`（61 帧、三路 H.264 视频）。

本轮使用有界脚本输入完成安全验收，因为现场 SpaceMouse Pro 虽被发现且服务 active，
但没有产生 live motion event；真实物理 SpaceMouse 输入仍需现场重新移动/绑定后验收。
Wuji 相机手势 retargeting 现在由默认控制进程启动独立的
`wuji_retargeting_camera` Python 3.11 + MediaPipe 0.10.21 worker；真实相机无手测试会
fail-closed 返回 `None`，不再依赖 Python 3.13 中不存在的 legacy `mp.solutions`。

第十轮已真实跑通 PI0.5 v3→v2.1、norm stats、1-step LoRA checkpoint 和 checkpoint
推理。`slai-pi05 convert --execute` 会按同一 schema 原子派生 OpenPI v2.1、native v2.1
和可训练 native v3 视图，并校验训练所需 q01/q99。执行入口为
`slai-pi05 ... --execute` 与 `slai-mi-infer --execute`；旧的 `slai-infer` 仍作为兼容别名保留；详见
[`docs/pi05.md`](docs/pi05.md)。

## Simulation asset

The 28-DoF UR5 + wrist + Wujihand model is located at:

```text
assets/robots/ur5_wrist_wujihand/ur5_wrist_wuji_right.urdf
```

## Configuration safety

Physical device addresses and robot poses are intentionally disabled by default.
Set `enabled: true` only after filling and validating the corresponding values in
`configs/hardware.yaml` and `configs/poses/`.

相机字段、state/action DoF 顺序与 mask、同步通道、policy slots、FPS/horizon 都由
`configs/input_schema.yaml` 声明。增删输入只改 YAML；转换器对缺失字段报错，禁止静默
padding/truncation。模型固定尺寸 padding 也必须在该 schema 中显式声明。

## 数据与模型

原始数据、处理后数据、模型权重和日志默认被 `.gitignore` 忽略。

迁移覆盖范围和剩余发布门槛见 `docs/migration-status.md`。

## 归属

SLaI Academy · EACV Center
