# SLaI Manipulation Intelligence

UR5 + WujiHand 操作智能项目：真机/仿真遥操作、LeRobot 数据采集、PI0.5 训练、推理和部署。

项目目录：`/home/user/shiyi/slai-manipulation`

## 0. 超短快捷命令（完整清单）

下面是 `shell/slai-shortcuts.sh` 当前定义的全部公开快捷命令。所有运行类命令都能从任意目录调用，并会先自动进入 `/home/user/shiyi/slai-manipulation`；`s` 会让当前终端直接进入该目录。

新终端自动生效。脚本更新后，已经打开的终端必须执行一次：

```bash
source ~/.bash_aliases
```

| 命令 | 准确用途 | 实际执行方式 | 会让真机运动 |
|---|---|---|---|
| `Home` | 在任意 TASK2 状态下抢占控制，使 UR5、双轴腕部和夹爪返回统一零位 | 底层 socket；无人持有硬件时直控 | **是** |
| `jax-home` | 让 TASK2 连续 JAX 链路先回统一零位后退出 | `Sensecore_H100/.venv` | **是** |
| `jax-slow` | 回零后以低速启动指定 2DoF/3DoF JAX checkpoint 推理 | OpenPI `.venv` | **是** |
| `jax-normal` | 回零后以正常速度启动指定 2DoF/3DoF JAX checkpoint 推理 | OpenPI `.venv` | **是** |
| `s` | 当前终端进入项目目录 | 无 | 否 |
| `sr` | 显示 UR5 + WujiHand 真机遥操作计划 | `uv run` 项目环境 | 否 |
| `srx` | 启动 UR5 + WujiHand 真机遥操作 | `uv run` 项目环境 | **是** |
| `sim` | 显示仿真遥操作计划 | `uv run` 项目环境 | 否 |
| `simx` | 启动仿真遥操作 | `uv run` 项目环境 | 否 |
| `sc` | 用 LeRobot v3 环境显示真机采集计划 | `.venv-lerobot-v3` | 否 |
| `scx` | 正式连续采集；采多少段由你决定，LOCK 或 `Ctrl-C` 才结束 | `.venv-lerobot-v3` | **是** |
| `scw` | 显示 UR5e + 双轴腕部 8DoF 采集计划 | `.venv-lerobot-v3` | 否 |
| `scwx` | 正式进行 UR5e + 双轴腕部 8DoF 采集 | `.venv-lerobot-v3` | **是** |
| `sc9` | 显示 TASK2：UR5e + 手腕 + 夹爪 9DoF 采集计划 | `.venv-lerobot-v3` | 否 |
| `sc9x` | 正式进行 TASK2 9DoF 采集（当前仍需完成验收） | `.venv-lerobot-v3` | **是** |
| `scc` | `scx` 的兼容连续采集命令 | `.venv-lerobot-v3` | **是** |
| `scs` | 显示仿真采集计划 | `uv run` 项目环境 | 否 |
| `scsx` | 启动仿真采集 | `uv run` 项目环境 | 否 |
| `su` | 启动离线采集监控台 | `uv run` 项目环境 | 否 |
| `sul` | 启动实时采集监控台 | `uv run` 项目环境 | 否 |
| `st` | 运行通用训练入口 | `uv run` 项目环境 | 否 |
| `si` | 运行 PI0.5 推理入口；是否执行由参数决定 | `uv run` 项目环境 | 否 |
| `se` | 运行数据集或策略评估 | `uv run` 项目环境 | 否 |
| `sp` | 运行 PI0.5 转换、统计、配置或训练流水线 | `uv run` 项目环境 | 否 |
| `sd` | 显示真机策略部署计划 | `uv run` 项目环境 | 否 |
| `sdx` | 执行真机策略部署 | `uv run` 项目环境 | **是** |
| `sph` | 显示 Pose Hub 启动计划 | `uv run --extra pose-hub` | 否 |
| `sphx` | 启动 Pose Hub | `uv run --extra pose-hub` | 否 |
| `spb` | 显示 Pose Hub Bridge 启动计划 | `uv run --extra pose-hub` | 否 |
| `spbx` | 启动 Pose Hub Bridge | `uv run --extra pose-hub` | 否 |
| `sv` | 运行 pytest；后面可接测试路径或 pytest 参数 | `uv run` 项目环境 | 否 |
| `sl` | 对项目运行 Ruff 检查 | `uv run` 项目环境 | 否 |
| `rec` | 启动 SpaceMouse 遥操作、Wuji retargeting 和命名姿态记录器 | `.venv` | **是** |
| `wuji-check` | 只开 4K USB 相机画面并检查 21 个手部关键点和 20 DoF retargeting | `.venv` | 否 |
| `wrist-park` | 将手腕移动到安全停放姿态 | `uv run` 项目环境 | **是** |
| `sm collect` | 兼容旧命令，行为与 `scc` 完全相同 | `.venv-lerobot-v3` | **是** |

不是所有快捷命令都走 `uv run`：普通应用入口使用 `uv run`；`Home`、`sc`、`scx`、`scc` 和 `sm collect` 固定直接使用 `/home/user/shiyi/slai-manipulation/.venv-lerobot-v3/bin/python`，这是为了避免再次误用当前激活的 `(base)`、Conda 或普通 `.venv` 而报 `LeRobot v3 is not installed`；`rec` 和 `wuji-check` 固定直接使用项目 `.venv`；`s` 只执行 `cd`。采集环境缺少 `lerobot` 或 `slai_mi` 时，会在连接和运动前报错退出。

命令名末尾的 `x` 表示直接执行对应任务，不再只是打印计划。表中标为“是”的命令会连接或驱动真机；`jax-home`、`jax-slow`、`jax-normal` 也会先执行回零并可能启动 JAX 推理。

当前 TASK2 rollout 阶段，`jax-slow`、`jax-normal` 以及 `P01` 到 `P100` 的所有
`Pxx-JAX2-01` / `Pxx-JAX3-01` episode 快捷命令统一使用 Sensecore_H100 中已完成离线 action
检查的 `local4090_aug_2k` checkpoint。切换入口集中在
`shell/slai-shortcuts.sh`；训练增强只在训练阶段启用，推理配置不会随机丢相机。

参数直接接在短命令后，例如：

```bash
sc --episodes 3
scx
scx --episodes 3  # 只有明确需要固定上限时才使用
scc               # 兼容写法，等同连续采集
sm collect
sp all --run-id r11 --execute --smoke
si --config outputs/pi05/r11/inference.yaml --execute
sd --config outputs/pi05/r11/inference.yaml
sv tests/test_apps_entrypoints.py
sl
```

当前真机工作方式通过策略组选择。下面的命令只查看配置，不连接设备：

```bash
uv run slai-strategies
uv run slai-strategies ur5e_wujihand_26dof_collection
```

已有四个组：

- `ur5e_wujihand_26dof_collection`：现有正式数采；真实 26DoF，文件为兼容旧训练链路仍含两个腕部零值。
- `ur5e_wujihand_retargeting`：SpaceMouse 控制 UR5e，4K 相机 retargeting 控制 WujiHand。
- `ur5e_wrist_8dof_teleop`：SpaceMouse 控制 UR5e，ESP32 主腕控制 OpenRB 从腕；不启动 WujiHand。
- `ur5e_wrist_8dof_collection`：同一组控制加三路 RGB 采集；state/action 都是实际 8DoF，不含 WujiHand 和补零腕部。

`slai-collect-real` 默认选择第一组，`slai-teleop-real` 和 `slai-record-pose` 默认选择第二组。也可以显式传入 `--strategy STRATEGY_ID`。腕部组合的只读计划和执行命令为：

```bash
uv run slai-teleop-real --strategy ur5e_wrist_8dof_teleop
uv run slai-teleop-real --strategy ur5e_wrist_8dof_teleop \
  --execute-real --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
```

腕部组合运行时，WujiHand 和采集相机被显式关闭。供应商目录 `third_party/02_Python_Client_CLI` 只读使用，腕部临时 episode 写入主项目的 `data/wrist-teleop`。

4090 的待机网页服务启用 `--spacemouse-collection-launch` 后，可以直接用实体
SpaceMouse 进入腕部正式数采：保持中央帽不动，同时按下 `Menu + Fit`，完全松开，
再重复，共完成 3 次；3 次需在 5 秒内完成。待机服务随后释放三台相机、SpaceMouse
和 8765 端口，再启动 `ur5e_wrist_8dof_collection` 连续采集。数采运行后，再按同样
方式完成 3 次，会结束数采、丢弃尚未保存的当前段，并自动恢复待机网页。
这条数采链路在启动和 Episode 收尾时的自动回零都会同时包含 UR5 和双轴腕部。

腕部正式采集使用：

```bash
scw                  # 只显示计划，不连接设备
scwx --episodes 1    # 采一段并退出
scwx                 # 连续采集，LOCK 或 Ctrl-C 结束
```

`scwx` 会先暂停独立的 `record.leai.me` 相机服务，让采集进程独占三台 RealSense 和 8765 监控端口；采集退出后自动恢复网页服务。同一路由器下始终使用 `http://192.168.1.102:8765/`，不需要开启代理。双轴腕部每次控制进程启动都会执行 `HOME_ALL`，等待机械 Home 完成，读取当次绝对编码器，并把该读数设置为本次运行的 FE/RU `0/0`；历史 YAML 零点只保留为供应商格式 fallback，不参与本次运行零点。诊断副本写入 Git 忽略的 `data/wrist_home_baseline.yaml`。腕部主手随后自动建立当前主手零点，不需要再按 ESP32 的 START 按钮。数据保存为 8D state（UR5 关节 6 + 腕部 FE/RU 实测 2）和 8D action（UR5 TCP 速度 6 + 腕部目标 2），所有腕部角度在主项目数据边界均为弧度。

TASK2 9DoF 采集只使用专用入口，避免误启动 8DoF：

```bash
sc9                  # 只显示 TASK2 9DoF 计划
sc9x --episodes 1    # 真实执行一段（当前 commissioning 会安全拒绝）
sc9x                 # 连续真实采集（当前 commissioning 会安全拒绝）
```

9DoF 为 UR5 6 + OpenRB 手腕 2 + Hiwonder 夹爪 1；SpaceMouse 3（bnum 14）关闭夹爪，4（bnum 15）打开夹爪。夹爪 state/action 主维度均为二值 0/1。完成真实验收后，才把策略从 `commissioning` 改为 `ready`。

TASK2 统一回零命令区分大小写，可从任意目录直接执行：

```bash
Home
```

采集控制器正在运行时，`Home` 通过 `/run/user/1000/slai-manipulation/task2-home.sock`
直接抢占控制循环，不会再开一套 RTDE 或串口连接；若当前 Episode 尚未完成，它会立即
停止并丢弃该段，再让 UR5 返回 `task2_start` 六轴关节位、腕部执行 `HOME_ALL` 返回本次
FE/RU `0/0`、夹爪打开为 `0`。没有活动控制器时，命令才独占硬件并执行同一套回零。
急停、保护停、驱动故障和过温保护始终有效，不会被 `Home` 绕过。

采完一段后的 PI0.5 一步冒烟训练命令为：

```bash
sp all --config configs/pi05_wrist_8dof.yaml \
  --source data/lerobot/实际数据集目录 \
  --run-id wrist8d_acceptance --execute --smoke
```

正式训练前先运行统一数据验收：

```bash
se dataset data/lerobot/实际数据集目录
```

`scx` 默认恢复旧版正式采集逻辑：启动后不限制 Episode 数量；Menu 开始一段；Fit 触发自动归零，但 Episode 在整个回零过程中继续录制，确认到达任务零位后才停止录制并保存，然后等待下一次 Menu；Esc 丢弃当前段并归零，归零后同样等待下一次 Menu，绝不主动开始录制；LOCK 或 `Ctrl-C` 才 finalize 数据集并退出。只有显式传入 `scx --episodes N` 时才会保存满 `N` 段后自动结束。归零失败，或归零期间按 LOCK/`Ctrl-C`，当前尚未提交的 Episode 不会保存。0 有效帧的 Episode 也按丢弃处理，归零后等待 Menu。

当前正式采集默认任务为 `task1`。数据文件中的 `observation.state` 和 `action` 是 28 维：真实 UR5 6 + WujiHand 20，最后两维 `wrist_pitch_joint`、`wrist_yaw_joint` 固定记录为 `0.0`。这两个零值只用于数据 schema，不参与真实机器人归零或运动控制。

独立检查 Wuji retargeting（只开 4K USB 相机窗口，不控制机器人）：

```bash
wuji-check
```

该命令只使用 `wujihand.retarget_camera_device` 指定的 HBVCAM 4K USB 相机，序列号为 `HB202400001`，绝不会占用三台 RealSense 采集相机。未连接或被占用时命令会直接报错退出，不会自动换相机。

把右手放入画面，窗口会绘制编号 `0-20` 的 21 个红色关键点和绿色骨架。终端出现 `LANDMARKS=21 RETARGET_DOF=20 PASS` 表示关键点与 20 维 Wuji retarget 输出均有效；按 `q` 退出。

### `rec` 命名姿态记录器

在任意目录运行：

```bash
rec
```

启动后直接在终端输入分组名称并回车：

- 输入 `箱内`：此后采集的姿态全部进入 `箱内` 分组。
- 再输入 `箱外`：立即新建或切换到 `箱外`，后续姿态全部进入该分组。
- 长按 **Menu** 0.8 秒：读取当前关节位置，自动命名为 `pose_001`、`pose_002`……
- 长按 **Fit** 0.8 秒：保存并退出。
- `Ctrl-C`：安全保存已有记录并退出。

第一个输入的分组名同时是 YAML 文件名，例如输入 `箱内` 会保存为 `data/pose-recordings/箱内.yaml`。之后输入的新名字仍作为该文件里的新分组。同名文件已存在时会加载并继续追加，不覆盖旧姿态。

每次捕获和新建分组都会立即写盘。`rec` 会开启 SpaceMouse UR5 遥操作和 Wuji 相机 retargeting，并通过原有 supervisor、watchdog、速度/关节/温度限制控制真机。终端只有在 retarget 相机实际初始化完成后才显示“Wuji retargeting 已启动”。

**运行 `rec` 会让机器人实际运动。** 开始前确认现场安全、急停可用、任务零位和控制配置正确。

当前真机实际控制仍是 `UR5 6 + WujiHand 20 = 26 DoF`；`task1` 数据另外保留两个固定零值 wrist 字段以匹配 28 维训练 schema。两个 wrist 值不是实测值，不能用于声称已经获得真实 28 DoF 传感器数据。

可选参数：

```bash
rec --hold-seconds 1.2
rec --output-root data/pose-recordings
rec --task configs/tasks/block_into_box.yaml
```

## 1. 每次开始先做

```bash
cd /home/user/shiyi/slai-manipulation
uv sync
```

常用环境检查：

```bash
uv run python --version       # Python 3.11+
uv run pytest                 # 完整测试
uv run ruff check .           # 代码检查
git diff --check              # 空白字符检查
```

所有 CLI 默认只打印 dry-run 计划，不连接设备、不控制机器人。先不加执行开关看计划，确认配置正确后再执行。

## 2. 完整命令总表

| 用途 | 命令 |
|---|---|
| 真机遥操作计划 | `uv run slai-teleop-real` |
| 真机遥操作执行 | `uv run slai-teleop-real --execute-real --confirm I_UNDERSTAND_REAL_ROBOT_MOTION` |
| 仿真遥操作计划 | `uv run slai-teleop-sim` |
| 启动仿真遥操作 | `uv run slai-teleop-sim --run` |
| 真机采集计划 | `uv run slai-collect-real --episodes 1` |
| 真机采集执行 | 见第 3 节，需确认短语 |
| 仿真采集计划 | `uv run slai-collect-sim --episodes 1` |
| 启动仿真采集 | `uv run slai-collect-sim --episodes 1 --run` |
| 采集监控台 | `uv run slai-collection-ui --host 127.0.0.1 --port 8080` |
| 通用训练计划 | `uv run slai-train --config configs/training.yaml` |
| PI0.5 一键流水线 | `uv run slai-pi05 all --run-id RUN_NAME --execute --smoke` |
| PI0.5 推理计划 | `uv run slai-mi-infer --config configs/inference.yaml` |
| PI0.5 推理执行 | `uv run slai-mi-infer --config configs/inference.yaml --execute` |
| 真机策略部署计划 | `uv run slai-deploy-real --config outputs/pi05/RUN_NAME/inference.yaml` |
| 数据集检查 | `uv run slai-evaluate dataset data/lerobot/DATASET` |
| Pose Hub 计划 | `uv run --extra pose-hub slai-pose-hub` |

`slai-infer` 是 `slai-mi-infer` 的兼容旧名称；新命令优先使用 `slai-mi-infer`。

## 3. 遥操作

### 真机 UR5 + WujiHand

先查看将启用的设备和任务：

```bash
uv run slai-teleop-real \
  --hardware-config configs/hardware.yaml \
  --task configs/tasks/block_into_box.yaml
```

确认现场零位、硬件地址和 adapter 后才允许运动：

```bash
uv run slai-teleop-real \
  --hardware-config configs/hardware.yaml \
  --task configs/tasks/block_into_box.yaml \
  --execute-real \
  --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
```

外部站点 adapter 可显式指定：

```bash
uv run slai-teleop-real --execute-real \
  --adapter-plugin my_adapter.module:factory \
  --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
```

### Isaac 仿真

```bash
uv run slai-teleop-sim --task configs/tasks/block_into_box.yaml
uv run slai-teleop-sim --task configs/tasks/block_into_box.yaml --run
uv run slai-teleop-sim --run --headless
```

`--run` 才会启动仿真；仿真环境需要已安装 Isaac Lab 的 `robot_teaching_isaaclab` 环境。

## 4. 数据采集

### 真机采集

只看计划（安全）：

```bash
uv run slai-collect-real \
  --task configs/tasks/remove_objects_from_box_20mm.yaml \
  --episodes 1
```

正式采集（会让 UR5/WujiHand 运动，启动时会自动归零）：

```bash
uv run slai-collect-real \
  --task configs/tasks/remove_objects_from_box_20mm.yaml \
  --dataset-config configs/dataset.yaml \
  --episodes 1 \
  --execute-real \
  --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
```

连续采集直到 Ctrl-C：

```bash
uv run slai-collect-real --continuous --execute-real \
  --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
```

监控台默认是 `http://127.0.0.1:8765`。排查浏览器时才使用：

```bash
uv run slai-collect-real --no-open-dashboard
uv run slai-collect-real --no-dashboard
```

按键语义：Menu 开始；Fit 开始归零且继续录制，确认到位后才停止并保存；Esc 丢弃并归零，随后等待 Menu；Button4 归零。任何丢弃都不会主动开始下一段。零位和 SpaceMouse Shift 旋转未完成现场复验前，不要正式采集。

如果采集服务因 USB 重连、预览 UI 占用相机或 UR5 worker fail-close 而停止，使用一键恢复命令：

```bash
/home/user/shiyi/slai-manipulation/deploy/restart_real_9d_collection.sh
```

脚本会检查 UR5 网络和 30001–30004 端口、夹爪/OpenRB/ESP32 串口、`spacenavd`，停止冲突的预览 UI，重启 `slai-wrist-collection.service`，等待自动归零，并确认三台 RealSense、UR5、腕部、夹爪和 SpaceMouse 都处于新鲜有效状态。

成功输出包含：

```text
UR5_PREFLIGHT_OK
COLLECTION_READY ... dataset=/home/user/shiyi/slai-manipulation/data/lerobot/real-9D/task-01-9dof-<timestamp>
```

脚本是幂等的：如果服务已经健康处于 `ready` 状态，只输出 `COLLECTION_ALREADY_READY`，不会重新归零或打断当前采集。真实机器人启动前仍必须确认工作区无人且急停可用。

### 仿真采集

```bash
uv run slai-collect-sim --task configs/tasks/block_into_box.yaml --episodes 10
uv run slai-collect-sim --episodes 10 --seed 42 --max-steps 900 --run
uv run slai-collect-sim --episodes 10 --run --headless
```

## 5. 通用训练入口

`slai-train` 是与具体训练框架解耦的入口。当前 `configs/training.yaml` 的 backend 为 `null`，因此默认只用于检查计划；配置生产 backend 后才执行。

```bash
uv run slai-train --config configs/training.yaml
uv run slai-train --config configs/training.yaml --dataset data/lerobot/DATASET
uv run slai-train --config configs/training.yaml --dataset data/lerobot/DATASET --execute
```

## 6. PI0.5 数据转换、训练和推理

逐步执行时，先 dry-run：

```bash
uv run slai-pi05 convert
uv run slai-pi05 norm
uv run slai-pi05 config --execute --smoke
uv run slai-pi05 train --execute --smoke
```

常用参数：`--config configs/pi05.yaml`、`--source DATA_ROOT`、`--steps N`、`--batch-size N`。

新数据的一键流程（convert → norm → config → train）：

```bash
uv run slai-pi05 all --run-id RUN_NAME --execute --smoke
```

正式训练去掉 `--smoke`：

```bash
uv run slai-pi05 all --run-id RUN_NAME --execute
```

H100/JAX 配置：

```bash
uv run slai-pi05 train --config configs/pi05_h100_jax.yaml --execute
```

H100 调度节点上的前台启动脚本使用该平台固定路径，并支持环境变量覆盖：

```bash
NUM_GPUS=4 STEPS=30000 BATCH_SIZE=4 bash deploy/h100_train.sh
NUM_GPUS=4 STEPS=30000 BATCH_SIZE=16 bash deploy/h100_train_jax.sh
```

将 wrist 8DoF 数据同步到在线 H100、检出干净代码 worktree，并在后台依次执行转换、
norm stats 和 JAX PI0.5 训练：

```bash
bash deploy/h100_wrist8d_pipeline.sh
```

可通过 `H100_HOST`、`REMOTE_PROJECT`、`NUM_GPUS`、`STEPS` 和 `BATCH_SIZE` 覆盖默认值。
脚本不会修改 H100 上保留有现场改动的原工作目录。

### ModelScope → CCI App → H100 中转训练

安装独立客户端并在终端交互登录；令牌不得写入脚本、命令参数或 Git：

```bash
uv sync --extra modelscope
export MODELSCOPE_ENDPOINT=https://modelscope.cn
export MODELSCOPE_NAMESPACE=你的魔搭用户名
uv run --extra modelscope ms-hub login
uv run --extra modelscope ms-hub whoami
```

4090 将一个已提交的 LeRobot v3 数据集创建为私有
`<namespace>/tmp-<task>-dataset` 并上传（默认命令只打印计划）：

```bash
uv run slai-modelscope dataset-upload \
  --task task2 --source data/lerobot/task2-wrist8d-20260824T232021
uv run --extra modelscope slai-modelscope dataset-upload \
  --task task2 --source data/lerobot/task2-wrist8d-20260824T232021 --execute
```

上传会在源目录生成 `.slai-modelscope-manifest.json`，记录每个文件的 SHA-256、大小、
文件数和总字节数。H100 下载后同时执行 `ms-hub cache verify` 和该 manifest 校验。

安装并初始化河套 SCO-CLI；Access Key 只交互写入 SCO profile，不得提交到仓库：

```bash
curl -sSfL https://sco.sensecore.cn/registry/install.sh | bash
export PATH="${SCO_HOME:-$HOME/.sco}/bin:$PATH"
sco init
sco doctor
```

4090 本地可把 SCO 凭据保存在 Git 忽略的 `.env`（权限必须为 `600`），变量名参考
`.env.example`。项目提交脚本会读取该文件但不会打印 Secret。SCO v2.0.1 的 headless
`init` 在部分终端中可能在认证成功后不写 profile；遇到该情况请在真实交互终端运行
一次 `sco init`，不要把 Secret 直接写进 Shell 命令历史。凭据曾出现在聊天或日志中时
应在流程跑通后立即轮换。

在共享 AFS 上创建私有环境文件（例如
`/mnt/afs/250010074/yifan/slai-manipulation/.secrets/modelscope.env`）：

```bash
MODELSCOPE_NAMESPACE=你的魔搭用户名
TASK=task2
CONFIG=configs/pi05_task2_9dof.yaml
STEPS=30000
BATCH_SIZE=1
```

该文件不进入 Git；ModelScope 登录仍在容器内通过 `ms-hub login` 建立。CCI App
`app-o56pqogh` 需要在网页中预先保存以下启动命令：

```bash
exec bash /mnt/afs/250010074/yifan/slai-manipulation/deploy/sensecore_start.sh --cci
```

`--cci` 是严格验收模式，强制 1 GPU、1 step、batch size 1。启动和停止 CCI：

```bash
deploy/modelscope_cci_submit.sh cci-start
EXECUTE=1 deploy/modelscope_cci_submit.sh cci-start

EXECUTE=1 deploy/modelscope_cci_submit.sh cci-stop
```

默认 SCO 参数来自当前河套 workspace：

```text
subscription = 0197ee17-b6eb-7846-b2b4-a77c5f509b92
workspace    = share-space
region       = cn-sh-01
CCI app      = app-o56pqogh
```

1 step 只验证：ModelScope 下载、数据完整性、视频解码、normalizer、模型加载、一次
前向/反向、优化器更新和 checkpoint 写入；它不用于判断策略质量。验收通过后，复制
`lyf-challenge-4` 的 4-GPU ACP 模板进行正式训练：

```bash
TASK=task2 deploy/modelscope_cci_submit.sh formal
TASK=task2 EXECUTE=1 deploy/modelscope_cci_submit.sh formal
```

ACP 的容器启动命令是：

```bash
exec bash /mnt/afs/250010074/yifan/slai-manipulation/deploy/sensecore_start.sh
```

没有 `--cci` 时入口强制 4 GPU，并使用环境文件中的正式 steps/batch size。正式 H100
worker 会把最终 checkpoint 上传到私有 `<namespace>/<task>-checkpoint`。
4090 下载并双重校验：

```bash
uv run --extra modelscope slai-modelscope checkpoint-download \
  --task task2 --target outputs/modelscope/task2-checkpoint \
  --inference-template outputs/pi05/task2_gripper_1000_v2/inference.yaml \
  --inference-output outputs/modelscope/task2-inference.yaml --execute
```

命令会在双重校验后识别下载的 `pretrained_model`，复制 inference 模板并只替换本地
checkpoint 路径，不覆盖原有可用配置。下载后仍应先离线实际加载 checkpoint，再运行
真机推理。任务恢复完成并确认不再有下载者后，才通过 ModelScope 网页删除临时仓库，
删除临时令牌，并在 4090/H100 分别执行 `ms-hub logout`。

推理：

```bash
uv run slai-mi-infer --config configs/inference_pi05_round10.yaml
uv run slai-mi-infer --config configs/inference_pi05_round10.yaml --execute
uv run slai-mi-infer --config outputs/pi05/RUN_NAME/inference.yaml --checkpoint PATH --execute
```

`--execute` 会调用配置中的模型环境；PI0.5 通常使用 `.venv-lerobot-v3/bin/python`，不要随意替换 Python 环境。

## 7. 真机策略部署

只生成/检查部署计划：

```bash
uv run slai-deploy-real --config outputs/pi05/RUN_NAME/inference.yaml
```

现场验收通过后才允许执行：

```bash
uv run slai-deploy-real \
  --config outputs/pi05/RUN_NAME/inference.yaml \
  --hardware-config configs/hardware.yaml \
  --task configs/tasks/remove_objects_from_box_20mm.yaml \
  --max-steps 100 \
  --execute-real \
  --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
```

部署入口要求已 commissioning 的任务零位，并且所有动作经过速度限制和 supervisor。未完成现场验收时不要执行真机策略。

## 8. 评估与数据检查

```bash
uv run slai-evaluate actions predictions.npz --output reports/actions.json
uv run slai-evaluate domain-gap real.npz sim.npz --output reports/domain.json
uv run slai-evaluate camera landmarks.json --output reports/camera.json
uv run slai-evaluate dataset data/lerobot/my_dataset
```

## 9. 监控台与 iPhone Pose Hub

只读采集监控台：

```bash
uv run slai-collection-ui --host 127.0.0.1 --port 8080
uv run slai-collection-ui --live --host 127.0.0.1 --port 8080
```

Pose Hub 服务（默认 dry-run，`--run` 才监听端口）：

```bash
uv run --extra pose-hub slai-pose-hub --host 127.0.0.1 --port 8706
uv run --extra pose-hub slai-pose-hub --host 0.0.0.0 --port 8706 --run
```

4090 本地桥接：

```bash
export POSE_HUB_BRIDGE_TOKEN='机器本地 token'
uv run --extra pose-hub slai-pose-hub-bridge --session SESSION_ID
uv run --extra pose-hub slai-pose-hub-bridge --session SESSION_ID --run
```

桥接端口约定：`5005` 姿态输出，`5006` 实测关节状态输入。

## 10. 配置文件速查

- `configs/hardware.yaml`：UR5、WujiHand、SpaceMouse、相机和 adapter。
- `configs/tasks/*.yaml`：任务 ID、起始姿态和任务参数。
- `configs/poses/*.yaml`：home/open/grasp 等命名姿态。
- `configs/dataset.yaml`：LeRobot v3 数据根目录和字段契约。
- `configs/input_schema.yaml`：相机角色、DoF 顺序、mask、FPS、horizon、PI0.5 padding。
- `configs/pi05.yaml`：PI0.5 转换、norm、训练输出路径。
- `configs/inference*.yaml`：checkpoint、backend 和部署参数。

## 11. 常用排错

```bash
uv run slai-teleop-real --help
uv run slai-collect-real --help
uv run slai-pi05 --help
uv run slai-mi-infer --help
uv run slai-evaluate --help
ss -ltnp | rg ':(5005|5006|8080|8765|8706)'
ps aux | rg 'slai-|pose-hub|uvicorn' | rg -v rg
```

遇到真机问题先保留日志和 `git diff`，不要执行 `git reset --hard`、`git checkout -- .` 或清理未跟踪文件。修改硬件配置后，先 dry-run，再做受控现场验收。

## 12. 项目结构

```text
configs/                 硬件、任务、数据集、模型配置
src/slai_mi/apps/        所有 slai-* CLI
src/slai_mi/devices/     UR5、WujiHand、相机、SpaceMouse、iPhone
src/slai_mi/collection/  episode、同步、写盘和遥操作采集
src/slai_mi/datasets/    LeRobot v3/PI0.5 转换与校验
src/slai_mi/policies/    推理和动作执行
src/slai_mi/simulation/  Isaac 仿真
tests/                   单元和集成测试
```
