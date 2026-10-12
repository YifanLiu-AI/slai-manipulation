# slai-manipulation 交接说明

更新时间：2026-08-16  
项目目录：`/home/user/shiyi/workspace/active/slai-manipulation`  
行为基准：`/home/user/shiyi/workspace/legacy/Robot_Teleoperation`

## 先看这里

今天只把新项目的**采集链路**推进到了可运行状态。用户现场确认还有两个问题没有解决：

1. 任务零位仍不对，最近一次修改只有软件测试，没有完成修改后的真机验收。
2. SpaceMouse 按住 Shift 的实时 TCP 旋转仍不对。

因此，**零位重新现场确认前，不要直接开始正式采集**。`slai-collect-real` 启动后会先
自动归零，这一步本身就可能让 UR5 和 WujiHand 运动。

数据集中的 rotation-6D 列布局和 SpaceMouse Shift 旋转是两件事。前者已经在软件层改成
连续按列布局；后者是实时真机控制问题，仍未解决。不要因为 rotation-6D 单元测试通过，
就判断 Shift 旋转已经正常。

当前没有采集进程运行，`127.0.0.1:8765` 也没有服务监听。工作区包含今天尚未提交的
大量改动和新增文件，接手后不要执行 `git reset --hard`、`git checkout -- .` 或清理
未跟踪文件。

## 项目现在是什么状态

新项目负责 UR5 + WujiHand + SpaceMouse + 三台 RealSense 的真机遥操作和 LeRobot 数据
采集，并继续接 PI0.5 转换、训练和推理。老项目只作为只读行为基准，后续修复应把老版
实际运行方式逐项搬过来，不要另起一套控制语义。

采集入口：

```bash
cd /home/user/shiyi/workspace/active/slai-manipulation

# 只打印计划，不打开硬件
uv run slai-collect-real --episodes 1

# 真机命令。零位问题解决前不要直接执行。
uv run slai-collect-real \
  --episodes 1 \
  --execute-real \
  --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
```

正式命令默认启动并自动打开黑色采集监控台：

```text
http://127.0.0.1:8765
```

不要传 `--no-open-dashboard`，除非只是排查浏览器行为。采集按钮沿用老版：Menu 开始，
Fit 保存，Esc 丢弃。Button4 是 home。采集生命周期会在启动、保存和丢弃后请求归零。

正常运行参数不是临时 safe 档：

- UR5 控制循环：125 Hz
- 正常平移：0.080 m/s
- 正常旋转：0.45 rad/s
- Ctrl 档：0.25 m/s、0.60 rad/s
- 加速度：0.50
- Wuji 手控：30 Hz
- UR5 worker watchdog：0.25 s
- Wuji worker watchdog：0.5 s

这些 watchdog、限位、温度保护和 fail-closed 逻辑不能为了排障而关闭或放宽。

## 今天已经完成的部分

### 黑色采集监控台

新项目已经换成老版 `record/rgb_preview.py` 风格：黑底、上方三路真实相机、左下
SpaceMouse Pro 示意图、右下采集日志，下面显示 Wuji 温度。界面由采集进程自己启动，
不需要另起旧的 `collection_frontend`。

现场曾真实看到三路 640x480 图像，物理 Menu/Fit 边沿也成功触发过录制和保存。有效数据
`data/lerobot/block_into_box-20260816T182557` 包含 244 帧和三路视频。

### 速度、UR5 和 Wuji 控制骨架

- 恢复老版 125 Hz / 0.080 m/s 正常档以及 Ctrl 档。
- RTDE control 初始化移到 125 Hz 循环开始前，修掉“1 秒内没有第一条 UR5 命令”的误报。
- UR5/Wuji 分进程运行，跨设备 heartbeat、watchdog 和 supervisor fail-closed 保留。
- Wuji 手控恢复老版 30 Hz、Button3 单按闭合/双击后再按张开、F/R 仲裁、ROLL/T 和
  Button4 home 入口。
- MediaPipe 运行在独立 Python 3.11 环境，避免默认 Python 3.13 缺少旧 API。

这些说明软件结构和普通采集控制已对齐，不代表下面两个现场问题已经解决。

### 保存、归零状态机和温度

新项目已经补回老版 `record/main.py` 的采集生命周期：

- 启动先同步设备并请求归零；
- Fit 保存非空 episode，随后归零；
- 空 episode 的 Fit 按丢弃处理并自动重录；
- Esc 丢弃、归零、自动开始同一 episode 的下一次尝试；
- 30 秒归零超时后禁止录制；
- Menu/Fit/Esc 使用上升沿，归零过程中会消费按键，避免误触发。

Wuji 温度按 70/75/80 摄氏度分为 warning、critical 和 fail-closed，界面显示缓存温度。
状态机的软件测试通过，但它依赖“零位目标本身正确”，所以现在不能据此开始正式采集。

### 数据和训练管线

- LeRobot v3 原子保存、三相机视频、状态/action、同步统计已经跑通过。
- 最新检查过的源数据是 `data/lerobot/block_into_box-20260816T194416`：1 episode、
  67 帧、三路 H.264。
- PI0.5 转换、norm、LoRA 1 step 训练和 checkpoint 写盘曾真实跑通过。
- 当前 rotation-6D 软件契约是连续按列：

```text
[r00, r10, r20, r01, r11, r21]
```

采集、转换和训练代码已经统一到这个字段顺序。不过旧的 `*_rot6d_*` 转换产物和
`outputs/pi05/block_into_box_20260816T194416_rot6d_1step/checkpoints/000001` 是在最终列布局
修正前生成的，只能留作管线曾跑通的证据，**不能当作当前 schema 可部署模型**。
`configs/pi05.yaml` 已指向新的 `rot6d_columns` 路径，但这些新产物还没重新生成。解决现场
控制问题后，需要重新 convert、norm 和训练。

### 当前软件回归

最近一次完整回归：

```text
uv run pytest       -> 154 passed
uv run ruff check . -> All checks passed
git diff --check    -> clean
```

测试通过只能证明软件契约没有回归，不能替代零位和 Shift 旋转的真机验收。

## 未解决 1：零位仍不对

### 老版基准

老版从任务自己的 start pose 读取 canonical 26 DoF：UR5 用前 6 维，Wuji 用后 20 维。
Button4 和 recorder 发出的 home 请求都回到同一任务零位。Wuji 的 open/state 0 是另一种
手部语义，不能偷偷代替 zero。

主要参考：

- `legacy/Robot_Teleoperation/src/robot_teleoperation/collection/task_start_pose.py`
- `legacy/Robot_Teleoperation/src/robot_teleoperation/ur5/runtime.py`
- `legacy/Robot_Teleoperation/src/robot_teleoperation/wuji/runtime.py`
- `legacy/Robot_Teleoperation/record/main.py`
- `/home/user/shiyi/workspace/shared/task/block_into_box/config.yaml`

### 新版当前实现

`configs/tasks/block_into_box.yaml` 指向
`configs/poses/tasks/block_into_box_start.yaml`。`StationSession` 对它做 `configured: true`、
`real_v1`、26 维和 canonical joint order 校验，然后拆成：

```text
UR5 home  = task_home[0:6]
Wuji home = task_home[6:26]
```

`ManualWujiController` 已有独立 `home_target`，不再拿 open preset 顶替。Button4 和生命周期
home 都走同一个 session 目标，`home_status()` 也比较这组目标。通用
`configs/poses/home.yaml` 只是禁用的 26 维占位，不进入 production task home。

### 为什么仍算未解决

用户在现场明确确认零位还是不对。最近一次修改只证明 YAML 中的 26 个数与共享 legacy
配置逐元素一致，也通过了软件测试；修改后没有完成真机归零验收。数值“与某个文件一致”
不等于当前现场真正需要的零位正确，也不能证明启动时选中的任务、目标和最终关节运动都对。

目前没有一份修改后现场日志同时记录：实际选中的任务文件、发给 UR5/Wuji 的目标、归零前
实际关节、每周期命令、最终实际关节和用户认可结果。因此不能再写“已修复”。

### 下一步怎么查

1. 先确认老版用户正常运行时**实际传入的任务文件和启动参数**。不要只看默认值；检查旧
   launcher 最终给 UR5/Wuji 的 `task_start_file`，确认是否真是当前 shared config。
2. 在新项目打开硬件前打印一次只读诊断：任务配置绝对路径、pose provenance、完整 26 维
   target、UR5/Wuji 分片。确认没有选错 task，也没有加载通用 `home.yaml`。
3. 只读记录当前 `actual_q`，与目标分别计算 UR5 6 维和 Wuji 20 维误差。先让现场用户确认
   “预期零位”到底是哪一组，不要先发命令猜。
4. 对照老版 UR5/Wuji home 循环，逐项比较方向、速度限制、到位阈值、Button4 松开语义和
   recorder remote-home 保持语义。重点看新版 `_apply_legacy_ur5_command()`、
   `ManualWujiController.update()` 和 `ControlledSpaceMouse.home_status()`。
5. 用户确认目标后，再做有人值守的分设备极低风险验收：静止检查，先 UR5，再 Wuji，最后
   联动。每次记录 target、actual、误差和停止原因；异常立即停并只读复核。
6. 真机结果得到用户确认后，才允许采集启动时自动 home，并补一条基于真实目标的回归。

不要通过放宽到位阈值、延长 watchdog 或跳过初始 home 来绕过这个问题。

## 未解决 2：SpaceMouse 按住 Shift 旋转仍不对

### 老版基准

历史 legacy 链路是：spnav 原始六轴经过 `SPNAV_TO_Z_UP` 归一化；不按 Shift 时只取平移三轴；
按住 Shift 时只取旋转三轴，进入 TCP-local 模式；然后使用当前 UR TCP rotvec 把角速度转换
到 base frame，最后通过 125 Hz `speedL` 发送。新项目当前改为 Diffusion Policy 的
base-frame 增量方案，见下方“新版当前实现”。

主要参考：

- `legacy/Robot_Teleoperation/src/robot_teleoperation/spacemouse/device.py`
- `legacy/Robot_Teleoperation/src/robot_teleoperation/spacemouse/mapping.py`
- `legacy/Robot_Teleoperation/src/robot_teleoperation/ur5/geometry.py`
- `legacy/Robot_Teleoperation/src/robot_teleoperation/ur5/runtime.py`

### 新版当前实现

新版现在采用 Diffusion Policy 的真实 UR5 方案：Shift 按钮码为 24，输入六轴先统一经
`SPNAV_TO_Z_UP` 变换，按住 Shift 后 `cap[3:]` 直接作为 base-frame 角速度发送；不再使用
当前 TCP rotvec 做第二次坐标变换。

采集时实际路径不是老版单进程 runtime，而是：

```text
spnav/evdev
  -> SpaceMouseProcess
  -> ControlledSpaceMouse 线程
  -> _apply_legacy_ur5_command
  -> supervisor/UR5 worker
  -> RTDE speedL
```

### 已知差异和未知点

用户实测确认 Shift 旋转行为仍不对，但目前没有留下足够的现场 trace，不能武断说是某个轴
反了、按钮没读到，还是坐标系重复转换。纯 mapping 文件看起来相同，说明下一步不能继续
只改公式碰运气，应该检查新进程链路中的真实数据。

优先怀疑和验证这些边界：

- 物理 Shift 是否稳定上报为 button 24，是否与 cap event 同一时刻进入缓存；
- spnav 的 `rx/ry/rz` 到 normalized Z-up 三轴的实际符号和顺序；
- `ControlledSpaceMouse` 读到的 motion/buttons 是否陈旧或跨线程错拍；
- 当前 `tcp_pose[3:]` 是否是本周期的新反馈；
- TCP-local 角速度是否只做了一次 base-frame 变换；
- worker 收到的 target twist 与 RTDE `actual_tcp_speed[3:]` 是否方向一致。

### 下一步怎么查

1. 先不 arm 机器人，只采物理输入 trace：raw spnav 六轴、normalized 六轴、button 24、
   `MotionMode` 和 mapping 后 twist。分别推动/扭动单一物理轴，建立真实符号表。
2. 用同一组录制输入离线喂老版和新版纯函数，逐值比较。若相同，问题就在缓存、当前姿态、
   worker 或 RTDE 边界；若不同，先修输入映射。
3. 加结构化控制 trace：当前 TCP rotvec、local angular、base angular、worker request 和
   actual TCP angular speed。一次只测一个轴，避免三轴耦合掩盖符号问题。
4. 现场有人值守时，用低角速度短脉冲逐轴验证。先确认 Shift 按下只旋转不平移，再确认
   TCP x/y/z 三个正方向。异常立即松开、停止并只读复核。
5. 把用户认可的轴/符号/坐标系做成固定测试向量，再跑全量回归。

这个问题与 `src/slai_mi/rotation.py` 和 `configs/input_schema.yaml` 中的 rotation-6D 列顺序
没有直接关系。不要通过改训练数据字段顺序来修实时控制。

## 明日任务清单（2026-08-17）

按以下顺序继续，前两项是采集安全前置条件：

1. 解决零位问题，得到用户现场确认，并保存目标/actual/误差证据。
2. 解决 SpaceMouse Shift 实时 TCP 旋转，完成三个旋转轴的现场方向验收。
3. 完成 Wuji retarget 链路。现在只有独立 Python 3.11 worker、相机打开、无手返回 `None`
   的软件/现场证据；还没有完成“真实手画面 -> 20 维 target -> 限速轨迹 -> Wuji 实际跟随”
   的端到端验收。老版重点看：
   - `legacy/Robot_Teleoperation/record/wuji_hand_retargeting/`
   - `legacy/Robot_Teleoperation/src/robot_teleoperation/wuji/runtime.py`
   - 新版 `src/slai_mi/devices/wujihand/retarget_worker.py`
   - 新版 `src/slai_mi/site_adapter.py` 的 `WujiRetargetTargetProvider` / `WujiTargetController`
   - `configs/hardware.yaml:retarget_python/retarget_config`
   - `configs/input_schema.yaml` 的 `retarget_input`
4. 完成 iPhone 操作链路。现在协议、Pose Hub、桥接、iOS 工程和 5005/5006 端口约定存在，
   但没有完成 iPhone 真机姿态到 supervised UR5 命令的现场验收。老版/新版重点看：
   - `legacy/Robot_Teleoperation/ios/` 及其 iPhone/Record3D 相关实现
   - `src/slai_mi/devices/iphone/`
   - `src/slai_mi/apps/pose_hub.py`
   - `src/slai_mi/apps/pose_hub_bridge.py`
   - `src/slai_mi/ui/pose_hub/`
   - `clients/ios/IPhonePoseStreamer/`
   - `docs/iphone_pose.md`

Wuji retarget 和 iPhone 都必须继续走 supervisor、最终命令限位和 fail-closed；软件能启动、
无手返回 `None` 或手机页面能连上，都不能冒充真机控制链路已经打通。

## 关键文件索引

新项目采集和控制：

- `src/slai_mi/apps/collect_real.py`
- `src/slai_mi/runtime/real_workflows.py`
- `src/slai_mi/site_adapter.py`
- `src/slai_mi/runtime/hardware_supervisor.py`
- `src/slai_mi/collection/operator_control.py`
- `src/slai_mi/collection/vla_recorder.py`
- `src/slai_mi/collection/synchronization.py`

新项目设备和配置：

- `src/slai_mi/devices/ur5/`
- `src/slai_mi/devices/wujihand/`
- `src/slai_mi/devices/spacemouse/`
- `src/slai_mi/devices/cameras/`
- `configs/hardware.yaml`
- `configs/controls/spacemouse_standard.yaml`
- `configs/tasks/block_into_box.yaml`
- `configs/poses/tasks/block_into_box_start.yaml`
- `configs/input_schema.yaml`

界面和数据：

- `src/slai_mi/ui/collection_dashboard.py`
- `src/slai_mi/ui/collection_frontend.py`
- `src/slai_mi/ui/static/`
- `src/slai_mi/datasets/lerobot_v3/`
- `src/slai_mi/datasets/pi05.py`
- `src/slai_mi/rotation.py`
- `configs/pi05.yaml`

旧版唯一行为基准：

- `/home/user/shiyi/workspace/legacy/Robot_Teleoperation/record/main.py`
- `/home/user/shiyi/workspace/legacy/Robot_Teleoperation/record/rgb_preview.py`
- `/home/user/shiyi/workspace/legacy/Robot_Teleoperation/src/robot_teleoperation/`
- `/home/user/shiyi/workspace/legacy/Robot_Teleoperation/record/wuji_hand_retargeting/`

## 相关记录

- `/tmp/dashboard_restore_report.md`
- `/tmp/rebase_report.md`
- `/tmp/ur5_first_cmd_fix_report.md`
- `/tmp/record_lifecycle_restore_report.md`
- `/tmp/data_pipeline_rot_report.md`
- `/tmp/dk_rot_report.md`
- `/tmp/zero_pose_fix_report.md`
- `/tmp/slai_real_hardware_report.md`

最后提醒：目前最危险的误判是“代码看起来与老版一样，所以真机行为也一定一样”。用户已经
明确说明零位和 Shift 旋转不对。后续结论必须以用户现场确认和真实 telemetry 为准。
