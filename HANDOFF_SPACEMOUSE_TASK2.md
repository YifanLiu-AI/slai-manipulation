# 给 Agents 的用户视角交接说明：TASK2 9DoF 真实采集

## 我现在遇到的实际问题

我是现场用户。我现在不是要看一堆“服务 active”或“进程存在”的状态，我要的是：

```text
我移动 SpaceMouse，UR5 必须动；
我按住 Ctrl 移动 SpaceMouse，手腕两个自由度必须动；
我按 Menu+Fit 双击，必须能启动/退出采集；
我按 Fit，必须结束当前 Episode；
我按 3/4，夹爪必须关闭/打开；
采集出错后，界面必须自动恢复，不能卡死。
```

目前现场表现是：**正式采集界面启动后，SpaceMouse 完全没有反应。** UR5 不响应帽体移动，Menu+Fit 双击没有反应，无法正常开始或退出 Episode。之前待机阶段曾经有输入，但一进入正式采集就失效。不要再告诉我“worker 在运行所以应该没问题”，我需要你实际证明输入事件已经到达控制循环。

## 你必须先承认的事实

目前不能声称 TASK2 真实验收通过。之前的错误是把下面这些当成了成功证据：

```text
systemd active
SpaceMouse worker PID 存在
/dev/input/by-id 路径存在
```

这些只能说明进程/设备文件存在，**不能说明我移动 SpaceMouse 时输入已经到达 UR5 控制器和 Menu/Fit 状态机。**

请按用户实际体验验收：必须让我看到/确认以下数据链路真的通了：

```text
我移动/按键
→ Linux evdev 事件
→ EvdevReader
→ SpaceMouse worker
→ JSON socket
→ SpaceMouseProcess.state()
→ 主采集控制循环
→ UR5 / Menu-Fit / Ctrl 手腕动作
```

任何一层没有证据，都不能说“已经修好”。

## 给下一位 Agent 的任务

请接手排查并修复 `/home/user/shiyi/slai-manipulation` 的 TASK2 9DoF 真实采集链路。当前最大问题是：**正式采集服务看起来 active，但 SpaceMouse 输入没有被证明到达主采集控制循环；用户现场完全无法控制 UR5，也无法用 Menu+Fit 操作采集生命周期。不要把“worker 进程存在”当成输入链路验收通过。必须做输入数据级验收。**

## 当前已知运行状态（最近快照）

- `slai-wrist-collection.service`: `active (running)`
- 主进程 PID: `3773470`
- SpaceMouse worker PID: `3773621`
- worker 命令：
  ```text
  .venv-lerobot-v3/bin/python -m slai_mi.devices.spacemouse.workers.events
    --socket-fd 19 --deadzone 0.12 --stale-timeout 0.05 --rate-hz 250
    --backend evdev
    --event-device /dev/input/by-id/usb-3Dconnexion_SpaceMouse_Pro-event-mouse
  ```
- `slai-collection-ui.service`: inactive（正式采集运行时与采集服务冲突，这是设计行为）
- `spacenavd.service`: active，但正式采集使用 `evdev`，不应依赖 spacenavd
- 当前没有 8766 旧 UI（此前已清理）
- 当前正式采集服务没有日志显示 `Menu/Fit`、Episode start、SpaceMouse event 或 UR5 控制动作。

## 关键硬件事实

SpaceMouse 型号：
```text
3Dconnexion SpaceMouse Pro
VID:PID 046d:c62b
```
稳定 udev 路径：
```text
/dev/input/by-id/usb-3Dconnexion_SpaceMouse_Pro-event-mouse
```

但设备曾反复 USB 断开并重新枚举，内核日志明确出现：
```text
00:21:29 usb 1-3 disconnect
00:21:42 usb 1-6.1 SpaceMouse Pro reconnect
00:25:38 usb 1-6.1 disconnect
00:25:42 usb 1-6.1 reconnect
00:29:08 usb 1-6.1 disconnect
00:29:11 usb 1-4 reconnect
```
更严重的一次：
```text
00:31:23 usb 5-2.3 disconnect
00:31:52/53 Product: ATm32U4DFU
00:32:07 ATm32U4DFU disconnect
00:32:08 SpaceMouse Pro reconnect
```
这说明 USB/HID 设备曾短暂进入 Atmel DFU 引导模式，不是普通应用层按键错误。用户后来把 SpaceMouse 换到了更稳定的位置，但之后仍需连续观察内核日志，确认不再有 disconnect/DFU。

最近一次采集退出的明确根因是：
```text
SpaceMouse reader failed: [Errno 19] No such device
```
随后才出现：
```text
hardware supervisor failed closed: ur5 driver is not healthy
```
不要把后者当第一根因。

## 已做过的代码修改（未完成真实验收）

### 1. evdev 按键码映射
文件：`src/slai_mi/devices/spacemouse/evdev.py`

Linux 实际产生：
```text
BTN_0 = 256, BTN_1 = 257, ..., BTN_14 = 270, BTN_15 = 271, BTN_25 = 281
```
项目逻辑期望：
```text
Menu=0, Fit=1, 3=14, 4=15, Ctrl=25
```
已加入：
```python
logical_bnum = code - 256 if code >= 256 else code
```
并新增 `tests/test_spacemouse_evdev.py`。

### 2. evdev USB 掉线重连
文件：`src/slai_mi/devices/spacemouse/device.py`

已尝试在 `ENODEV/ENXIO/EIO/ENOENT` 时：
- 清零 motion；
- 清空 buttons；
- 关闭旧 fd；
- 通过 by-id 路径重新打开；
- 继续读事件。

但这段重连逻辑尚未进行真实 USB 掉线/重连验收；需审查实现是否正确，尤其是：
- `_ready` 与 `_failure` 的语义；
- worker socket 是否持续发送状态包；
- 重连后主进程是否能恢复；
- 不能在重连期间把旧 motion/button 留给 UR5。

### 3. UI 服务环境
文件：`/home/user/.config/systemd/user/slai-collection-ui.service`
已改为使用含 `pyrealsense2` 的：
```text
/home/user/shiyi/slai-manipulation/.venv-lerobot-v3/bin/slai-collection-ui
```
并改：
```ini
Restart=on-failure
```
此前 `.venv` 没有 `pyrealsense2`，导致温度保护后 UI 无限重启，无法回到待机界面。

### 4. 其他较大修改
当前工作树是大量未提交改动，不能随意 reset：
- TASK2 9DoF schema/strategy/task；
- Hiwonder gripper；
- dynamic wrist Home baseline；
- Menu+Fit 改为双击；
- Episode Menu→Fit 有效窗口；
- Fit 后立即停止 recorder，回零不进入数据；
- 每个 Episode 开始前 UR5+手腕+夹爪 HOME；
- 夹爪 HOME = `command(0)`，打开；
- emergency save；
- Ctrl+SpaceMouse X/Y 控制 FE/RU；
- ESP32 START/TELE 幂等恢复。

外部 Wrist2Wrist 项目通过软链接，必须保持不修改：
```text
third_party/02_Python_Client_CLI -> /home/user/haitao_files/robowrist/wrist_teleop_firmaware/02_Python_Client_CLI
```
外部仓库此前已恢复 clean。不要改外部 `wrist_output_v2.yaml`。本项目运行配置是：
```text
runtime/wrist_output_v2.yaml
```
`configs/hardware.yaml` 应指向本项目 runtime 配置。

## 最重要的验收要求

### A. 先停止/隔离当前未知状态的真实采集
当前采集服务可能仍在控制真实 UR5。下一位 Agent 在做输入诊断前必须先评估安全状态，必要时经用户确认后停止。不要并行启动第二个采集实例。

### B. 做不发机器人命令的 SpaceMouse 回显验收
必须证明完整链路：
```text
/dev/input/by-id/...event-mouse
  -> EvdevReader.feed/poll_event
  -> SpaceMouse._handle_event
  -> worker JSON state packet
  -> SpaceMouseProcess.state()
  -> 主采集控制对象收到 motion/buttons
```
不要只看：
```text
ps 里有 workers.events
systemd 是 active
```

建议新增一个独立的、不会打开 UR5/手腕/夹爪的测试工具或测试入口，持续输出：
```text
motion=[...]
buttons={...}
packet_count
last_event_time
worker_pid
```
用真实设备按 Menu、Fit、Ctrl、3、4、移动帽体逐项验证。

### C. 解析当前 evdev 事件是否真的产生
设备节点正常时，读取 `/dev/input/event5` 的内核 event；但不要在正式 worker 已打开时用竞争方式误导结论。最好使用同一个 worker 增加诊断计数/日志，或短暂停掉正式采集后做专用回显。

### D. 检查 worker socket 数据流
`SpaceMouseProcess.state()` 当前只是从 socket 收 JSON；如果 worker 活着但 socket 没有 state 包，主进程会一直得到旧的零 motion/空 buttons。需要明确记录：
- packet 是否持续到达；
- worker 是否发 error packet；
- 主进程是否调用 `state()`；
- `CachedSpaceMouse` 是否重复/错误消费状态。

### E. 检查采集状态机
当前正式采集启动后，日志只打印 plan/dashboard，没有明确打印：
```text
HOME complete
ready
Menu detected
Episode started
```
因此需要补充最小的状态日志或 dashboard event，不能靠猜。

## 本次接手后的输入验收工具与代码状态

新增无机器人回显入口：

```bash
.venv-lerobot-v3/bin/python -m slai_mi.apps.spacemouse_echo \
  --event-device /dev/input/by-id/usb-3Dconnexion_SpaceMouse_Pro-event-mouse \
  --rate-hz 20
```

它只启动 SpaceMouse worker，并逐行输出 `motion`、按下的 `buttons`、`packet_count`、
`last_event_time`、`event_count`、`reconnect_count`、`connected` 和 `worker_pid`。客户端
启动现在必须先收到至少一个 worker state 包；后续 socket 包停止也会失败闭环，不再把
默认零状态当成 ready。worker 重连期间 motion/buttons 会清零并发送 `connected=false`。

本次现场观察到：回显曾连续收到 state 包（`packet_count=3..343`，`connected=true`），
但当时没有物理输入（`event_count=0`）。之后内核在 `02:46:59` 再次记录
`usb 1-6.1: USB disconnect`，当前 by-id 节点不存在，因此 Menu/Fit/帽体的真实事件
尚未验收；不能启动正式真实采集。正式服务已保持停止。

### F. 注意双击语义
当前已改为：
```text
Menu+Fit 同时按下 -> 松开 -> 再同时按下 -> 松开
```
`RepeatedChordGesture(repeats=2, timeout_s=5.0)`。
必须确认 evdev 转换后确实得到按钮 `{0: True, 1: True}`，而不是只在 spnav 路径有效。

## 过去的真实错误清单

按时间/类型：
1. 固定 output-zero 不匹配：
   ```text
   saved output-zero profile disagrees with motor Home
   measured≈55°, saved=88.46°/64.93°
   ```
   已设计本项目每次成功 HOME 建立动态 baseline，但必须真实验证。
2. ESP32 主手：
   ```text
   Timed out waiting for {'START'}; last_line='OK TELE ...'
   ```
   已加入 START/TELE 容错，但需真实验证。
3. OpenRB 串口：
   ```text
   Could not configure port: (5, 'Input/output error')
   Cannot auto-detect OpenRB-150
   ```
4. UR5：
   ```text
   UR5 joint speed exceeds worker limit
   hardware supervisor failed closed: ur5 driver is not healthy
   ```
5. 夹爪：
   ```text
   gripper temperature 60C reached safety limit; torque disabled
   ```
   保护不能被简单关闭；但 emergency save/UI 恢复必须可靠。
6. SpaceMouse：
   ```text
   OSError: [Errno 19] No such device
   ```
   第一根因是 USB disconnect/重新枚举，之后旧 fd 失效。
7. UI：
   ```text
   ModuleNotFoundError: No module named 'pyrealsense2'
   ```
   已改 UI 到 `.venv-lerobot-v3`。

## 用户明确要求

- TASK2 = UR5 6DoF + OpenRB/ESP32 手腕 2DoF + Hiwonder 夹爪 1DoF；禁用 WujiHand。
- 夹爪：`0=打开`, `1=关闭`; 3/bnum14 关闭，4/bnum15 打开。
- 每次 Episode 开始前三者 HOME；夹爪 HOME 必须打开。
- 只有 Menu 到 Fit 之间的数据有效；Fit 后立即停止有效 recorder，回零不进 Episode。
- 正式采集无限 `--continuous`，不是最多 99。
- Menu+Fit 现在是双击，不是三击。
- 发生安全限制时先紧急保存已采集帧，再安全停机并恢复 UI。
- 不允许擅自修改外部软链接 Wrist2Wrist 项目。
- 真实运动必须以安全确认短语为门槛：`I_UNDERSTAND_REAL_ROBOT_MOTION`。

## 建议下一步

1. 不要继续宣称链路“已验收”。
2. 安全停止当前未知状态的 `slai-wrist-collection.service`（需要符合真实运动安全流程）。
3. 用无机器人输入回显测试确认 evdev 内核事件和 worker JSON socket。
4. 如果回显无事件，继续查 USB/设备本身；如果回显有事件，查 `SpaceMouseProcess.state()` 和采集主线程。
5. 修复后先跑离线测试，再做一次极短真实启动，只观察输入回显和状态日志。
6. 输入确认后，才做 UR5 极短运动、手腕、夹爪、Menu/Fit、数据保存联合验收。
7. 每一步都记录事实，不要用 `active`/worker PID 替代输入数据证据。
