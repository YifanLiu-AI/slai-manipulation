# Robot_Teleoperation migration status

This document audits capabilities from the legacy `Robot_Teleoperation` repository against
this repository. It is a migration ledger, not a claim that similarly named files are
behaviorally equivalent.

Status meanings:

- **migrated**: implementation and hardware-independent tests exist in the new package.
- **partial**: useful implementation exists, but an entry point, integration, dependency, or
  parity test is still missing.
- **deferred**: intentionally left for a later milestone.
- **excluded**: generated, machine-local, duplicated, or historical material that should not be
  migrated as source.

## Capability map

| Capability | Legacy source | New destination | Status | Remaining work |
| --- | --- | --- | --- | --- |
| SpaceMouse input, buttons, mapping, diagnostics | `src/robot_teleoperation/spacemouse/` | `src/slai_mi/devices/spacemouse/` | migrated | The workstation enumerated a SpaceMouse Pro and validated daemon binding, but round 9 received no physical motion event; operator input acceptance remains. |
| UR5 configuration, geometry, IK, runtime and zero pose | `src/robot_teleoperation/ur5/` | `src/slai_mi/devices/ur5/` | migrated | Round 9 real acceptance passed: mode 7/1, +Z 0.005 m/s commissioning steps, independent RTDE monitor, bounded displacement, and double-stop. |
| Wujihand client, filters, runtime and safety | `src/robot_teleoperation/wuji/` | `src/slai_mi/devices/wujihand/` | migrated | Round 10 added a fail-closed retarget worker boundary pinned to Python 3.11 / `mediapipe==0.10.21`; camera selection comes from `input_schema.yaml`. External-frame and real-RealSense no-hand tests returned `None`, with 20x2 limits loaded. |
| RealSense RGB/RGB-D acquisition | `collection/realsense_rgb*.py` | `devices/cameras/`, `collection/realsense_rgb*.py` | migrated | Round 9 opened all three configured cameras, read 10 real frame sets, verified 640x480 RGB, increasing sequences, and maximum host skew 15.567 ms. Calibration remains site-specific. |
| Stereo hand retargeting / MANO | `multiview_mano/`, `wrist_tracking/`, `realsense_only/` | `retargeting/`, `devices/wrist_sensor/`, `third_party/wuji-retargeting/` | partial | Pure geometry, robust extrinsic calibration and process boundaries exist; fitter worker and live device path need integration. |
| iPhone pose and Record3D | `iphone_pose/`, `ios/IPhonePoseStreamer/` | `devices/iphone/`, `clients/ios/IPhonePoseStreamer/`, `ui/pose_hub/` | partial | Protocol/client, Pose Hub bridge, measured-state handoff, coordinate calibration, relative TCP mapping, rate limiting, static-pose metrics and native client are present; real-hardware acceptance and supervised command integration remain. |
| Episode lifecycle and operator controls | `collection/episode.py`, `operator_control.py`, `home_control.py` | `collection/`, `runtime/real_workflows.py` | migrated | Round 9 committed one real 61-frame LeRobot v3 episode with Menu/Fit edge controls, three H.264 videos, Parquet, stats and atomic finalize. Physical button input was not available, so the acceptance used bounded scripted edges. |
| Multi-device synchronization and continuity | record pipeline and `collection/continuity.py` | `collection/`, `datasets/lerobot_v3/continuity.py` | migrated | Legacy ClockMapper (`[0.98,1.02]` slope gate), retained TimeSeries, configurable camera/state channels, interpolation/ZOH, outage gates and drop/sequence accounting are implemented. Round 10 replayed all 61 real rows; source drops were zero and 3 over-skew samples were correctly rejected. |
| LeRobot Dataset v3 schema, contract, writer, derive and merge | `collection/vla_v3*.py` | `datasets/lerobot_v3/` | migrated | Round 9 wrote and validated a real one-episode v3 dataset with 61 Parquet rows and three decodable videos. Keep the 26-DoF real and 28-DoF simulation schemas explicit. |
| Task start pose and combined zero pose | `collection/task_start_pose.py`, `combined_zero_pose.py`, task files | `configs/tasks/`, `configs/poses/`, `datasets/lerobot_v3/` | migrated | Replace unconfigured defaults only through machine-specific calibration; never copy an absolute legacy path into runtime config. |
| Session logs and manifests | `observability/session.py` | `observability/session.py` | migrated | Integrate with each runnable app and define retention policy. Runtime logs themselves are excluded. |
| Collection camera preview | `record/rgb_preview.py` | `ui/collection_dashboard.py`, `ui/collection_frontend.py`, `ui/static/` | migrated | The embedded port-8765 UI preserves the authoritative black `采集相机预览` layout: schema-driven RGB views, the CSS SpaceMouse Pro with all physical buttons, UR5/Wuji state, recording badge and chronological journal. It consumes the recorder's synchronized sources and exposes no command endpoint. |
| Isaac Sim articulation and drawer scene | `integrations/isaac/`, simulation apps | `simulation/`, `simulation/isaac/` | partial | Backend-neutral loops, lazy Isaac launch, robot scene plugin and NPZ writer are wired. Isaac Sim 5.1 headless loaded the articulation, ran two steps and wrote an NPZ episode on RTX 4090. Task-specific sensor/action validation remains. |
| OpenPI / PI0.5 pipeline | `integrations/openpi/`, conversion/training/evaluation tools | `policies/openpi.py`, `datasets/pi05*.py`, `training/`, `apps/pi05.py`, `evaluation/heldout.py` | migrated | Round 10 used the pinned OpenPI checkout for v2.1 norm stats and LeRobot/PyTorch for the available PI0.5 safetensors. Schema-driven conversion, LoRA smoke checkpoint, offline PEFT inference and the supervisor-routed real-policy action bridge all passed. |
| Other policies (XVLA, PI0-FAST) | bridge apps | `policies/` | deferred | Add only when a maintained model dependency and evaluation target are selected. |
| Training and inference entry points | legacy scripts/apps and `inference/` | `train.py`, `inference.py`, `training/`, `policies/` | migrated | `slai-pi05 config/train --execute --smoke` now launches the pinned LeRobot environment; `slai-mi-infer --execute` loads the PEFT checkpoint and returns schema-split UR5/Wuji actions. `slai-infer` remains a backwards-compatible alias. Random smoke-policy actions were deliberately not sent to hardware. |
| RL environment and RSL-RL configuration | `rl/`, train/eval apps | future `training/rl/` | deferred | Keep out of the base install until simulation training is a current milestone. |
| Sim-real evaluation | `simreal/`, analysis apps/reports | `evaluation/` | partial | Action/domain-gap metrics, camera landmark fitting and a runnable offline evaluation CLI exist. Dataset-scale reports and task-specific acceptance thresholds remain. |
| iOS client | `ios/IPhonePoseStreamer/` | `clients/ios/IPhonePoseStreamer/` | migrated | Build on a supported Xcode/iOS toolchain; it is intentionally outside the Python package. |
| Vendored retargeting library | `third_party/wuji-retargeting/` | `third_party/wuji-retargeting/` | migrated | Preserve its license and upstream provenance. Decide whether to pin as a submodule/package before modifying vendor code. |
| Robot URDF and meshes | `assets/` | `assets/robots/ur5_wrist_wujihand/` | partial | URDF and meshes are present and represent 28 movable joints. Redistribution licenses/provenance for UR5, wrist and Wujihand assets are not documented and must be resolved before public release. |
| Legacy guides and research notes | `docs/`, `research_*` | `docs/legacy/` | migrated | Treat as historical context, not current operating instructions. |
| Legacy tests | `tests/` | `tests/` | partial | App CLI, real supervisor, real synchronization, simulation runtime, dashboard provider, device parity helpers and train/inference plugin contracts have hardware-independent tests. Heavy SDK/model, iPhone build and full system acceptance tests remain. |

## Runnable workflow status

The five top-level workflows now have safe CLIs and hardware-independent runtime contracts. Their
status below distinguishes a tested orchestration path from a production deployment:

| Workflow | Entry point | Status |
| --- | --- | --- |
| Real teleoperation | `slai_mi.apps.teleop_real` | real commissioning and default retarget process verified; physical SpaceMouse cap motion remains pending |
| Simulation teleoperation | `slai_mi.apps.teleop_sim` | runnable plugin CLI; Isaac Sim 5.1 headless smoke passed |
| Real collection | `slai_mi.apps.collect_real` | real commissioning episode and default retarget process verified; physical SpaceMouse cap motion remains pending |
| Simulation collection | `slai_mi.apps.collect_sim` | Isaac headless two-step NPZ smoke passed; LeRobot writer plugin pending |
| Collection camera preview | `slai-collect-real --execute-real` (embedded) | legacy-aligned black RGB/SpaceMouse/journal UI at `127.0.0.1:8765`; recording remains gated by Menu/Fit/Esc |
| Standalone device monitor | `slai_mi.ui.collection_frontend --live` | read-only diagnostics only; it is not the recording workflow dashboard |

PI0.5 training and offline inference now have maintained execute backends. Policy output can be
split by the shared schema and routed only through the existing hardware supervisor; an untrained
smoke checkpoint is never an authorization to command the real robot.

## Round 10 closure evidence (2026-08-16)

- `configs/input_schema.yaml` is the single mapping for capture camera roles/fields, retarget input,
  state/action components and masks, policy slots/padding, synchronization channels, FPS and
  horizon. The converter, native training view, inference split, recorder schema, dashboard and
  MediaPipe worker consume it.
- The real 61-frame v3 episode converted to
  `data/training/pi05/block_into_box_v21_schema_round10`: 1 episode, 31 frames at 15 Hz, three
  224x224 videos, state[32], action[26]. OpenPI wrote `norm_stats.json` from all 31 frames.
- A fresh `slai-pi05 convert --execute` run also generated the schema-native v2.1 and v3 training
  views through one maintained entry point. The official v2.1-to-v3 converter operated on an
  isolated copy; the final v3 state/action are both [32] and include q01/q10/q50/q90/q99. Static
  PI0.5/PI0-Fast train YAMLs with fixed dimensions and duplicate generator scripts were removed.
- `slai-pi05 train --execute --smoke` loaded all 812 base weight keys and ran one LoRA update:
  loss 1.069, grad norm 0.164, 11.52M trainable / 4.15B total parameters, 9.92 GiB peak GPU memory.
  The 115 MiB local checkpoint is under `outputs/pi05/cli-smoke-round10/checkpoints/000001`.
- `slai-mi-infer --execute` loaded that checkpoint and one real frame with three
  `[1,3,224,224]` images plus state `[1,32]`; it produced model action `[1,32]`, physical action
  `[1,26]`, and schema-split UR5[6]/Wuji[20] components in 0.346 s after model load.
- The pinned MediaPipe worker opened the configured real camera and returned fail-closed `None`
  for three no-hand samples. The live dashboard opened all three cameras and emitted three JPEGs.
- Full lightweight regression: `126 passed`; Ruff: clean. Model training/inference and real camera
  checks were executed separately because they are intentionally heavy/hardware-dependent.

## Collection camera preview alignment (2026-08-16)

- Restored the black `采集相机预览` interface from the legacy `record/rgb_preview.py` as the
  collection-facing UI: three RGB views across the top, the full CSS SpaceMouse Pro at lower left,
  and the chronological recording journal at lower right. The new implementation remains an
  in-process read-only view fed by the exact `SynchronizedInputs` and recorder lifecycle used by
  `RealCollectionWorkflow`; it never opens a second set of camera handles.
- `slai-collect-real` enables the dashboard by default on `127.0.0.1:8765`. Dry-run prints the URL;
  real execution starts it before hardware resources and keeps recording gated by the existing
  Menu/Fit/Esc rising-edge state machine.
- A real ready-state run reported 3/3 configured cameras, 26 schema-declared DoF, valid SpaceMouse
  and robot feedback, zero source drops, and live camera skew/age values. All three camera JPEG
  endpoints returned 640x480 images; no Episode was started during UI acceptance.
- Camera roles/order/labels and dataset identities remain derived from `configs/input_schema.yaml`;
  changing the enabled camera list requires no UI code change. The compatibility API mirrors the
  legacy `/api/spacemouse`, `/api/cameras`, `/api/recording`, `/api/devices` and `/frame/*.jpg`
  contract while retaining the consolidated `/api/status` endpoint.
- Full regression after the alignment: `128 passed in 4.48s`; Ruff and `git diff --check` clean.

## Round 9 real evidence (2026-08-16)

- UR5 and Wuji real workers passed the ordered static -> UR5 step -> Wuji step gate. Watchdogs
  remain 0.25 s and 0.5 s; per-driver background heartbeats and unarmed collection setup were
  added without changing those thresholds.
- `teleop_real --execute-real` completed through the standard workflow using a bounded scripted
  input adapter: real UR5 delta Z 0.7289 mm / actual speed peak 0.00406 m/s and real Wuji
  progress 0.00961 rad with 0.000269 rad restore error.
- `collect_real --episodes 1 --execute-real` committed
  `data/lerobot/block_into_box-20260816T133658`: 1 episode, 61 frames, one Parquet data file,
  episode/tasks/stats metadata, and three 640x480 H.264 videos with 61 decodable frames each.
- A direct SpaceMouse service/evdev gate found the Pro device and active daemon but received no
  physical motion event. The physical input path therefore remains unaccepted until an operator
  moves the cap or the service is restarted with the required local privilege.
- The production camera-to-MediaPipe provider was verified in the existing Python 3.11
  `wuji_retargeting_camera` environment (three real frames, no hand detected, fail-closed
  `None`). It is not yet available in the default Python 3.13 environment.

## Safety and data invariants

The following are release gates, not optional cleanup:

1. Real hardware starts disabled and unconfigured. No repository default may contain an active
   robot address, calibration, zero pose, or command-enable flag.
2. UR5 workspace/joint checks and Wujihand joint, temperature and disconnect protections must be
   applied at the final command boundary, including policy-driven commands.
3. Home pose, task start pose and hand semantic poses are separate concepts. Each stored pose must
   state joint order, units, device identity or calibration provenance, and dimension.
4. Every state/action dimension and camera field must come from a versioned YAML schema. Different
   26-DoF/28-DoF or camera layouts need distinct declarations; values must never be silently
   inserted or removed.
5. Every recorded sample needs a documented clock domain and source timestamp. Episode commit must
   be atomic; discard and interrupted-write recovery must be tested.
6. Dataset/video/checkpoint paths are relative configuration or CLI inputs. Legacy absolute paths
   are documentation evidence only and must not become runtime defaults.

## Excluded generated and local material

Do not migrate `.cache/`, `.nv/`, `.pytest_cache/`, `.ruff_cache/`, `.venvs/`, `__pycache__/`,
`*.egg-info/`, `.swanlab/`, `swanlog/`, `logs/`, `outputs/`, `tmp/`, captured datasets,
checkpoints, or machine-generated reports as source. The new `.gitignore` covers the principal
runtime outputs; add tool-specific entries when a tool is adopted.

Large legacy `Datasets/`, `record/` outputs and analysis artifacts should be retained in managed
data storage with checksums and metadata, not copied into Git. Duplicate dashboard bundles and
historical reproduction scripts should only return if a current test or operating procedure needs
them.

## Recommended completion order

1. Implement production real-hardware adapter factories and verify final-command safety gates.
2. Implement the real synchronizer/dataset adapters and verify atomic LeRobot v3 episode commit.
3. Add a production live dashboard provider without granting motion control through the read-only API.
4. Add task-specific sensor/action validation on top of the passing Isaac headless smoke test.
5. Select and pin one maintained model backend, then run training, checkpoint and inference tests.
6. Run hardware acceptance, dataset compatibility, Isaac, iOS and license/provenance checks in
   their respective environments before declaring migration complete.
