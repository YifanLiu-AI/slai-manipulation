# Recovered Pose And SpaceMouse Records

This is an archive index, not an authorization to move hardware. The records were recovered on
2026-08-17 from local legacy and shared-task files. Runtime continues to use task YAML files;
every historical target requires a supervised, read-only comparison to current feedback before
motion is enabled.

## Canonical Historical Zeros

| Task | Captured | UR5 target | Wuji target | Evidence |
| --- | --- | --- | --- | --- |
| `block_into_box` | 2026-07-25T00:12:45+08:00 | first 6 values | final 20 values | `configs/recovered/legacy_28dof_zero_20260725.json`, legacy `record/28DOF零位`, shared task config |
| `remove_objects_from_box` | 2026-07-30T20:31:01+08:00 | `[3.089571952819824, -1.3231414000140589, -0.8841875235186976, -1.9886615912066858, 1.5532960891723633, 0.5235987755982988]` | `remove_objects_from_box_start.yaml[6:26]` | shared task config; legacy 2026-08-08 manifests and UR5 logs |
| `remove_objects_from_box-20mm` | 2026-07-30T20:31:01+08:00 | same as `remove_objects_from_box` | same as `remove_objects_from_box` | shared task config; legacy 2026-08-08 manifests and UR5 logs |

The `block_into_box` archived JSON is an exact value-level recovery. Its historical filename
states 28DOF, while its actual vector dimension is 26: `UR5[6] + WujiHand[20]`.

## Task Runtime Files

- `configs/tasks/block_into_box.yaml` -> `configs/poses/tasks/block_into_box_start.yaml`
- `configs/tasks/remove_objects_from_box.yaml` -> `configs/poses/tasks/remove_objects_from_box_start.yaml`
- `configs/tasks/remove_objects_from_box_20mm.yaml` -> `configs/poses/tasks/remove_objects_from_box_start.yaml`
- `configs/poses/home.yaml` is deliberately unconfigured and must never substitute a task start pose.

The active task pose YAML values match the recovered shared-task definitions. `block_into_box` is
currently marked configured, but the 2026-08-16 handoff records that its physical zero was not
accepted after the migration. Do not interpret the archive as that missing acceptance.

## Additional Wuji Pose Evidence

- `configs/poses/tasks/block_into_box_open.yaml`: task zero Wuji slice.
- `configs/poses/tasks/block_into_box_grasp.yaml`: legacy task state 1.
- `configs/poses/tasks/remove_objects_from_box_open.yaml`: task state 0.
- `configs/poses/tasks/remove_objects_from_box_grasp.yaml`: measured actual 2026-07-29T09:44:37Z.
- `configs/poses/tasks/remove_objects_from_box_aux_open.yaml`: measured actual 2026-07-29T09:43:26Z.
- `configs/poses/tasks/remove_objects_from_box_aux_grasp.yaml`: measured actual 2026-07-29T09:43:26Z.

Original one-vector YAML files remain at
`/home/user/shiyi/workspace/legacy/Robot_Teleoperation/record/wuji_hand_retargeting/poses/`.

## SpaceMouse Evidence

`configs/recovered/legacy_spacemouse_profiles.yaml` preserves the task profiles and the parameters
used by the old controller: 125 Hz, 0.080 m/s translation, 0.45 rad/s rotation, Ctrl 0.25 m/s and
0.60 rad/s, deadzone 0.12, acceleration 0.50. It also records the legacy semantics: Shift selects
TCP-local rotation and Button 4 requests the paired UR5/Wuji home.

The old 2026-08-08 manifests provide direct evidence that the actual launched tasks were
`remove_objects_from_box` and `remove_objects_from_box-20mm`, with those values. The runtime
summaries include an observed successful UR5 home completion, but do not prove that the current
migrated hardware path has the same behavior.

Relevant raw evidence:

- `/home/user/shiyi/workspace/legacy/Robot_Teleoperation/logs/record/runs/*/manifest.json`
- `/home/user/shiyi/workspace/legacy/Robot_Teleoperation/logs/record/runs/*/summary.txt`
- `/home/user/shiyi/workspace/legacy/Robot_Teleoperation/logs/record/history/*.json`

The `20260808_015502_129649` run contains many `rotation_tcp` samples, useful for replaying the
legacy Shift investigation. It is evidence of old input/status logging, not an approved axis map
for the new process-based controller.
