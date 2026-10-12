# Configuration

Configuration files are versioned YAML committed under `configs/`. Runtime code should load a
task file first and follow its references rather than combining unrelated files implicitly.

## Layout

- `dataset.yaml` defines joint groups and the real 26-dimensional and simulation 28-dimensional
  state contracts.
- `hardware.yaml` is a disabled template. Put IP addresses and device serial numbers in a local,
  ignored override or environment variables, never in the repository.
- `tasks/*.yaml` contains task instructions and references to the applicable state schema,
  controls, start pose, and named hand presets.
- `poses/home.yaml` is an unset 26-DoF real-device placeholder. Production Button 4 and
  lifecycle homing use the selected task's commissioned `poses/tasks/*_start.yaml` target.
- `controls/*.yaml` maps physical controls to semantic actions. Motion code, limits, and safety
  checks do not belong in YAML.

## Safety

Runtime rejects every motion pose unless it has `configured: true`. The commissioned
`block_into_box` task poses are enabled; generic and uncommissioned task poses remain disabled.
Before enabling another pose, verify the robot identity, joint order, limits, environment
clearance, and current hardware revision.

The real schema is 26-dimensional (`UR5[6] + Wujihand[20]`). The simulation schema is
28-dimensional (`UR5[6] + wrist[2] + Wujihand[20]`). Automatic padding or truncation is disabled;
any conversion must be an explicit, tested adapter with a documented physical interpretation.
