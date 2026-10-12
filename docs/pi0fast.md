# LeRobot PI0-Fast

The active training backend is LeRobot's PyTorch PI0 policy. The base model is
the official Hugging Face release `lerobot/pi0fast-base`, which is a
`model.safetensors` checkpoint (about 11.7 GiB). It is intentionally separate
from the legacy OpenPI JAX/Orbax PI0.5 checkout.

Download the model into the ignored model directory:

```bash
cd /home/user/shiyi/workspace/active/slai-manipulation
.venv-lerobot-v3/bin/huggingface-cli download lerobot/pi0fast-base \
  --local-dir data/models/pi0fast-base
```

PI0-Fast remains a deferred policy backend. It must consume the same
`configs/input_schema.yaml` camera, DoF, padding, FPS and horizon declaration as
PI0.5; there is deliberately no separately maintained static train YAML with a
fixed camera count or fixed 32-dimensional vectors.

The checkpoint cannot be loaded by OpenPI's JAX `CheckpointWeightLoader`.
Use the LeRobot `lerobot-train` entrypoint with the PI0 policy configuration
after the download completes.

The current LeRobot environment is pinned to `transformers>=5.4,<5.6`.
Using a newer Transformers release changes the PaliGemma module names and
causes false-looking missing-key warnings while loading this checkpoint.

Before enabling this backend, add a schema-driven generator equivalent to
`slai_mi.training.lerobot_pi05`, select and download a maintained checkpoint,
then run a one-step checkpoint/inference acceptance. The current PI0.5 command
must not be relabeled as PI0-Fast, and a handwritten LeRobot config is not an
accepted production entry point.
