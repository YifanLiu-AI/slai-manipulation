"""Pure OpenPI/PI0.5 observation and action transforms."""

from __future__ import annotations

import numpy as np


def parse_rgb(value: object) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {image.shape}")
    return np.ascontiguousarray(image, dtype=np.uint8)


def make_pi05_observation(
    *,
    state: object,
    prompt: str,
    images: dict[str, object] | None = None,
    primary_rgb: object | None = None,
    secondary_rgb: object | None = None,
    image_slots: tuple[str, ...] | None = None,
) -> dict[str, object]:
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.ndim != 1 or not np.isfinite(state_array).all():
        raise ValueError("state must be a finite vector")
    if images is None:
        if primary_rgb is None or secondary_rgb is None:
            raise ValueError("policy images must be provided")
        primary = parse_rgb(primary_rgb)
        secondary = parse_rgb(secondary_rgb)
        images = {"base_0_rgb": primary, "left_wrist_0_rgb": secondary}
    parsed = {key: parse_rgb(value) for key, value in images.items()}
    if not parsed or len({value.shape for value in parsed.values()}) != 1:
        raise ValueError("policy images must be non-empty and have matching shapes")
    template = next(iter(parsed.values()))
    model_keys = image_slots or ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    if not model_keys or len(set(model_keys)) != len(model_keys):
        raise ValueError("PI0.5 policy image slots must be non-empty and unique")
    unknown = sorted(set(parsed) - set(model_keys))
    if unknown:
        raise ValueError(f"unsupported PI0.5 policy camera slots: {unknown}")
    return {
        "state": state_array,
        "image": {key: parsed.get(key, np.zeros_like(template)) for key in model_keys},
        "image_mask": {key: np.bool_(key in parsed) for key in model_keys},
        "prompt": prompt,
    }


def hand_position_to_delta(
    actions: object,
    state: object,
    *,
    arm_dim: int = 6,
    action_indices: object | None = None,
    state_indices: object | None = None,
) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32).copy()
    current = np.asarray(state, dtype=np.float32)
    action_selection, state_selection = _delta_indices(
        result, current, arm_dim, action_indices, state_indices
    )
    result[..., action_selection] -= current[..., state_selection]
    return result


def hand_delta_to_position(
    actions: object,
    state: object,
    *,
    action_dim: int,
    arm_dim: int = 6,
    action_indices: object | None = None,
    state_indices: object | None = None,
) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32)[..., :action_dim].copy()
    current = np.asarray(state, dtype=np.float32)
    action_selection, state_selection = _delta_indices(
        result, current, arm_dim, action_indices, state_indices
    )
    result[..., action_selection] += current[..., state_selection]
    return result


def _delta_indices(
    actions: np.ndarray,
    state: np.ndarray,
    arm_dim: int,
    action_indices: object | None,
    state_indices: object | None,
) -> tuple[np.ndarray, np.ndarray]:
    if (action_indices is None) != (state_indices is None):
        raise ValueError("action_indices and state_indices must be configured together")
    if action_indices is None:
        hand_dim = actions.shape[-1] - arm_dim
        if hand_dim <= 0 or state.shape[-1] < hand_dim:
            raise ValueError("state/action dimensions do not contain a hand segment")
        return np.arange(arm_dim, actions.shape[-1]), np.arange(state.shape[-1] - hand_dim, state.shape[-1])
    action_selection = np.asarray(action_indices, dtype=np.int64)
    state_selection = np.asarray(state_indices, dtype=np.int64)
    if (
        action_selection.ndim != 1
        or state_selection.ndim != 1
        or len(action_selection) == 0
        or len(action_selection) != len(state_selection)
        or np.any(action_selection < 0)
        or np.any(action_selection >= actions.shape[-1])
        or np.any(state_selection < 0)
        or np.any(state_selection >= state.shape[-1])
    ):
        raise ValueError("configured delta indices do not match state/action dimensions")
    return action_selection, state_selection
