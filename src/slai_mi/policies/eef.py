"""Absolute end-effector pose representation and safety bounds."""

from __future__ import annotations

import numpy as np

COMMAND_DIM = 10


def _vector(value: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


def quaternion_wxyz_to_matrix(quaternion: object) -> np.ndarray:
    value = _vector(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = value / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_rotation6d(matrix: object) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("rotation matrix must be finite 3x3")
    return np.concatenate((value[:, 0], value[:, 1]))


def rotation6d_to_matrix(rotation6d: object) -> np.ndarray:
    value = _vector(rotation6d, 6, "rotation6d")
    first = value[:3]
    second = value[3:]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-6:
        raise ValueError("degenerate first rotation6d column")
    first /= first_norm
    second -= float(np.dot(first, second)) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-6:
        raise ValueError("degenerate second rotation6d column")
    second /= second_norm
    return np.stack((first, second, np.cross(first, second)), axis=1)


def pose_to_policy_state(position: object, quaternion_wxyz: object) -> np.ndarray:
    return np.concatenate(
        (
            _vector(position, 3, "position"),
            matrix_to_rotation6d(quaternion_wxyz_to_matrix(quaternion_wxyz)),
        )
    ).astype(np.float32)


def _matrix_to_rotation_vector(matrix: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
    )
    return axis * angle / (2.0 * np.sin(angle))


def _rotation_vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-8:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    # Stable enough after the bounded axis-angle update; normalize defensively.
    w = np.sqrt(max(0.0, 1.0 + np.trace(matrix))) / 2.0
    if w < 1e-7:
        values, vectors = np.linalg.eigh(matrix)
        axis = vectors[:, np.argmin(abs(values - 1.0))]
        result = np.concatenate(([0.0], axis))
    else:
        result = np.asarray(
            [
                w,
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )
        result[1:] /= 4.0 * w
    return result / np.linalg.norm(result)


def bound_absolute_command(
    current_position: object,
    current_quaternion_wxyz: object,
    command: object,
    *,
    max_translation_m: float = 0.02,
    max_rotation_rad: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Bound a 10D ``xyz + rotation6d + closure`` absolute command."""
    if max_translation_m <= 0 or max_rotation_rad <= 0:
        raise ValueError("motion bounds must be positive")
    position = _vector(current_position, 3, "current position")
    current_rotation = quaternion_wxyz_to_matrix(current_quaternion_wxyz)
    value = _vector(command, COMMAND_DIM, "command")
    delta = value[:3] - position
    distance = float(np.linalg.norm(delta))
    if distance > max_translation_m:
        delta *= max_translation_m / distance
    predicted = rotation6d_to_matrix(value[3:9])
    rotation_vector = _matrix_to_rotation_vector(predicted @ current_rotation.T)
    angle = float(np.linalg.norm(rotation_vector))
    if angle > max_rotation_rad:
        rotation_vector *= max_rotation_rad / angle
    target_rotation = _rotation_vector_to_matrix(rotation_vector) @ current_rotation
    return (
        (position + delta).astype(np.float32),
        _matrix_to_quaternion(target_rotation).astype(np.float32),
        float(np.clip(value[9], 0.0, 1.0)),
    )
