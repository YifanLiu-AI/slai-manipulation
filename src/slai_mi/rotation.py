"""Rotation representations shared by real collection and policy datasets."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

UR_BASE_XYZ_ROTVEC = "xyz_rotvec_in_ur_base_frame"
UR_BASE_XYZ_ROTATION6D_COLUMNS = "xyz_rotation6d_columns_in_ur_base_frame"


def _finite_vector(value: object, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {size} finite values")
    return result


def _rotation_matrix(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrix must be finite 3x3")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1e-6
    ):
        raise ValueError("rotation matrix must be right-handed and orthonormal")
    return matrix


def rotation_vector_to_matrix(vector: object) -> np.ndarray:
    """Decode the UR axis-angle vector into its base-frame orientation matrix."""
    return Rotation.from_rotvec(_finite_vector(vector, 3, "rotation vector")).as_matrix()


def matrix_to_rotation_vector(matrix: object) -> np.ndarray:
    return Rotation.from_matrix(_rotation_matrix(matrix)).as_rotvec()


def quaternion_wxyz_to_matrix(quaternion: object) -> np.ndarray:
    wxyz = _finite_vector(quaternion, 4, "quaternion wxyz")
    norm = float(np.linalg.norm(wxyz))
    if norm < 1e-8:
        raise ValueError("quaternion norm is zero")
    wxyz /= norm
    return Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()


def matrix_to_quaternion_wxyz(matrix: object) -> np.ndarray:
    xyzw = Rotation.from_matrix(_rotation_matrix(matrix)).as_quat(canonical=True)
    return xyzw[[3, 0, 1, 2]]


def matrix_to_rotation6d_columns(matrix: object) -> np.ndarray:
    """Encode consecutive first and second rotation-matrix columns."""
    value = _rotation_matrix(matrix)
    return np.concatenate((value[:, 0], value[:, 1]))


def rotation6d_columns_to_matrix(rotation6d: object) -> np.ndarray:
    """Decode [r00,r10,r20,r01,r11,r21] with Gram-Schmidt."""
    values = _finite_vector(rotation6d, 6, "column-contiguous rotation6d")
    first = values[:3]
    second = values[3:]
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


def convert_tcp_pose(
    pose: object, *, source_representation: str, target_representation: str
) -> np.ndarray:
    """Convert one absolute TCP pose without changing its UR base frame."""
    if source_representation == UR_BASE_XYZ_ROTVEC:
        source = _finite_vector(pose, 6, "UR base TCP pose")
        position = source[:3]
        matrix = rotation_vector_to_matrix(source[3:])
    elif source_representation == UR_BASE_XYZ_ROTATION6D_COLUMNS:
        source = _finite_vector(pose, 9, "UR base column-contiguous rotation6d TCP pose")
        position = source[:3]
        matrix = rotation6d_columns_to_matrix(source[3:])
    else:
        raise ValueError(f"unsupported TCP pose representation: {source_representation}")

    if target_representation == UR_BASE_XYZ_ROTVEC:
        rotation = matrix_to_rotation_vector(matrix)
    elif target_representation == UR_BASE_XYZ_ROTATION6D_COLUMNS:
        rotation = matrix_to_rotation6d_columns(matrix)
    else:
        raise ValueError(f"unsupported TCP pose representation: {target_representation}")
    return np.concatenate((position, rotation)).astype(np.float32)


def normalize_ur_base_tcp_pose_to_rotation6d_columns(pose: object) -> np.ndarray:
    """Upgrade legacy xyz+rotvec data or validate current xyz+rotation6d data."""
    values = np.asarray(pose).reshape(-1)
    source = (
        UR_BASE_XYZ_ROTVEC
        if values.shape == (6,)
        else UR_BASE_XYZ_ROTATION6D_COLUMNS
        if values.shape == (9,)
        else None
    )
    if source is None:
        raise ValueError(f"UR base TCP pose must have 6 or 9 values, got {values.shape}")
    return convert_tcp_pose(
        values,
        source_representation=source,
        target_representation=UR_BASE_XYZ_ROTATION6D_COLUMNS,
    )


def rotate_pose_about_base_z(
    position: object, quaternion_wxyz: object, yaw_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """Express a pose after a left-multiplied rotation about the UR base +Z axis."""
    position_vector = _finite_vector(position, 3, "position")
    orientation = quaternion_wxyz_to_matrix(quaternion_wxyz)
    base_rotation = rotation_vector_to_matrix([0.0, 0.0, float(yaw_rad)])
    return (
        (base_rotation @ position_vector).astype(np.float32),
        matrix_to_quaternion_wxyz(base_rotation @ orientation).astype(np.float32),
    )
