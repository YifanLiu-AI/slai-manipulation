from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from slai_mi.rotation import (
    UR_BASE_XYZ_ROTATION6D_COLUMNS,
    UR_BASE_XYZ_ROTVEC,
    convert_tcp_pose,
    matrix_to_quaternion_wxyz,
    matrix_to_rotation6d_columns,
    matrix_to_rotation_vector,
    normalize_ur_base_tcp_pose_to_rotation6d_columns,
    quaternion_wxyz_to_matrix,
    rotate_pose_about_base_z,
    rotation6d_columns_to_matrix,
    rotation_vector_to_matrix,
)


def test_rotation_representations_round_trip() -> None:
    matrix = Rotation.from_euler("xyz", [0.4, -0.7, 1.2]).as_matrix()
    rotvec = matrix_to_rotation_vector(matrix)
    quaternion = matrix_to_quaternion_wxyz(matrix)
    rotation6d = matrix_to_rotation6d_columns(matrix)

    np.testing.assert_allclose(rotation_vector_to_matrix(rotvec), matrix, atol=1e-7)
    np.testing.assert_allclose(quaternion_wxyz_to_matrix(quaternion), matrix, atol=1e-7)
    np.testing.assert_allclose(rotation6d_columns_to_matrix(rotation6d), matrix, atol=1e-7)


def test_tcp_pose_conversion_preserves_ur_base_frame_orientation() -> None:
    source = np.asarray([-0.75, 0.12, 0.42, 1.8, -0.7, 0.4])
    encoded = convert_tcp_pose(
        source,
        source_representation=UR_BASE_XYZ_ROTVEC,
        target_representation=UR_BASE_XYZ_ROTATION6D_COLUMNS,
    )
    decoded = convert_tcp_pose(
        encoded,
        source_representation=UR_BASE_XYZ_ROTATION6D_COLUMNS,
        target_representation=UR_BASE_XYZ_ROTVEC,
    )

    np.testing.assert_allclose(encoded[:3], source[:3], atol=1e-7)
    np.testing.assert_allclose(
        rotation_vector_to_matrix(decoded[3:]),
        rotation_vector_to_matrix(source[3:]),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        normalize_ur_base_tcp_pose_to_rotation6d_columns(encoded), encoded, atol=1e-7
    )


def test_rotation6d_uses_consecutive_first_two_columns() -> None:
    matrix = Rotation.from_euler("xyz", [0.3, -0.5, 0.7]).as_matrix()
    encoded = matrix_to_rotation6d_columns(matrix)
    np.testing.assert_allclose(encoded, np.concatenate((matrix[:, 0], matrix[:, 1])))
    np.testing.assert_allclose(encoded[:3], matrix[:, 0])
    np.testing.assert_allclose(encoded[3:], matrix[:, 1])


def test_base_z_transform_left_multiplies_position_and_orientation() -> None:
    position, quaternion = rotate_pose_about_base_z(
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], np.pi / 2
    )
    np.testing.assert_allclose(position, [0.0, 1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(
        quaternion_wxyz_to_matrix(quaternion),
        Rotation.from_euler("z", np.pi / 2).as_matrix(),
        atol=1e-7,
    )
