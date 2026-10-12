import numpy as np

from slai_mi.policies.action_chunk import ActionChunkPolicy
from slai_mi.policies.eef import bound_absolute_command, matrix_to_rotation6d, rotation6d_to_matrix
from slai_mi.policies.openpi import hand_delta_to_position, hand_position_to_delta


def test_rotation6d_round_trip_and_command_bounds() -> None:
    matrix = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(matrix_to_rotation6d(matrix), [0, 1, 0, 0, 0, 1])
    assert np.allclose(rotation6d_to_matrix(matrix_to_rotation6d(matrix)), matrix)
    identity = np.eye(3)
    command = np.concatenate(([1.0, 0.0, 0.0], matrix_to_rotation6d(identity), [2.0]))
    position, quaternion, closure = bound_absolute_command([0, 0, 0], [1, 0, 0, 0], command)
    assert np.isclose(np.linalg.norm(position), 0.02)
    assert np.allclose(quaternion, [1, 0, 0, 0])
    assert closure == 1.0


def test_action_chunk_and_hand_delta_round_trip() -> None:
    class Stub:
        calls = 0

        def infer(self, observation):
            self.calls += 1
            return {"actions": np.ones((3, 4), dtype=np.float32) * self.calls}

    stub = Stub()
    policy = ActionChunkPolicy(stub, action_dim=4, open_loop_horizon=2)
    assert policy.infer({}).tolist() == [1, 1, 1, 1]
    assert policy.infer({}).tolist() == [1, 1, 1, 1]
    assert policy.infer({}).tolist() == [2, 2, 2, 2]
    state = np.arange(8, dtype=np.float32)
    action = np.arange(8, dtype=np.float32)[None]
    delta = hand_position_to_delta(action, state, arm_dim=2)
    assert np.allclose(hand_delta_to_position(delta, state, action_dim=8, arm_dim=2), action)


def test_hand_delta_mapping_accepts_yaml_selected_noncontiguous_dofs() -> None:
    state = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    action = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    delta = hand_position_to_delta(
        action,
        state,
        action_indices=[0, 3],
        state_indices=[2, 0],
    )
    assert delta.tolist() == [[-29.0, 2.0, 3.0, -6.0]]
    restored = hand_delta_to_position(
        delta,
        state,
        action_dim=4,
        action_indices=[0, 3],
        state_indices=[2, 0],
    )
    assert np.allclose(restored, action)
