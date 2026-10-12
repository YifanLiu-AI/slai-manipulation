from types import SimpleNamespace

import numpy as np
import pytest

from slai_mi.datasets.lerobot_v3.configured import ConfiguredDatasetContract
from slai_mi.devices.gripper.control import SpaceMouseGripperControl
from slai_mi.devices.spacemouse.buttons import Button
from slai_mi.input_schema import capture_vector_names, compose_capture_vector, load_input_schema


def test_gripper_state_and_action_are_binary() -> None:
    from slai_mi.devices.gripper import GripperState

    assert GripperState(100, 0.1, 1000, 30, 6.2, True, 1.0, 1).actual_q == (0.0,)
    assert GripperState(800, 0.8, 1000, 30, 6.2, True, 1.0, 1).actual_q == (1.0,)
    assert GripperState(100, 0.1, 1000, 30, 6.2, True, 1.0, 1).target_q == (1.0,)


def test_dual_gripper_capture_preserves_binary_and_raw_values() -> None:
    from slai_mi.devices.gripper import GripperState

    schema = load_input_schema(
        "configs/input_schemas/ur5e_wrist_gripper_dual_10dof_binary.yaml"
    )
    gripper = GripperState(437, 0.437, 1000, 30, 6.2, True, 1.0, 1)
    state = compose_capture_vector(
        schema,
        "state",
        {
            "ur5": SimpleNamespace(actual_q=np.zeros(6)),
            "wrist": SimpleNamespace(actual_q=np.zeros(2)),
            "gripper": gripper,
        },
    )
    action = compose_capture_vector(
        schema,
        "action",
        {
            "ur5": SimpleNamespace(target_tcp_speed=np.zeros(6)),
            "wrist": SimpleNamespace(target_q=np.zeros(2)),
            "gripper": gripper,
        },
    )
    assert state[-2:].tolist() == [0.0, 437.0]
    assert action[-2:].tolist() == [1.0, 1000.0]
    assert capture_vector_names(schema, "state")[-2:] == (
        "gripper.position",
        "gripper.raw_position",
    )


def load_schema_task2():
    from pathlib import Path

    import yaml

    return yaml.safe_load(Path("configs/tasks/task2.yaml").read_text())


def test_task2_schema_is_nine_dof_with_binary_gripper_channel() -> None:
    schema = load_input_schema("configs/input_schemas/ur5e_wrist_gripper_9dof.yaml")
    assert len(capture_vector_names(schema, "state")) == 9
    assert load_schema_task2().get("state_schema") == "task2_9dof"
    assert len(capture_vector_names(schema, "action")) == 9
    assert capture_vector_names(schema, "state")[-1] == "gripper.position"
    assert capture_vector_names(schema, "action")[-1] == "gripper.target_position"


def test_nine_dof_contract_rejects_nonbinary_gripper_values() -> None:
    schema = load_input_schema("configs/input_schemas/ur5e_wrist_gripper_9dof.yaml")
    contract = ConfiguredDatasetContract(schema)
    frame = {"task": "task2"}
    for key, feature in contract.features().items():
        dtype = "uint8" if feature["dtype"] == "video" else feature["dtype"]
        frame[key] = np.zeros(feature["shape"], dtype=dtype)
    # A valid rotation6d identity occupies the first two rotation columns.
    frame["observation.tcp_pose"][3:9] = [1, 0, 0, 0, 1, 0]
    contract.validate_frame(frame)

    frame["observation.state"][-1] = np.float32(0.5)
    with pytest.raises(ValueError, match="gripper state must be binary"):
        contract.validate_frame(frame)
    frame["observation.state"][-1] = np.float32(0.0)
    frame["action"][-1] = np.float32(-1.0)
    with pytest.raises(ValueError, match="gripper action must be binary"):
        contract.validate_frame(frame)


def test_spacemouse_gripper_button_edges() -> None:
    commands = []
    gripper = SimpleNamespace(command=commands.append)
    control = SpaceMouseGripperControl(gripper)
    assert control.update({int(Button.THREE): True}) == 1
    assert control.update({int(Button.THREE): True}) is None
    assert control.update({int(Button.THREE): False}) is None
    assert control.update({int(Button.FOUR): True}) == 0
    assert commands == [1, 0]


def test_open_command_unloads_after_stroke_without_changing_target() -> None:
    from slai_mi.devices.gripper.hiwonder import (
        LOAD_READ,
        LOAD_WRITE,
        POSITION_READ,
        TEMP_READ,
        VOLTAGE_READ,
        HiwonderGripper,
    )

    class Port:
        def __init__(self) -> None:
            self.writes = []

        def write(self, packet: bytes) -> None:
            self.writes.append(packet)

        def flush(self) -> None:
            return

    now = [10.0]
    port = Port()
    gripper = HiwonderGripper("unused", monotonic=lambda: now[0], open_unload_delay_s=1.5)
    gripper._serial = port
    registers = {POSITION_READ: 0, TEMP_READ: 40, VOLTAGE_READ: 6200, LOAD_READ: 1}
    gripper._read_register_unlocked = lambda command, size, signed=False: registers[command]

    gripper.command(0)
    now[0] += 1.4
    assert gripper.read_state().load_enabled is True
    assert not any(packet[4:6] == bytes((LOAD_WRITE, 0)) for packet in port.writes)

    now[0] += 0.1
    state = gripper.read_state()
    assert state.load_enabled is False
    assert state.target_position == 0
    assert any(packet[4:6] == bytes((LOAD_WRITE, 0)) for packet in port.writes)


def test_close_command_keeps_load_after_open_unload_delay() -> None:
    from slai_mi.devices.gripper.hiwonder import (
        LOAD_READ,
        LOAD_WRITE,
        POSITION_READ,
        TEMP_READ,
        VOLTAGE_READ,
        HiwonderGripper,
    )

    class Port:
        def __init__(self) -> None:
            self.writes = []

        def write(self, packet: bytes) -> None:
            self.writes.append(packet)

        def flush(self) -> None:
            return

    now = [10.0]
    port = Port()
    gripper = HiwonderGripper("unused", monotonic=lambda: now[0], open_unload_delay_s=1.5)
    gripper._serial = port
    registers = {POSITION_READ: 600, TEMP_READ: 40, VOLTAGE_READ: 6200, LOAD_READ: 1}
    gripper._read_register_unlocked = lambda command, size, signed=False: registers[command]

    gripper.command(1)
    now[0] += 5.0
    assert gripper.read_state().load_enabled is True
    assert not any(packet[4:6] == bytes((LOAD_WRITE, 0)) for packet in port.writes)


def test_gripper_filters_contact_bounce_but_allows_deliberate_same_target_retry() -> None:
    commands = []
    now = [10.0]
    control = SpaceMouseGripperControl(
        SimpleNamespace(command=commands.append),
        debounce_s=0.12,
        monotonic=lambda: now[0],
    )

    assert control.update({int(Button.THREE): True}) == 1
    control.update({})
    now[0] += 0.05
    assert control.update({int(Button.THREE): True}) is None
    control.update({})
    now[0] += 0.20
    assert control.update({int(Button.THREE): True}) == 1
    assert commands == [1, 1]
