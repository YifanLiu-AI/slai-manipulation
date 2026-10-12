from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from slai_mi.apps import strategies
from slai_mi.runtime.strategy_profiles import (
    READY_STATUS,
    StrategyProfileError,
    available_strategy_profiles,
    load_strategy_profile,
)


def hardware(*names: str) -> dict[str, object]:
    return {name: {"enabled": True} for name in names}


def test_three_station_strategy_groups_are_explicit() -> None:
    profiles = {profile.id: profile for profile in available_strategy_profiles()}

    collection = profiles["ur5e_wujihand_26dof_collection"]
    assert collection.status == READY_STATUS
    assert collection.supported_apps == ("collect_real",)
    assert collection.physical_dof == 26
    assert collection.recorded_dof == 28
    assert collection.synthetic_channels == ("wrist_fe_ru_zero",)
    assert collection.disabled_hardware == ("wrist_sensor",)
    assert collection.control_modes == {
        "ur5e": "spacemouse_twist",
        "wujihand": "spacemouse_manual_presets",
    }
    configured = collection.configure_hardware(
        {"ur5": {"enabled": True}, "wrist_sensor": {"enabled": True, "port": "test"}}
    )
    assert configured["wrist_sensor"] == {"enabled": False, "port": "test"}

    retargeting = profiles["ur5e_wujihand_retargeting"]
    assert retargeting.status == READY_STATUS
    assert retargeting.supported_apps == ("teleop_real", "record_pose")
    assert retargeting.control_modes["wujihand"] == "camera_retargeting"

    wrist = profiles["ur5e_wrist_8dof_teleop"]
    assert wrist.status == READY_STATUS
    assert wrist.physical_dof == 8
    assert wrist.recorded_dof is None
    assert wrist.synthetic_channels == ()
    assert "wrist_master_esp32" in wrist.input_devices
    assert "wrist_slave_openrb" in wrist.output_devices

    wrist_collection = profiles["ur5e_wrist_8dof_collection"]
    assert wrist_collection.status == READY_STATUS
    assert wrist_collection.supported_apps == ("collect_real",)
    assert wrist_collection.physical_dof == 8
    assert wrist_collection.recorded_dof == 8
    assert wrist_collection.input_schema == "configs/input_schemas/ur5e_wrist_8dof.yaml"


def test_strategy_checks_app_hardware_and_task() -> None:
    profile = load_strategy_profile("ur5e_wrist_8dof_teleop")
    enabled = hardware("ur5", "wrist_sensor", "spacemouse")

    profile.validate_for("teleop_real", hardware=enabled, execute=True)
    with pytest.raises(StrategyProfileError, match="requires enabled hardware: wrist_sensor"):
        profile.validate_for("teleop_real", hardware=hardware("ur5", "spacemouse"))
    with pytest.raises(StrategyProfileError, match="does not support collect_real"):
        profile.validate_for("collect_real", hardware=enabled)

    collection = load_strategy_profile("ur5e_wujihand_26dof_collection")
    collection_hardware = hardware("ur5", "wujihand", "spacemouse", "cameras")
    with pytest.raises(StrategyProfileError, match="requires task state_schema"):
        collection.validate_for(
            "collect_real",
            hardware=collection_hardware,
            task={"state_schema": "simulation_v1"},
        )


def test_strategy_id_must_match_its_filename(tmp_path: Path) -> None:
    path = tmp_path / "wrong_name.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "strategy": {
                    "id": "different_name",
                    "label": "test",
                    "status": "ready",
                    "supported_apps": ["collect_real"],
                    "required_hardware": ["ur5"],
                    "inputs": ["input"],
                    "outputs": ["output"],
                    "control": {"ur5": "test"},
                    "physical_dof": 6,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StrategyProfileError, match="must match filename"):
        load_strategy_profile(path)


def test_strategy_inspection_cli_is_read_only_and_lists_all_groups(capsys) -> None:
    assert strategies.main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["app"] == "strategies"
    assert [item["strategy"] for item in output["strategies"]] == [
        "ur5e_wrist_8dof_collection",
        "ur5e_wrist_8dof_teleop",
        "ur5e_wrist_gripper_9dof_collection",
        "ur5e_wrist_gripper_dual_10dof_collection",
        "ur5e_wujihand_26dof_collection",
        "ur5e_wujihand_retargeting",
    ]

    assert strategies.main(["ur5e_wujihand_retargeting"]) == 0
    selected = json.loads(capsys.readouterr().out)
    assert selected["strategy"] == "ur5e_wujihand_retargeting"
    assert selected["physical_dof"] == 26
