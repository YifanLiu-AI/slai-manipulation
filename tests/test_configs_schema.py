from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs"


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict), f"{path} must contain a YAML mapping"
    return value


def resolve_ref(source: Path, reference: str) -> Path:
    relative_path = reference.split("#", maxsplit=1)[0]
    return (source.parent / relative_path).resolve()


def test_all_yaml_configs_are_versioned_and_parseable() -> None:
    paths = sorted(CONFIGS.rglob("*.yaml"))
    assert paths
    for path in paths:
        config = load_yaml(path)
        assert config.get("schema_version") == 1, path


def test_input_schema_declares_consecutive_rotation6d_columns() -> None:
    schema = load_yaml(CONFIGS / "input_schema.yaml")
    tcp_pose = schema["capture"]["tcp_pose"]
    assert tcp_pose["dataset_representation"] == "xyz_rotation6d_columns_in_ur_base_frame"
    assert tcp_pose["names"] == ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21"]
    assert (
        schema["pi05"]["state"]["sources"][0]["transform"]
        == "ur_base_tcp_pose_to_rotation6d_columns"
    )


def test_dataset_joint_schemas_are_explicit_and_dimensionally_consistent() -> None:
    dataset = load_yaml(CONFIGS / "dataset.yaml")
    groups = dataset["joint_groups"]

    for name, group in groups.items():
        assert len(group["joint_names"]) == group["dimension"], name
        assert len(set(group["joint_names"])) == group["dimension"], name

    assert dataset["schemas"]["real_v1"]["dimension"] == 26
    assert dataset["schemas"]["real_v1"]["joint_groups"] == ["ur5", "wujihand"]
    assert dataset["schemas"]["simulation_v1"]["dimension"] == 28
    assert dataset["schemas"]["simulation_v1"]["joint_groups"] == [
        "ur5",
        "wrist",
        "wujihand",
    ]
    assert dataset["cross_schema_conversion"] == "disabled"


def test_pose_dimensions_and_safety_defaults() -> None:
    for path in sorted((CONFIGS / "poses").rglob("*.yaml")):
        pose = load_yaml(path)
        assert isinstance(pose["configured"], bool), path
        assert pose["units"] == "rad", path
        assert len(pose["joint_names"]) == len(pose["joint_positions"]), path
        assert len(set(pose["joint_names"])) == len(pose["joint_names"]), path
        assert "source" in pose["provenance"], path

        expected_dimension = 20 if pose["kind"] == "hand_preset" else None
        if pose["kind"] in {"task_start", "device_home"}:
            expected_dimension = 26
        assert len(pose["joint_positions"]) == expected_dimension, path

    home = load_yaml(CONFIGS / "poses" / "home.yaml")
    assert home["configured"] is False
    assert home["schema_ref"].endswith("#schemas.real_v1")
    assert "wrist_pitch_joint" not in home["joint_names"]
    assert "wrist_yaw_joint" not in home["joint_names"]


def test_commissioned_block_task_poses_are_enabled() -> None:
    pose_root = CONFIGS / "poses" / "tasks"
    for name in (
        "block_into_box_start.yaml",
        "block_into_box_open.yaml",
        "block_into_box_grasp.yaml",
    ):
        assert load_yaml(pose_root / name)["configured"] is True


def test_task_references_exist_and_match_dataset_schema() -> None:
    dataset = load_yaml(CONFIGS / "dataset.yaml")
    for path in sorted((CONFIGS / "tasks").glob("*.yaml")):
        task = load_yaml(path)
        references = [
            task["dataset_ref"],
            task["control_profile_ref"],
            task["start_pose_ref"],
            *task["hand_presets"].values(),
        ]
        for reference in references:
            assert resolve_ref(path, reference).is_file(), f"broken reference in {path}: {reference}"

        schema = dataset["schemas"][task["state_schema"]]
        start_pose = load_yaml(resolve_ref(path, task["start_pose_ref"]))
        assert len(start_pose["joint_positions"]) == schema["dimension"]


def test_hardware_template_contains_no_device_identity() -> None:
    hardware = load_yaml(CONFIGS / "hardware.yaml")
    if hardware["configured"] is False:
        assert hardware["ur5"]["enabled"] is False
        assert hardware["ur5"]["host"] is None
        assert hardware["wujihand"]["usb_serial"] is None
        assert hardware["wujihand"]["product_serial"] is None
        assert all(device["serial"] is None for device in hardware["cameras"]["devices"])
    else:
        # A checked-in commissioning override is valid only when every real identity
        # and the production adapter are present together.
        assert hardware["ur5"]["enabled"] is True
        assert str(hardware["ur5"]["host"]).strip()
        assert hardware["wujihand"]["enabled"] is True
        assert str(hardware["wujihand"]["usb_serial"]).strip()
        assert str(hardware["wujihand"]["product_serial"]).strip()
        assert all(str(device["serial"]).strip() for device in hardware["cameras"]["devices"])
        assert hardware.get("adapter_plugin") == "slai_mi.site_adapter:make_dependencies"
    assert hardware["iphone"] == {
        "enabled": False,
        "source": "pose_hub",
        "local_host": "127.0.0.1",
        "pose_port": 5005,
        "robot_state_port": 5006,
    }
    assert hardware["stereo_camera"]["left_serial"] is None
    assert hardware["stereo_camera"]["right_serial"] is None
