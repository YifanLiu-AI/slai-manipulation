"""Read-only TASK2 9DoF state snapshot; never commands motion or torque."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from slai_mi.devices.gripper import create_gripper
from slai_mi.devices.ur5.process import UR5DriverProcess
from slai_mi.devices.wrist_sensor.teleop import VENDOR_ROOT

ROOT = Path(__file__).resolve().parents[3]

def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def _error(name: str, exc: BaseException) -> dict[str, Any]:
    return {"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

def _ur5(hardware: dict[str, Any]) -> dict[str, Any]:
    section = hardware["ur5"]
    driver = UR5DriverProcess(
        python=Path(section["driver_python"]), host=str(section["host"]),
        watchdog_s=float(section.get("driver_watchdog_s", .25)),
        max_linear_m_s=float(section.get("max_linear_m_s", .25)),
        max_angular_rad_s=float(section.get("max_angular_rad_s", .6)),
    )
    # DriverProcess context starts the isolated read-only worker. Do not arm
    # it and do not call prepare_control; read_state uses RTDE receive only.
    with driver:
        state = driver.read_state()
    return {"name": "ur5", "ok": True, "joints_rad": state.get("joints")}

def _wrist(hardware: dict[str, Any]) -> dict[str, Any]:
    import sys as _sys
    vendor = VENDOR_ROOT
    if str(vendor) not in _sys.path:
        _sys.path.insert(0, str(vendor))
    from openrb_bridge.output_v2 import load_output_v2_config
    from openrb_bridge.pc_client.openrb_client import OpenRBClient
    from openrb_bridge.serial_ports import resolve_openrb_port
    section = hardware["wrist_sensor"]
    config_path = ROOT / str(section["config"])
    config = load_output_v2_config(config_path)
    port = resolve_openrb_port(str(section.get("openrb_port", "auto")))
    client = OpenRBClient(port, int(section.get("baud", 115200)), timeout=.6).connect()
    try:
        fields = client.read_wrist_state(retries=3).fields
    finally:
        client.close()
    enc0 = int(fields["enc0_deg"]) / 100.0
    enc1 = int(fields["enc1_deg"]) / 100.0
    baseline_path = ROOT / "data" / "wrist_home_baseline.yaml"
    baseline = _load(baseline_path) if baseline_path.is_file() else {
        "enc0_abs_deg": config.output_zero.enc0_abs_deg,
        "enc1_abs_deg": config.output_zero.enc1_abs_deg,
    }
    return {"name": "wrist", "ok": True, "encoder_abs_deg": [enc0, enc1],
            "relative_home_deg": [enc0 - float(baseline["enc0_abs_deg"]),
                                   enc1 - float(baseline["enc1_abs_deg"])]}

def _gripper(hardware: dict[str, Any]) -> dict[str, Any]:
    section = hardware["gripper"]
    gripper = create_gripper(section)
    with gripper:
        state = gripper.read_state(enforce_safety=False)
    return {
        "name": "gripper",
        "ok": True,
        "position": state.position,
        "normalized_position": state.normalized_position,
        "target_position": state.target_position,
        "normalized_target_position": state.normalized_target_position,
        "temperature_c": state.temperature_c,
        "voltage_v": state.voltage_v,
        "load_enabled": state.load_enabled,
    }

def _print_chinese(results: list[dict[str, Any]], ok: bool) -> None:
    print("TASK2 当前状态（只读，不回零，不发送运动指令）")
    print(f"整体状态：{'正常' if ok else '部分读取失败'}")
    for item in results:
        name = item["name"]
        if not item["ok"]:
            label = {"ur5": "UR5", "wrist": "腕部", "gripper": "夹爪"}.get(name, name)
            print(f"{label}：读取失败 — {item['error']}")
            continue
        if name == "ur5":
            values = item.get("joints_rad", [])
            print("UR5：" + " ".join(f"J{i + 1}={float(v):+.6f} rad" for i, v in enumerate(values)))
        elif name == "wrist":
            absolute = item["encoder_abs_deg"]
            relative = item["relative_home_deg"]
            print(
                "腕部："
                f"FE绝对={float(absolute[0]):+.2f}° "
                f"RU绝对={float(absolute[1]):+.2f}° "
                f"FE相对Home={float(relative[0]):+.2f}° "
                f"RU相对Home={float(relative[1]):+.2f}°"
            )
        elif name == "gripper":
            print(
                "夹爪："
                f"实际位置={float(item['normalized_position']):.4f}"
                f"({int(item['position'])} tick) "
                f"目标位置={float(item['normalized_target_position']):.4f}"
                f"({int(item['target_position'])} tick) "
                f"温度={int(item['temperature_c'])}°C "
                f"电压={float(item['voltage_v']):.3f}V "
                f"扭矩={'开启' if item['load_enabled'] else '关闭'}"
            )

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读打印 TASK2 9DoF 当前状态")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    args = parser.parse_args(argv)
    hardware = _load(ROOT / "configs" / "hardware.yaml")
    results = []
    for name, fn in (("ur5", _ur5), ("wrist", _wrist), ("gripper", _gripper)):
        try:
            results.append(fn(hardware))
        except Exception as exc:  # noqa: BLE001 - report each independent device failure.
            results.append(_error(name, exc))
    ok = all(bool(item["ok"]) for item in results)
    if args.json:
        print(json.dumps({"ok": ok, "dof": 9, "state": results}, ensure_ascii=False, indent=2))
    else:
        _print_chinese(results, ok)
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
