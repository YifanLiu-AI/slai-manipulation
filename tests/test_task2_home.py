from __future__ import annotations

from slai_mi.apps import task2_home


def test_active_controller_dispatch_never_opens_hardware(monkeypatch, tmp_path) -> None:
    endpoint = tmp_path / "task2-home.sock"
    monkeypatch.setattr(task2_home, "request_home_if_active", lambda path: path == endpoint)
    monkeypatch.setattr(
        task2_home,
        "load_yaml",
        lambda _path: (_ for _ in ()).throw(AssertionError("hardware must stay unopened")),
    )

    result = task2_home.main(
        [
            "--execute-real",
            "--confirm",
            "I_UNDERSTAND_REAL_ROBOT_MOTION",
            "--active-socket",
            str(endpoint),
        ]
    )

    assert result == 0
