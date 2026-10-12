from __future__ import annotations

import socket

import pytest

from slai_mi.collection.home_control import (
    HomeCommandReceiver,
    StationControlBusyError,
    exclusive_station_control,
    request_home_if_active,
    station_control_is_busy,
)


def test_active_home_request_reaches_low_level_receiver(tmp_path) -> None:
    endpoint = tmp_path / "task2-home.sock"
    with HomeCommandReceiver(endpoint) as receiver:
        assert request_home_if_active(endpoint)
        assert receiver.poll()
        assert not receiver.poll()


def test_receiver_recovers_an_orphaned_socket(tmp_path) -> None:
    endpoint = tmp_path / "task2-home.sock"
    orphan = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    orphan.bind(str(endpoint))
    orphan.close()

    with HomeCommandReceiver(endpoint) as receiver:
        assert request_home_if_active(endpoint)
        assert receiver.poll()


def test_station_lock_reports_cross_owner_contention(tmp_path) -> None:
    lock = tmp_path / "station.lock"
    with exclusive_station_control(lock):
        assert station_control_is_busy(lock)
        with pytest.raises(StationControlBusyError), exclusive_station_control(lock):
            pass
    assert not station_control_is_busy(lock)
