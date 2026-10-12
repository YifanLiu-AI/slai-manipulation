from slai_mi.collection.operator_control import EpisodeAction, SpaceMouseEpisodeControls


def test_episode_buttons_are_rising_edge_triggered() -> None:
    controls = SpaceMouseEpisodeControls()
    assert controls.update({0: True}) is EpisodeAction.START
    assert controls.update({0: True}) is None
    assert controls.update({}) is None
    assert controls.update({1: True}) is EpisodeAction.SAVE


def test_fit_requests_home_while_idle() -> None:
    controls = SpaceMouseEpisodeControls()
    assert controls.update({1: True}) is EpisodeAction.HOME
    assert not controls.recording
    assert controls.update({1: True}) is None


def test_escape_discards_an_active_episode() -> None:
    controls = SpaceMouseEpisodeControls()
    assert controls.update({0: True}) is EpisodeAction.START
    assert controls.update({}) is None
    assert controls.update({22: True}) is EpisodeAction.DISCARD
    assert not controls.recording


def test_rotation_lock_finalizes_on_rising_edge() -> None:
    controls = SpaceMouseEpisodeControls()
    assert controls.update({26: True}) is EpisodeAction.FINALIZE
    assert controls.update({26: True}) is None
    assert controls.update({}) is None


def test_menu_fit_chord_is_reserved_for_collection_service_control() -> None:
    controls = SpaceMouseEpisodeControls()
    assert controls.update({0: True, 1: True}) is None
    assert not controls.recording
