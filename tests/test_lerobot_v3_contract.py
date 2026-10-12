import json

import numpy as np
import pytest

from slai_mi.datasets.lerobot_v3.contract import (
    CONTRACT_FILENAME,
    contract_manifest,
    validate_frame,
    write_contract_manifest,
)
from slai_mi.datasets.lerobot_v3.schema import (
    ACTION,
    OBSERVATION_TCP_POSE,
    SOURCE_DROPS,
    lerobot_features,
)


def canonical_frame() -> dict[str, object]:
    frame = {
        key: np.zeros(
            feature["shape"],
            dtype="uint8" if feature["dtype"] == "video" else feature["dtype"],
        )
        for key, feature in lerobot_features().items()
    }
    frame[OBSERVATION_TCP_POSE][3:] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    frame["task"] = "place the block in the box"
    return frame


def test_frame_and_manifest_follow_contract(tmp_path) -> None:
    validate_frame(canonical_frame())
    path = write_contract_manifest(tmp_path)
    assert path == tmp_path / "meta" / CONTRACT_FILENAME
    assert json.loads(path.read_text()) == contract_manifest()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.pop(ACTION), "keys differ"),
        (lambda frame: frame.__setitem__(ACTION, np.zeros(32, np.float32)), "shape"),
        (lambda frame: np.asarray(frame[SOURCE_DROPS]).__setitem__(0, -1), "nonnegative"),
        (
            lambda frame: np.asarray(frame[OBSERVATION_TCP_POSE]).__setitem__(
                slice(3, None), 0.0
            ),
            "degenerate first rotation6d column",
        ),
    ],
)
def test_corrupt_frames_are_rejected(mutation, message) -> None:
    frame = canonical_frame()
    mutation(frame)
    with pytest.raises(ValueError, match=message):
        validate_frame(frame)
