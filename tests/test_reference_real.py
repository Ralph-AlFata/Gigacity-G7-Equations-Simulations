"""T011 checks on the real clips. Skipped unless `data/scene/` holds their scenes (`horizon-scene build`)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horizon_pipeline.raw.fetch import read_clip_list
from horizon_pipeline.reference.format import validate
from horizon_pipeline.reference.recorded import build_recorded

REPO = Path(__file__).resolve().parents[1]
SCENES = REPO / "data" / "scene"
CLIPS = read_clip_list(REPO / "clips_selected_train.txt")
CASES = [(c, s) for c in CLIPS for s in ("offline", "online")]

pytestmark = pytest.mark.skipif(
    not all((SCENES / c / s / "ego.parquet").is_file() for c, s in CASES), reason="scenes not built"
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_recorded_reference_lies_on_the_ego_track(case):
    scene_dir = SCENES / case[0] / case[1]
    ego = pd.read_parquet(scene_dir / "ego.parquet")
    ref, meta = build_recorded(scene_dir)
    validate(ref, meta)
    n = len(ego)
    assert len(ref) == n * (n + 1) // 2 and meta["speed_filled_ticks"] == [0]
    on = ref["tick"].to_numpy() + ref["step"].to_numpy()
    for c in ["x", "y", "z", "yaw"]:
        assert np.array_equal(ref[c].to_numpy(), ego[c].to_numpy()[on])
    assert np.array_equal(ref.loc[on > 0, "speed"].to_numpy(), ego["speed"].to_numpy()[on[on > 0]])
    assert (ref.loc[on == 0, "speed"] == ego["speed"].iloc[1]).all()
