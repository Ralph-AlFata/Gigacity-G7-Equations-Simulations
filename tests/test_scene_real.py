"""T007 checks on the real clips. Skipped unless `data/raw/` holds them (fetch the `scene` preset)."""

from pathlib import Path

import numpy as np
import pytest

from horizon_pipeline.raw.fetch import read_clip_list
from horizon_pipeline.scene.build import OBJECT_COLUMNS, build_scene
from horizon_pipeline.scene.checks import parked_spread

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
CLIPS = read_clip_list(REPO / "clips_selected_train.txt")

pytestmark = pytest.mark.skipif(
    not all((RAW / c / "obstacle.offline.parquet").is_file() for c in CLIPS), reason="real data not fetched"
)


@pytest.fixture(scope="module", params=[(c, s) for c in CLIPS for s in ("offline", "online")], ids=str)
def scene(request):
    clip, source = request.param
    return build_scene(RAW / clip, source)


def test_grid_and_columns(scene):
    e, o = scene.ego, scene.objects
    assert e["tick"].tolist() == list(range(scene.meta["n_ticks"]))
    assert scene.meta["t0_us"] == 0 and e["t_s"].iloc[-1] >= 19.9
    assert list(o.columns) == OBJECT_COLUMNS
    assert o["tick"].between(0, scene.meta["n_ticks"] - 1).all()
    assert not o.duplicated(["tick", "track_id"]).any()
    assert not e.isna().any().any()
    assert scene.report["unknown_classes"] == []


def test_parked_objects_stay_still(scene):
    """T004's floor for parked sightings is a median RMS of 0.16-0.21 m with offline poses."""
    spread = parked_spread(scene.objects)
    assert len(spread) > 20
    limit = 0.25 if scene.meta["ego_source"] == "offline" else 0.30
    assert spread.median() < limit
    assert spread.quantile(0.95) < 0.5


def test_objects_move_smoothly(scene):
    """No tick-to-tick jump an object could not make: 3 m in 0.1 s is 108 km/h."""
    o = scene.objects.sort_values(["track_id", "tick"])
    same = (o["track_id"].to_numpy()[1:] == o["track_id"].to_numpy()[:-1]) & (np.diff(o["tick"]) == 1)
    step = np.linalg.norm(np.diff(o[["x", "y"]].to_numpy(), axis=0), axis=1)[same]
    assert step.max() < 3.0
