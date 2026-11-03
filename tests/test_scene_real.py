"""T007 and T008 checks on the real clips. Skipped unless `data/raw/` holds them (fetch the `scene` preset).

The lidar checks decode every spin of both clips (~25 s each)."""

from pathlib import Path

import numpy as np
import pytest

from horizon_pipeline.raw.fetch import read_clip_list
from horizon_pipeline.scene.build import OBJECT_COLUMNS, build_scene
from horizon_pipeline.scene.checks import parked_spread
from horizon_pipeline.scene.lidar import POINTCLOUDS, SpinReader

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
CLIPS = read_clip_list(REPO / "clips_selected_train.txt")

pytestmark = pytest.mark.skipif(
    not all((RAW / c / "obstacle.offline.parquet").is_file() for c in CLIPS), reason="real data not fetched"
)


@pytest.fixture(scope="module", params=[(c, s) for c in CLIPS for s in ("offline", "online")], ids=str)
def scene(request):
    clip, source = request.param
    return build_scene(RAW / clip, source, lidar=False)


@pytest.fixture(scope="module", params=CLIPS, ids=str)
def lidar_scene(request):
    return build_scene(RAW / request.param, "offline")


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


def test_lidar_columns_are_filled(lidar_scene):
    o = lidar_scene.objects
    assert not o[["lidar_points", "observed", "in_scene", "lidar_spin", "lidar_dt_s"]].isna().any().any()
    assert (o["in_scene"] == o["observed"]).all()
    assert (o["observed"] == (o["lidar_points"] >= 5)).all()


def test_lidar_crossing_is_within_half_a_spin_of_the_tick(lidar_scene):
    """Except within 0.1 s of the first spin start or the last spin end, where a tick has spins
    on one side only."""
    o = lidar_scene.objects
    spins = SpinReader(RAW / lidar_scene.meta["clip_id"] / POINTCLOUDS)
    inner = o["t_s"].between((spins.starts[0] + 100_000) / 1e6, (spins.ends[-1] - 100_000) / 1e6)
    assert inner.mean() > 0.98
    assert o.loc[inner, "lidar_dt_s"].abs().max() <= 0.051


def test_near_objects_are_observed(lidar_scene):
    """Objects 4-20 m away (outside the ~3.2 m blind zone) are seen at almost every tick; the rest
    are hidden behind other objects (T008 Outcome)."""
    o = lidar_scene.objects
    near = o[o["distance"].between(4, 20)]
    assert near["observed"].mean() > 0.95
