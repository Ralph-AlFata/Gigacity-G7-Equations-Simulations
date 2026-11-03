"""T002 checks against the real fetched data. Skipped unless `data/raw/` and the old copies exist.

Run `horizon-fetch --clips-file clips_selected_train.txt --features scene` first.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from horizon_pipeline.raw.fetch import read_clip_list
from horizon_pipeline.raw.store import sha256

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
OLD = REPO / "data" / "pai_selected_train"
CLIPS = read_clip_list(REPO / "clips_selected_train.txt")

pytestmark = pytest.mark.skipif(
    not (RAW / "manifest.json").is_file() or not OLD.is_dir(), reason="real data not fetched"
)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((RAW / "manifest.json").read_text())


@pytest.mark.parametrize("clip_id", CLIPS)
@pytest.mark.parametrize(
    "stored, old",
    [
        ("egomotion.offline.parquet", "raw_ego.parquet"),
        ("obstacle.offline.parquet", "raw_obstacles.parquet"),
        ("camera_front_wide_120fov/frame_timestamps.parquet", "front_camera_timestamps.parquet"),
    ],
)
def test_matches_the_old_copy(clip_id, stored, old):
    # Content, not bytes: the old pipeline re-wrote the label parquets with pandas.
    pd.testing.assert_frame_equal(pd.read_parquet(RAW / clip_id / stored), pd.read_parquet(OLD / clip_id / old))


@pytest.mark.parametrize("clip_id", CLIPS)
def test_manifest_hashes_match_the_stored_files(clip_id, manifest):
    files = manifest["clips"][clip_id]["files"]
    assert files, "nothing stored"
    for record in files.values():
        assert sha256(RAW / record["path"]) == record["sha256"]


@pytest.mark.parametrize("clip_id", CLIPS)
def test_scene_has_lidar(clip_id, manifest):
    clip = manifest["clips"][clip_id]
    assert clip["feature_presence"]["lidar_top_360fov"] is True
    record = clip["files"]["lidar_top_360fov:pointclouds"]
    assert record["path"] == f"{clip_id}/lidar_top_360fov/pointclouds.parquet"
    assert record["bytes"] == (RAW / record["path"]).stat().st_size > 0


@pytest.mark.parametrize("clip_id", CLIPS)
def test_calibration_holds_only_this_clip(clip_id, manifest):
    files = manifest["clips"][clip_id]["files"]
    for key, record in files.items():
        if record["member"] is None:
            ids = pd.read_parquet(RAW / record["path"]).index.get_level_values("clip_id")
            assert set(ids) == {clip_id}, key
