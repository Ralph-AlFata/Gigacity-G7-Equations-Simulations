"""T012 on the real clips: Alpamayo-shaped files made from each online scene's own future, with the
loader's transform, must import back exactly onto the ego track. Skipped unless the online scenes
are built (`horizon-scene build --ego-source online`)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horizon_pipeline.raw.fetch import read_clip_list
from horizon_pipeline.reference.alpamayo import N_STEPS, build_alpamayo, loader_frame, write_snapshot
from horizon_pipeline.reference.format import validate

REPO = Path(__file__).resolve().parents[1]
SCENES = REPO / "data" / "scene"
CLIPS = read_clip_list(REPO / "clips_selected_train.txt")

pytestmark = pytest.mark.skipif(
    not all((SCENES / c / "online" / "ego.parquet").is_file() for c in CLIPS), reason="scenes not built"
)


@pytest.mark.parametrize("clip", CLIPS)
def test_snapshots_from_the_ego_future_import_onto_the_ego_track(tmp_path, clip):
    scene_dir = SCENES / clip / "online"
    ego = pd.read_parquet(scene_dir / "ego.parquet")
    src = tmp_path / clip
    src.mkdir()
    (src / "run.json").write_text(json.dumps({
        "model": "test: ego future", "ego_source": "online", "snapshot_interval_s": 0.3,
        "n_cameras": 4, "history_s": 1.6,
    }))
    ticks = list(range(16, len(ego) - N_STEPS, 3))  # every 0.3 s from 1.6 s, with 6.4 s of future
    for t in ticks:
        xyz, rot = loader_frame(ego, t)
        write_snapshot(src, clip, int(ego.loc[t, "timestamp_us"]), xyz[None], rot[None])
    ref, meta = build_alpamayo(scene_dir, src)
    validate(ref, meta)
    assert sorted(ref["tick"].unique()) == ticks
    on = ref["tick"].to_numpy() + ref["step"].to_numpy()
    for c in ["x", "y", "z", "yaw"]:
        np.testing.assert_allclose(ref[c], ego[c].to_numpy()[on], atol=1e-6, rtol=0)
    np.testing.assert_allclose(ref["speed"], ego["speed"].to_numpy()[on], atol=1e-6, rtol=0)
    assert meta["step1_gap_m"]["max_abs"] < 0.1
