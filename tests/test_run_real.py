"""T014 on the real clips with the recorded stand-in. Skipped unless the scenes and references are
built (`horizon-scene build`, `horizon-reference recorded`)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horizon_pipeline.horizon.config import load_config
from horizon_pipeline.horizon.run import run_horizons, run_one
from horizon_pipeline.horizon.solver import ego_radius
from horizon_pipeline.raw.fetch import read_clip_list
from horizon_pipeline.reference.format import read_reference

REPO = Path(__file__).resolve().parents[1]
CLIPS = read_clip_list(REPO / "clips_selected_train.txt")
SCENES, REFS = REPO / "data" / "scene", REPO / "data" / "reference"
CASES = [(c, s) for c in CLIPS for s in ("offline", "online")]

pytestmark = pytest.mark.skipif(
    not all((REFS / c / s / "recorded" / "reference.parquet").is_file() for c, s in CASES),
    reason="scenes or references not built",
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_one_row_per_plan_with_consistent_values(case):
    clip, src = case
    cfg = load_config(REPO / "configs" / "horizon.yaml")
    out, meta = run_one(SCENES / clip / src, REFS / clip / src / "recorded", cfg)
    ref, _ = read_reference(REFS / clip / src / "recorded")
    assert len(out) == ref.groupby(["tick", "sample"]).ngroups == meta["n_plans"]
    assert ((out["H_s"] >= 0) & (out["H_s"] <= out["T_cap_s"]) & (out["T_cap_s"] <= cfg.t_valid_s + 1e-12)).all()
    assert (out["capped"] == out["limiting_track_id"].isna()).all()
    assert (out["n_no_velocity"] == out["no_velocity_track_ids"].map(len)).all()
    assert (out["n_no_velocity"] <= out["n_in_scene"]).all()


@pytest.mark.parametrize("clip", CLIPS, ids=str)
def test_h_at_t_does_not_depend_on_any_later_object_row(clip):
    """Causality on real data: deleting every object row after tick t leaves H at t unchanged."""
    scene = SCENES / clip / "offline"
    cfg = load_config(REPO / "configs" / "horizon.yaml")
    r_veh = ego_radius(json.loads((scene / "meta.json").read_text())["ego"]["footprint"])
    ego = pd.read_parquet(scene / "ego.parquet")
    objects = pd.read_parquet(scene / "objects.parquet")
    ref, _ = read_reference(REFS / clip / "offline" / "recorded")
    full = run_horizons(ego, objects, ref, "recorded", r_veh, cfg).set_index("tick")
    for t in [20, 80, 150]:
        cut = run_horizons(ego, objects[objects["tick"] <= t], ref[ref["tick"] == t], "recorded", r_veh, cfg)
        assert cut["H_s"].item() == full.at[t, "H_s"]
        assert cut["limiting_track_id"].item() == full.at[t, "limiting_track_id"]
