"""T013 solver on the real clips with the recorded stand-in. Skipped unless the scenes and references
are built (`horizon-scene build`, `horizon-reference recorded`)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horizon_pipeline.horizon.config import load_config
from horizon_pipeline.horizon.solver import Objects, Plan, ego_radius, solve
from horizon_pipeline.raw.fetch import read_clip_list
from horizon_pipeline.reference.format import read_reference

REPO = Path(__file__).resolve().parents[1]
CLIPS = read_clip_list(REPO / "clips_selected_train.txt")
SCENES, REFS = REPO / "data" / "scene", REPO / "data" / "reference"

pytestmark = pytest.mark.skipif(
    not all((REFS / c / "offline" / "recorded" / "reference.parquet").is_file() for c in CLIPS),
    reason="scenes or references not built",
)


@pytest.mark.parametrize("clip", CLIPS, ids=str)
def test_every_recorded_plan_solves(clip):
    cfg = load_config(REPO / "configs" / "horizon.yaml")
    scene = SCENES / clip / "offline"
    r_veh = ego_radius(json.loads((scene / "meta.json").read_text())["ego"]["footprint"])
    assert r_veh == pytest.approx(3.790, abs=5e-4)
    objects = pd.read_parquet(scene / "objects.parquet")
    ref, _ = read_reference(REFS / clip / "offline" / "recorded")
    for tick, plan in ref.groupby("tick"):
        o = objects[(objects["tick"] == tick) & objects["in_scene"]]
        o = o[o[["vx", "vy"]].notna().all(axis=1)]  # no velocity yet at a track's first tick: decided outside T013
        h = solve(Plan.from_rows(plan), Objects.from_rows(o), r_veh, cfg)
        assert 0 <= h.H_s <= h.T_cap_s <= cfg.t_valid_s
        assert h.T_cap_s == pytest.approx(min(plan["tau_s"].iloc[-1], cfg.t_valid_s))
        assert h.capped == (h.limiting_track_id is None)
        assert (h.H_s == h.T_cap_s) or not h.capped
        assert np.isfinite(h.q_bar_mps) and (np.isnan(h.margin_at_0_m) == o.empty)
