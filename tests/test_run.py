"""T014 horizon runner, on a synthetic scene: a car driving straight along x at 10 m/s."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from horizon_pipeline.horizon import run as run_mod
from horizon_pipeline.horizon.cli import main
from horizon_pipeline.horizon.config import HorizonConfig, parse_config
from horizon_pipeline.horizon.run import COLUMNS, run_horizons
from horizon_pipeline.reference.format import write_reference

N = 100
CFG = HorizonConfig(r_veh_m=2.0, rho0_m=0.2, e_v_bar_mps=0.3, e_theta_bar_rad=0.1, eps_p_m=0.3, eps_v_mps=0.5,
                    a_bar_mps2=2.0, t_valid_s=6.4, grid_s=0.01, time_tolerance_s=1e-3)  # fmt: skip


def ego_table():
    t = np.arange(N) * 0.1
    return pd.DataFrame({"tick": np.arange(N), "t_s": t, "x": 10 * t, "y": 0.0, "speed": np.r_[np.nan, np.full(N - 1, 10.0)]})


def plan_rows(tick, sample=0, y_offset=0.0, steps=65):
    tau = np.arange(steps) * 0.1
    return pd.DataFrame({"tick": tick, "sample": sample, "step": np.arange(steps), "tau_s": tau,
                         "x": 10 * (tick * 0.1 + tau), "y": y_offset, "z": 0.0, "yaw": 0.0, "speed": 10.0})  # fmt: skip


def reference(ticks, samples=1):
    return pd.concat([plan_rows(t, s, y_offset=3.0 * s) for t in ticks for s in range(samples)], ignore_index=True)


def obj(track, tick, x, y, vx=0.0, vy=0.0, in_scene=True, size=(4.0, 2.0)):
    return {"tick": tick, "track_id": track, "x": x, "y": y, "vx": vx, "vy": vy,
            "length": size[0], "width": size[1], "in_scene": in_scene}  # fmt: skip


def objects_table(rows):
    df = pd.DataFrame(rows, columns=["tick", "track_id", "x", "y", "vx", "vy", "length", "width", "in_scene"])
    df["in_scene"] = df["in_scene"].astype("boolean")
    return df


def parked_car_ahead(ticks=range(N), x=60.0, **kw):
    return [obj("car", t, x, 0.0, **kw) for t in ticks]


def run(objects, ref, cfg=CFG):
    return run_horizons(ego_table(), objects_table(objects), ref, "test", 2.0, cfg).set_index(["tick", "sample"])


def test_changing_an_object_at_t_plus_1_does_not_change_h_at_t():
    ref = reference([20, 21])
    base = run(parked_car_ahead(), ref)
    moved = [o if o["tick"] != 21 else {**o, "x": 30.0, "vx": -20.0} for o in parked_car_ahead()]
    changed = run(moved, ref)
    assert changed.loc[(20, 0), "H_s"] == base.loc[(20, 0), "H_s"]
    assert changed.loc[(21, 0), "H_s"] < base.loc[(21, 0), "H_s"]  # the change does reach tick 21


def test_an_object_not_in_scene_at_t_has_no_effect_on_h_at_t():
    ref = reference([20])
    hidden = [obj("ped", 20, 25.0, 0.0, in_scene=False, size=(0.6, 0.6))]
    seen = [obj("ped", 20, 25.0, 0.0, size=(0.6, 0.6))]
    alone = run(parked_car_ahead(), ref)
    with_hidden = run(parked_car_ahead() + hidden, ref)
    pd.testing.assert_series_equal(alone["H_s"], with_hidden["H_s"])
    assert with_hidden.loc[(20, 0), "n_in_scene"] == 1
    assert run(parked_car_ahead() + seen, ref).loc[(20, 0), "limiting_track_id"] == "ped"


def test_plans_only_at_some_ticks_give_exactly_those_rows():
    out = run(parked_car_ahead(), reference([16, 19, 22]))
    assert list(out.index.get_level_values("tick")) == [16, 19, 22]


def test_two_samples_at_one_tick_give_two_rows_with_their_own_h():
    # Sample 1 runs 3 m to the left, 3 m closer to a car parked at y = 4.5: it hits it sooner.
    out = run([obj("car", 20, 40.0, 4.5)], reference([20], samples=2))
    assert len(out) == 2
    h0, h1 = out.loc[(20, 0), "H_s"], out.loc[(20, 1), "H_s"]
    assert 0 < h1 < h0


def test_no_velocity_objects_are_excluded_but_counted_or_treated_as_static():
    ref = reference([20])
    rows = parked_car_ahead() + [obj("new", 20, 35.0, 0.0, vx=np.nan, vy=np.nan)]
    excluded = run(rows, ref)
    assert excluded.loc[(20, 0), "n_no_velocity"] == 1
    assert list(excluded.loc[(20, 0), "no_velocity_track_ids"]) == ["new"]
    assert excluded.loc[(20, 0), "H_s"] == run(parked_car_ahead(), ref).loc[(20, 0), "H_s"]
    static = run(rows, ref, replace(CFG, no_velocity="static"))
    assert static.loc[(20, 0), "limiting_track_id"] == "new"
    assert static.loc[(20, 0), "H_s"] == run(parked_car_ahead(x=35.0), ref).loc[(20, 0), "H_s"]


def test_the_config_method_picks_the_solver(monkeypatch):
    calls = []

    def fake(plan, objects, r_veh, cfg):
        calls.append(len(objects.xy))
        return run_mod.solve(plan, objects, r_veh, cfg)

    monkeypatch.setitem(run_mod.SOLVERS, "baseline", fake)
    run(parked_car_ahead(), reference([20, 30]))
    assert calls == [1, 1]


def test_output_columns_and_values():
    out = run_horizons(ego_table(), objects_table(parked_car_ahead()), reference([0, 20]), "test", 2.0, CFG)
    assert list(out.columns) == COLUMNS
    row = out.iloc[1]
    assert row["t_s"] == pytest.approx(2.0) and row["ego_speed_mps"] == 10.0 and row["reference_source"] == "test"
    assert np.isnan(out.iloc[0]["ego_speed_mps"])  # tick 0 has no speed yet
    assert 0 < row["H_s"] < row["T_cap_s"] and not row["capped"] and row["limiting_track_id"] == "car"


def test_a_plan_at_a_tick_the_scene_lacks_is_refused():
    with pytest.raises(ValueError, match="tick 150"):
        run(parked_car_ahead(), reference([150]))


def write_inputs(tmp_path):
    scene = tmp_path / "scene" / "clip-0" / "offline"
    scene.mkdir(parents=True)
    ego_table().to_parquet(scene / "ego.parquet", index=False)
    objects_table(parked_car_ahead()).to_parquet(scene / "objects.parquet", index=False)
    footprint = {"x_min": -1.0, "x_max": 3.6, "y_min": -1.0, "y_max": 1.0}
    (scene / "meta.json").write_text(json.dumps({"clip_id": "clip-0", "ego_source": "offline", "ego": {"footprint": footprint}}))
    meta = {"source": "recorded", "clip_id": "clip-0", "ego_source": "offline", "frame": "world", "step_s": 0.1,
            "n_steps": {"min": 65, "max": 65}, "inputs": {}}  # fmt: skip
    write_reference(reference([16, 19, 22]), meta, tmp_path / "reference" / "clip-0" / "offline" / "recorded")
    raw = CFG.to_dict()
    raw["ego"]["r_veh_m"] = "auto"
    (tmp_path / "horizon.yaml").write_text(yaml.safe_dump(raw))
    return tmp_path


def test_cli_writes_results_config_and_meta(tmp_path, capsys):
    root = write_inputs(tmp_path)
    args = ["--clips", "clip-0", "--reference", "recorded", "--name", "demo", "--config", str(root / "horizon.yaml"),
            "--scenes", str(root / "scene"), "--references", str(root / "reference"), "--out", str(root / "results")]  # fmt: skip
    assert main(args) == 0
    out_dir = root / "results" / "demo" / "clip-0" / "offline" / "recorded"
    out = pd.read_parquet(out_dir / "horizons.parquet")
    assert list(out["tick"]) == [16, 19, 22]
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["n_plans"] == 3 and meta["n_ticks_without_plan"] == N - 3 and meta["method"] == "baseline"
    assert meta["r_veh_m"] == pytest.approx(np.hypot(3.6, 1.0))
    assert set(meta["inputs"]["files"]) >= {"scene/objects.parquet", "reference/recorded/reference.parquet", "config"}
    assert parse_config(yaml.safe_load((out_dir / "config_used.yaml").read_text())).r_veh_m is None
    assert "3 plans" in capsys.readouterr().out


def test_cli_says_what_to_build_when_the_reference_is_missing(tmp_path, capsys):
    root = write_inputs(tmp_path)
    args = ["--clips", "clip-0", "--reference", "alpamayo", "--name", "demo", "--config", str(root / "horizon.yaml"),
            "--scenes", str(root / "scene"), "--references", str(root / "reference"), "--out", str(root / "results")]  # fmt: skip
    assert main(args) == 1
    assert "horizon-reference alpamayo" in capsys.readouterr().err


def test_a_capped_plan_has_a_missing_limiting_track_not_the_string_nan():
    # Tick 20 is limited by a car, tick 21 sees nothing: a column mixing an ID and a missing value.
    out = run_horizons(ego_table(), objects_table([obj("car", 20, 40.0, 0.0)]), reference([20, 21]), "test", 2.0, CFG)
    assert list(out["capped"]) == [False, True]
    assert out["limiting_track_id"].iloc[0] == "car" and out["limiting_track_id"].iloc[1] is None
