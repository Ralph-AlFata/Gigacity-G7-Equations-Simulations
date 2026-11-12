"""T011 reference format and recorded-path reference, on a synthetic scene ego table."""

import json

import numpy as np
import pandas as pd
import pytest

from horizon_pipeline.reference.cli import main
from horizon_pipeline.reference.format import COLUMNS, ReferenceError, read_reference, validate, write_reference
from horizon_pipeline.reference.recorded import build_recorded, recorded_reference

N = 30


def synthetic_ego(n=N, seed=0):
    """An ego table shaped like the scene's: random poses, speed NaN at tick 0 as in T009."""
    rng = np.random.default_rng(seed)
    ego = pd.DataFrame({"tick": np.arange(n), "t_s": np.arange(n) * 0.1})
    ego[["x", "y", "z"]] = np.cumsum(rng.normal(size=(n, 3)), axis=0)
    ego["yaw"] = np.cumsum(rng.normal(scale=0.1, size=n))
    ego["speed"] = rng.uniform(0, 15, n)
    ego.loc[0, "speed"] = np.nan
    return ego


def write_scene_dir(root, ego):
    d = root / "clip-0" / "offline"
    d.mkdir(parents=True)
    ego.to_parquet(d / "ego.parquet", index=False)
    (d / "meta.json").write_text(json.dumps({"clip_id": "clip-0", "ego_source": "offline", "rate_hz": 10}))
    return d


@pytest.fixture
def built(tmp_path):
    return build_recorded(write_scene_dir(tmp_path / "scene", synthetic_ego()))


def test_step_k_of_tick_t_is_the_ego_pose_at_tick_t_plus_k():
    ego = synthetic_ego()
    ref, filled = recorded_reference(ego)
    assert len(ref) == N * (N + 1) // 2 and filled == 1
    for (t, k), row in ref.set_index(["tick", "step"]).iterrows():
        e = ego.iloc[t + k]
        assert row["sample"] == 0 and row["tau_s"] == pytest.approx(0.1 * k)
        assert [row[c] for c in ["x", "y", "z", "yaw"]] == [e[c] for c in ["x", "y", "z", "yaw"]]
        assert row["speed"] == (ego["speed"].iloc[1] if t + k == 0 else e["speed"])
    assert ref.groupby("tick").size().tolist() == list(range(N, 0, -1))  # the last tick plans 0 s ahead


def test_only_leading_missing_speeds_are_filled():
    ego = synthetic_ego()
    ego.loc[5, "speed"] = np.nan
    with pytest.raises(ValueError, match="after tick 1"):
        recorded_reference(ego)


def test_write_then_read_round_trips(tmp_path, built):
    ref, meta = built
    out = write_reference(ref, meta, tmp_path / "ref")
    back, back_meta = read_reference(out)
    pd.testing.assert_frame_equal(back, ref.reset_index(drop=True), check_dtype=False)
    assert back_meta == json.loads(json.dumps(meta))
    assert meta["n_steps"] == {"min": 1, "max": N} and meta["speed_filled_ticks"] == [0]
    assert "rear-axle" in meta["frame"] and len(meta["inputs"]["files"]["scene/ego.parquet"]) == 64


def broken(ref, how):
    ref = ref.copy()
    if how == "missing column":
        return ref.drop(columns="yaw")
    if how == "NaN":
        ref.loc[7, "x"] = np.nan
    elif how == "steps out of order":
        ref.iloc[[3, 4]] = ref.iloc[[4, 3]].to_numpy()
    elif how == "step skipped":
        ref = ref.drop(index=3).reset_index(drop=True)
    elif how == "plan not starting at step 0":
        ref = ref.drop(index=0).reset_index(drop=True)
    elif how == "plan repeated":
        ref = pd.concat([ref, ref[ref["tick"] == 0]], ignore_index=True)
    elif how == "tau_s off":
        ref.loc[5, "tau_s"] += 0.01
    elif how == "float steps":
        ref["step"] = ref["step"].astype(float)
    return ref


@pytest.mark.parametrize("how", [
    "missing column", "NaN", "steps out of order", "step skipped", "plan not starting at step 0",
    "plan repeated", "tau_s off", "float steps",
])  # fmt: skip
def test_reader_rejects_bad_files(tmp_path, built, how):
    ref, meta = built
    out = write_reference(ref, meta, tmp_path / "ref")
    broken(ref, how).to_parquet(out / "reference.parquet", index=False)
    with pytest.raises(ReferenceError):
        read_reference(out)


def test_meta_must_record_the_frame(built):
    ref, meta = built
    with pytest.raises(ReferenceError, match="frame"):
        validate(ref, {k: v for k, v in meta.items() if k != "frame"})


def test_writer_validates_too(tmp_path, built):
    ref, meta = built
    with pytest.raises(ReferenceError):
        write_reference(broken(ref, "NaN"), meta, tmp_path / "ref")
    assert not (tmp_path / "ref").exists()


def test_cli_writes_the_recorded_reference(tmp_path, capsys):
    write_scene_dir(tmp_path / "scene", synthetic_ego())
    assert main(["recorded", "--clips", "clip-0", "--scenes", str(tmp_path / "scene"), "--out", str(tmp_path / "ref")]) == 0
    ref, meta = read_reference(tmp_path / "ref" / "clip-0" / "offline" / "recorded")
    assert meta["source"] == "recorded" and list(ref.columns) == COLUMNS
    assert f"{N} plans, {N * (N + 1) // 2} rows" in capsys.readouterr().out
    assert main(["recorded", "--clips", "nope", "--scenes", str(tmp_path / "scene"), "--out", str(tmp_path / "ref")]) == 1
