"""T012 Alpamayo import, on synthetic scenes and snapshots."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from horizon_pipeline.reference.alpamayo import (
    N_STEPS,
    AlpamayoImportError,
    build_alpamayo,
    loader_frame,
    write_snapshot,
)
from horizon_pipeline.reference.cli import main
from horizon_pipeline.reference.format import read_reference, validate

CLIP = "clip-0"
N = 120


def synthetic_ego(n=N):
    """A car on a left curve at ~8 m/s with a little roll and pitch, shaped like the scene's ego table."""
    t = np.arange(n) * 0.1
    yaw = 0.05 * t + 0.002 * t**2
    speed = 8 + 0.1 * t
    ego = pd.DataFrame({"tick": np.arange(n), "t_s": t, "timestamp_us": np.arange(n) * 100_000})
    ego["x"] = np.cumsum(speed * np.cos(yaw)) * 0.1
    ego["y"] = np.cumsum(speed * np.sin(yaw)) * 0.1
    ego["z"] = 0.01 * t
    rot = Rotation.from_euler("ZYX", np.c_[yaw, np.full(n, 0.03), np.full(n, -0.02)])
    ego[["qx", "qy", "qz", "qw"]] = rot.as_quat()
    ego["yaw"] = yaw
    ego["speed"] = np.r_[np.nan, np.linalg.norm(np.diff(ego[["x", "y", "z"]].to_numpy(), axis=0), axis=1) / 0.1]
    ego["has_history"] = t >= 1.6 - 1e-9
    return ego


def scene_dir(root, ego, ego_source="online"):
    d = root / "scene" / CLIP / ego_source
    d.mkdir(parents=True)
    ego.to_parquet(d / "ego.parquet", index=False)
    (d / "meta.json").write_text(json.dumps({"clip_id": CLIP, "ego_source": ego_source, "rate_hz": 10}))
    return d


def run_dir(root, **run):
    d = root / "alpamayo" / CLIP
    d.mkdir(parents=True)
    run = {"model": "nvidia/Alpamayo-1.5-10B", "ego_source": "online", "snapshot_interval_s": 0.3,
           "n_cameras": 4, "history_s": 1.6, **run}
    (d / "run.json").write_text(json.dumps(run))
    return d


@pytest.fixture
def ego():
    return synthetic_ego()


def test_a_trajectory_in_the_ego_frame_lands_at_the_expected_world_coordinates(tmp_path):
    # The ego at tick 20 sits at (10, 5, 0) facing +y (yaw 90°), flat. Alpamayo says "1 m ahead per step".
    ego = synthetic_ego()
    ego.loc[20, ["x", "y", "z", "yaw"]] = [10.0, 5.0, 0.0, np.pi / 2]
    ego.loc[20, ["qx", "qy", "qz", "qw"]] = Rotation.from_euler("z", np.pi / 2).as_quat()
    xyz = np.zeros((1, N_STEPS, 3))
    xyz[0, :, 0] = np.arange(1, N_STEPS + 1)
    rot = np.broadcast_to(np.eye(3), (1, N_STEPS, 3, 3))
    clip = run_dir(tmp_path)
    write_snapshot(clip, CLIP, 2_000_000, xyz, rot)
    ref, meta = build_alpamayo(scene_dir(tmp_path, ego), clip)
    validate(ref, meta)
    assert len(ref) == N_STEPS + 1 and (ref["tick"] == 20).all()
    np.testing.assert_allclose(ref["x"], 10.0, atol=1e-12)
    np.testing.assert_allclose(ref["y"], 5.0 + np.arange(N_STEPS + 1), atol=1e-12)
    np.testing.assert_allclose(ref["z"], 0.0, atol=1e-12)
    np.testing.assert_allclose(ref["yaw"], np.pi / 2, atol=1e-12)
    np.testing.assert_allclose(ref["speed"].iloc[1:], 10.0)  # 1 m per 0.1 s
    assert ref["speed"].iloc[0] == ego.loc[20, "speed"]


def test_the_loaders_transform_is_undone_exactly(tmp_path, ego):
    clip = run_dir(tmp_path)
    for tick in [16, 40, 55]:
        xyz, rot = loader_frame(ego, tick)
        write_snapshot(clip, CLIP, tick * 100_000, xyz[None], rot[None])
    ref, meta = build_alpamayo(scene_dir(tmp_path, ego), clip)
    assert meta["n_ticks"] == 3 and meta["n_samples"] == 1
    on = ref["tick"].to_numpy() + ref["step"].to_numpy()
    for c in ["x", "y", "z", "yaw"]:
        np.testing.assert_allclose(ref[c], ego[c].to_numpy()[on], atol=1e-9)
    np.testing.assert_allclose(ref["speed"], ego["speed"].to_numpy()[on], atol=1e-9)


def test_a_snapshot_with_3_samples_gives_3_plans_at_that_tick_in_sample_order(tmp_path, ego):
    xyz, rot = loader_frame(ego, 30)
    shift = np.array([0.0, 1.0, 2.0])[:, None, None] * [0, 1, 0]  # sample s is s metres to the left
    clip = run_dir(tmp_path)
    write_snapshot(clip, CLIP, 3_000_000, xyz[None] + shift, np.stack([rot] * 3))
    ref, meta = build_alpamayo(scene_dir(tmp_path, ego), clip)
    assert meta["n_samples"] == 3
    assert list(ref.groupby("sample").size()) == [N_STEPS + 1] * 3
    assert (ref["tick"] == 30).all() and ref["sample"].is_monotonic_increasing
    # Left of the ego at tick 30 + k, by s metres, in the ego's frame at tick 30.
    left = Rotation.from_quat(ego.loc[30, ["qx", "qy", "qz", "qw"]].to_numpy(float)).apply([0, 1, 0])
    last = ref[ref["step"] == N_STEPS].set_index("sample")[["x", "y", "z"]].to_numpy()
    np.testing.assert_allclose(last - last[0], np.outer([0, 1, 2], left), atol=1e-9)


def test_an_output_not_on_the_tick_grid_is_rejected(tmp_path, ego):
    xyz, rot = loader_frame(ego, 30)
    clip = run_dir(tmp_path)
    write_snapshot(clip, CLIP, 3_050_000, xyz[None], rot[None])  # halfway between ticks 30 and 31
    with pytest.raises(AlpamayoImportError, match="matches no scene tick within 1000 µs"):
        build_alpamayo(scene_dir(tmp_path, ego), clip)


def test_an_output_within_1_ms_of_a_tick_is_matched(tmp_path, ego):
    xyz, rot = loader_frame(ego, 30)
    clip = run_dir(tmp_path)
    write_snapshot(clip, CLIP, 3_000_800, xyz[None], rot[None])
    ref, _ = build_alpamayo(scene_dir(tmp_path, ego), clip)
    assert (ref["tick"] == 30).all()


@pytest.mark.parametrize(
    "case, match",
    [
        ("no history", "less than the 1.6 s of history"),
        ("other ego source", "the run used offline egomotion"),
        ("other clip", "is for clip clip-9"),
        ("wrong shape", "pred_xyz has shape"),
        ("misnamed", "expected the name t_003000000.npz"),
        ("same tick twice", "two snapshots match the same scene tick"),
    ],
)
def test_contract_violations_are_refused(tmp_path, ego, case, match):
    xyz, rot = loader_frame(ego, 30)
    clip = run_dir(tmp_path, **({"ego_source": "offline"} if case == "other ego source" else {}))
    if case == "no history":
        xyz, rot = loader_frame(ego, 15)
        write_snapshot(clip, CLIP, 1_500_000, xyz[None], rot[None])
    elif case == "other clip":
        write_snapshot(clip, "clip-9", 3_000_000, xyz[None], rot[None])
    elif case == "wrong shape":
        write_snapshot(clip, CLIP, 3_000_000, xyz[None, :50], rot[None, :50])
    elif case == "misnamed":
        write_snapshot(clip, CLIP, 3_000_000, xyz[None], rot[None]).rename(clip / "t_000000001.npz")
    elif case == "same tick twice":
        write_snapshot(clip, CLIP, 3_000_000, xyz[None], rot[None])
        write_snapshot(clip, CLIP, 3_000_500, xyz[None], rot[None])
    else:
        write_snapshot(clip, CLIP, 3_000_000, xyz[None], rot[None])
    with pytest.raises(AlpamayoImportError, match=match):
        build_alpamayo(scene_dir(tmp_path, ego), clip)


def test_run_json_must_hold_the_run_settings(tmp_path, ego):
    clip = run_dir(tmp_path)
    (clip / "run.json").write_text(json.dumps({"model": "x"}))
    xyz, rot = loader_frame(ego, 30)
    write_snapshot(clip, CLIP, 3_000_000, xyz[None], rot[None])
    with pytest.raises(AlpamayoImportError, match="missing"):
        build_alpamayo(scene_dir(tmp_path, ego), clip)


def test_cli_writes_a_readable_reference_with_the_check_values(tmp_path, ego, capsys):
    scenes = scene_dir(tmp_path, ego).parents[1]
    clip = run_dir(tmp_path)
    for tick in [20, 23, 26]:
        xyz, rot = loader_frame(ego, tick)
        write_snapshot(clip, CLIP, tick * 100_000, xyz[None], rot[None])
    out = tmp_path / "reference"
    assert main(["alpamayo", "--clips", CLIP, "--from", str(clip.parent), "--scenes", str(scenes), "--out", str(out)]) == 0
    ref, meta = read_reference(out / CLIP / "online" / "alpamayo")
    assert sorted(ref["tick"].unique()) == [20, 23, 26]
    assert meta["source"] == "alpamayo" and meta["snapshot_interval_s"] == 0.3 and meta["run"]["n_cameras"] == 4
    assert meta["step1_gap_m"]["max_abs"] < 0.01  # the ego's own future: step 1 is 0.1 s at its speed
    assert "3 snapshots" in capsys.readouterr().out


def test_cli_reports_a_bad_snapshot_without_a_traceback(tmp_path, ego, capsys):
    scenes = scene_dir(tmp_path, ego).parents[1]
    clip = run_dir(tmp_path)
    xyz, rot = loader_frame(ego, 30)
    write_snapshot(clip, CLIP, 3_050_000, xyz[None], rot[None])
    assert main(["alpamayo", "--clips", CLIP, "--from", str(clip.parent), "--scenes", str(scenes), "--out", str(tmp_path / "r")]) == 1
    assert "matches no scene tick" in capsys.readouterr().err
