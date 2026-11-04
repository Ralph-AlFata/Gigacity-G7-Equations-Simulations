"""T007 scene builder, on synthetic clips whose true world states are known exactly."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from horizon_pipeline.scene.build import DuplicateError, Rules, build_scene, make_grid, write_scene
from horizon_pipeline.scene.frames import Poses, dense_end, to_local, to_parent, yaw_of

CLIP = "synthetic-0000"
EGO_V = np.array([8.0, -1.5, 0.05])  # m/s, world frame
EGO_P0 = np.array([0.0, 0.0, 0.0])
EGO_YAW_RATE = 0.2  # rad/s, about z: slerp and linear interpolation are both exact for this motion


# ---------------------------------------------------------------- an independent reference


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from (qx, qy, qz, qw), written out by hand (no scipy)."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------- synthetic clip


def ego_at(t_s):
    t_s = np.atleast_1d(np.asarray(t_s, float))
    return EGO_P0 + EGO_V * t_s[:, None], EGO_YAW_RATE * t_s


def write_clip(root, tracks: dict, ego_t_us=None, extra_rows=()):
    """`tracks`: track_id -> (class, sighting times in us, world center fn(t_s), world yaw fn(t_s), size).

    Sightings are written the way the dataset stores them: the box in the rig frame at its own time.
    """
    d = root / CLIP
    d.mkdir(parents=True)
    t_us = np.arange(0, 3_000_001, 10_000) if ego_t_us is None else np.asarray(ego_t_us)
    p, yaw = ego_at(t_us / 1e6)
    ego = pd.DataFrame({"timestamp": t_us, "x": p[:, 0], "y": p[:, 1], "z": p[:, 2]})
    ego[["qx", "qy", "qz", "qw"]] = Rotation.from_euler("z", yaw[:, None]).as_quat()
    ego.to_parquet(d / "egomotion.offline.parquet", index=False)
    ego.to_parquet(d / "egomotion.parquet", index=False)
    rows = []
    for track, (cls, times, center, box_yaw, size) in tracks.items():
        for t in times:
            p_e, yaw_e = ego_at(t / 1e6)
            r_e = Rotation.from_euler("z", yaw_e[0])
            c_rig = r_e.inv().apply(center(t / 1e6) - p_e[0])
            q_rig = (r_e.inv() * Rotation.from_euler("z", box_yaw(t / 1e6))).as_quat()
            rows.append({
                "timestamp_us": np.uint64(t), "track_id": track, "label_class": cls,
                "center_x": c_rig[0], "center_y": c_rig[1], "center_z": c_rig[2],
                "size_x": size[0], "size_y": size[1], "size_z": size[2],
                "orientation_x": q_rig[0], "orientation_y": q_rig[1], "orientation_z": q_rig[2], "orientation_w": q_rig[3],
            })  # fmt: skip
    rows += list(extra_rows)
    obs = pd.DataFrame(rows)
    obs["timestamp_us"] = obs["timestamp_us"].astype(np.uint64)
    obs.to_parquet(d / "obstacle.offline.parquet", index=False)
    pd.DataFrame(
        {"length": [4.688], "width": [1.999], "height": [1.439], "rear_axle_to_bbox_center": [1.312],
         "wheelbase": [2.73], "track_width": [1.6005]},
        index=pd.Index([CLIP], name="clip_id"),
    ).to_parquet(d / "vehicle_dimensions.parquet")  # fmt: skip
    return d


def mover(v=(5.0, 2.0, 0.0), c0=(20.0, -3.0, 0.8), yaw=0.4):
    v, c0 = np.array(v), np.array(c0)
    return (lambda t: c0 + v * t), (lambda t: yaw), v


def clock(start_us, end_us, step_us=100_000):
    return list(range(start_us, end_us + 1, step_us))


# ---------------------------------------------------------------- frames


def test_round_trip_world_ego_world():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 3)) * 30
    origin = rng.normal(size=3) * 10
    rot = Rotation.random(random_state=1)
    np.testing.assert_allclose(to_parent(to_local(pts, origin, rot), origin, rot), pts, atol=1e-9)


def test_to_local_matches_hand_written_matrices():
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(20, 3)) * 30
    origin = rng.normal(size=3) * 10
    q = Rotation.random(random_state=3).as_quat()
    expected = (quat_to_matrix(q).T @ (pts - origin).T).T  # R^T (x - p)
    np.testing.assert_allclose(to_local(pts, origin, Rotation.from_quat(q)), expected, atol=1e-9)


def test_yaw_of_matches_hand_written_matrix():
    for yaw in [-3.0, -1.0, 0.0, 0.5, 3.1]:
        r = Rotation.from_matrix(yaw_matrix(yaw)) * Rotation.from_euler("y", 0.05)  # a little pitch
        assert yaw_of(r) == pytest.approx(yaw, abs=1e-12)


def test_poses_interpolate_and_never_extrapolate():
    poses = Poses([0, 100], [[0, 0, 0], [10, 0, 0]], Rotation.from_euler("z", [[0], [1]]).as_quat())
    xyz, rot = poses.at([25])
    np.testing.assert_allclose(xyz[0], [2.5, 0, 0])
    assert yaw_of(rot)[0] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        poses.at([101])
    with pytest.raises(ValueError):
        poses.at([-1])


def test_dense_end_cuts_the_sparse_tail():
    t = np.r_[np.arange(0, 1000, 10), 1500, 2500]
    assert dense_end(t) == 99


def test_grid_starts_at_clip_time_zero():
    online_like = Poses([-190_530, 0, 20_199_554], np.zeros((3, 3)), np.tile([0, 0, 0, 1.0], (3, 1)))
    ticks = make_grid(online_like)
    assert ticks[0] == 0 and ticks[-1] == 20_100_000 and np.all(np.diff(ticks) == 100_000)


# ---------------------------------------------------------------- builder


def test_constant_velocity_object_on_an_offset_clock_is_exact(tmp_path):
    center, yaw, v = mover()
    write_clip(tmp_path, {"7": ("automobile", clock(37_000, 2_937_000), center, yaw, (4.0, 1.8, 1.5))})
    scene = build_scene(tmp_path / CLIP, "offline", lidar=False)
    o, e = scene.objects, scene.ego
    t = o["t_s"].to_numpy()
    assert t[0] == pytest.approx(0.1) and t[-1] == pytest.approx(2.9)
    np.testing.assert_allclose(o[["x", "y", "z"]], np.stack([center(s) for s in t]), atol=1e-9)
    np.testing.assert_allclose(o[["vx", "vy", "vz"]], np.tile(v, (len(o), 1)), atol=1e-9)
    np.testing.assert_allclose(o[["ax", "ay", "az"]], 0, atol=1e-6)
    np.testing.assert_allclose(o["yaw"], 0.4, atol=1e-12)
    np.testing.assert_allclose(o["yaw_rate"], 0, atol=1e-9)
    np.testing.assert_allclose(o[["length", "width", "height"]], np.tile([4.0, 1.8, 1.5], (len(o), 1)))
    # Ego-frame columns against hand-written matrices.
    p_e, yaw_e = ego_at(t)
    rel = np.stack([yaw_matrix(a).T @ (center(s) - p) for s, p, a in zip(t, p_e, yaw_e)])
    np.testing.assert_allclose(o[["rel_x", "rel_y", "rel_z"]], rel, atol=1e-9)
    np.testing.assert_allclose(o["rel_yaw"], (0.4 - yaw_e + np.pi) % (2 * np.pi) - np.pi, atol=1e-9)
    np.testing.assert_allclose(o["distance"], np.hypot(rel[:, 0], rel[:, 1]), atol=1e-9)
    np.testing.assert_allclose(o["last_sighting_age_s"], 0.063, atol=1e-9)
    assert (o["gap_s"] == 0).all()
    # Ego table: exact pose and motion.
    np.testing.assert_allclose(e[["vx", "vy", "vz"]], np.tile(EGO_V, (len(e), 1)), atol=1e-9)
    np.testing.assert_allclose(e["yaw_rate"], EGO_YAW_RATE, atol=1e-9)
    np.testing.assert_allclose(e["speed"], np.linalg.norm(EGO_V), atol=1e-9)
    assert e["has_history"].tolist() == [s >= 1.6 - 1e-9 for s in e["t_s"]]


def test_skips_up_to_half_a_second_are_bridged(tmp_path):
    center, yaw, _ = mover()
    short = clock(0, 1_000_000) + clock(1_400_000, 2_000_000)  # 0.4 s skip
    long = clock(0, 1_000_000) + clock(1_600_000, 2_000_000)  # 0.6 s skip
    write_clip(tmp_path, {"short": ("automobile", short, center, yaw, (4, 2, 1.5)),
                          "long": ("automobile", long, center, yaw, (4, 2, 1.5))})  # fmt: skip
    o = build_scene(tmp_path / CLIP, "offline", lidar=False).objects
    ticks = lambda tr: o.loc[o["track_id"] == tr, "tick"].tolist()  # noqa: E731
    assert ticks("short") == list(range(0, 21))
    assert ticks("long") == list(range(0, 11)) + list(range(16, 21))
    bridged = o[(o["track_id"] == "short") & o["tick"].between(11, 13)]
    np.testing.assert_allclose(bridged["gap_s"], 0.4)
    np.testing.assert_allclose(bridged["last_sighting_age_s"], [0.1, 0.2, 0.3], atol=1e-9)
    np.testing.assert_allclose(bridged[["x", "y", "z"]], np.stack([center(s) for s in bridged["t_s"]]), atol=1e-9)
    # Velocity is one-sided at the edges of each run of ticks and still exact for a straight line.
    np.testing.assert_allclose(o.loc[o["track_id"] == "long", "vx"], 5.0, atol=1e-9)


def test_no_rows_outside_a_track_lifetime(tmp_path):
    center, yaw, _ = mover()
    write_clip(tmp_path, {"a": ("person", clock(450_000, 1_250_000), center, yaw, (0.6, 0.6, 1.7))})
    o = build_scene(tmp_path / CLIP, "offline", lidar=False).objects
    assert o["t_s"].min() >= 0.45 and o["t_s"].max() <= 1.25
    assert o["tick"].tolist() == list(range(5, 13))


def test_single_tick_track_has_nan_motion_and_unknown_class_is_kept(tmp_path):
    center, yaw, _ = mover()
    write_clip(tmp_path, {"one": ("hovercraft", [1_000_000], center, yaw, (2, 2, 2))})
    scene = build_scene(tmp_path / CLIP, "offline", lidar=False)
    row = scene.objects.iloc[0]
    assert len(scene.objects) == 1 and row["class_group"] == "unknown"
    assert np.isnan(row["vx"]) and np.isnan(row["yaw_rate"])
    assert scene.report["unknown_classes"] == ["hovercraft"]


def test_consistent_duplicates_are_merged_keeping_the_later_row(tmp_path):
    center, yaw, _ = mover()
    d = write_clip(tmp_path, {"7": ("automobile", clock(0, 1_000_000), center, yaw, (4, 2, 1.5))})
    obs = pd.read_parquet(d / "obstacle.offline.parquet")
    dup = obs.iloc[[5]].copy()
    dup["timestamp_us"] = dup["timestamp_us"] + np.uint64(3_000)  # 3 ms later, 5 cm off
    dup["center_x"] += 0.05
    pd.concat([obs, dup]).to_parquet(d / "obstacle.offline.parquet", index=False)
    scene = build_scene(d, "offline", lidar=False)
    assert scene.report["duplicates_merged"] == 1
    # Tick 5 (0.5 s) now lies before the kept sighting at 0.503 s, so it is interpolated from 0.4 s.
    row = scene.objects.set_index("tick").loc[5]
    assert row["last_sighting_age_s"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "column, change",
    [("label_class", "person"), ("size_x", 0.5), ("center_y", 0.5)],
)
def test_inconsistent_duplicates_stop_the_build(tmp_path, column, change):
    center, yaw, _ = mover()
    d = write_clip(tmp_path, {"7": ("automobile", clock(0, 1_000_000), center, yaw, (4, 2, 1.5))})
    obs = pd.read_parquet(d / "obstacle.offline.parquet")
    dup = obs.iloc[[5]].copy()
    dup["timestamp_us"] = dup["timestamp_us"] + np.uint64(3_000)
    dup[column] = change if column == "label_class" else dup[column] + change
    pd.concat([obs, dup]).to_parquet(d / "obstacle.offline.parquet", index=False)
    with pytest.raises(DuplicateError, match="track 7"):
        build_scene(d, "offline", lidar=False)


def write_far_duplicate(tmp_path, c0, column="center_x", change=0.8):
    """A parked track at `c0` (world = rig frame at t=0, roughly), plus a duplicate of its 6th row
    1.2 ms later with `column` changed by `change` (T020: the b6cb8b12 truck)."""
    center, yaw, _ = mover(v=(0.0, 0.0, 0.0), c0=c0, yaw=0.0)
    d = write_clip(tmp_path, {"228": ("heavy_truck", clock(0, 1_000_000), center, yaw, (10.8, 2.9, 3.8))})
    obs = pd.read_parquet(d / "obstacle.offline.parquet")
    dup = obs.iloc[[5]].copy()
    dup["timestamp_us"] = dup["timestamp_us"] + np.uint64(1_200)
    dup[column] = change if column == "label_class" else dup[column] + change
    pd.concat([obs, dup]).to_parquet(d / "obstacle.offline.parquet", index=False)
    return d


def test_far_duplicates_get_a_range_allowance(tmp_path):
    # 0.8 m apart at 200 m: limit 0.2 + 0.005 x 200 = 1.2 m.
    d = write_far_duplicate(tmp_path, c0=(-200.0, 0.0, 1.0))
    pair = pd.read_parquet(d / "obstacle.offline.parquet").iloc[[5, -1]]
    scene = build_scene(d, "offline", lidar=False)
    report = scene.report["duplicate_check"]
    assert scene.report["duplicates_merged"] == 1
    assert report["pairs_over_flat_limit"] == 1
    assert report["largest_offset_m"] == pytest.approx(0.8, abs=1e-6)
    assert report["largest_offset_range_m"] == pytest.approx(np.hypot(*pair[["center_x", "center_y"]].mean()), abs=0.05)
    assert scene.meta["rules"]["dedup_offset_per_m"] == 0.005


def test_near_duplicates_still_stop_the_build(tmp_path):
    # 0.8 m apart at 10 m: limit 0.25 m.
    with pytest.raises(DuplicateError, match="track 228"):
        build_scene(write_far_duplicate(tmp_path, c0=(10.0, 0.0, 1.0)), "offline", lidar=False)


@pytest.mark.parametrize("column, change", [("label_class", "automobile"), ("size_x", 0.5)])
def test_class_or_size_mismatch_stops_the_build_at_any_range(tmp_path, column, change):
    with pytest.raises(DuplicateError, match="track 228"):
        build_scene(write_far_duplicate(tmp_path, (-200.0, 0.0, 1.0), column, change), "offline", lidar=False)


def test_write_scene_layout_and_meta(tmp_path):
    center, yaw, _ = mover()
    write_clip(tmp_path / "raw", {"7": ("automobile", clock(0, 2_000_000), center, yaw, (4, 2, 1.5))})
    out = write_scene(build_scene(tmp_path / "raw" / CLIP, "online", lidar=False), tmp_path / "scene")
    assert out == tmp_path / "scene" / CLIP / "online"
    meta = json.loads((out / "meta.json").read_text())
    assert meta["ego_source"] == "online" and meta["rate_hz"] == 10 and meta["t0_us"] == 0
    assert meta["n_ticks"] == len(pd.read_parquet(out / "ego.parquet"))
    assert meta["ego"]["footprint"]["x_min"] == pytest.approx(1.312 - 4.688 / 2)
    assert meta["ego"]["rear_axle_offset"] == [0.0, 0.0, 0.0]
    assert set(meta["inputs"]["files"]) == {"egomotion.parquet", "obstacle.offline.parquet", "vehicle_dimensions.parquet"}
    objects = pd.read_parquet(out / "objects.parquet")
    assert objects["lidar_points"].isna().all() and objects["in_scene"].isna().all()  # T008
