"""T008 lidar observability: box counts, raw-point placement, sweep timing, and an end-to-end run
against a fake lidar that sweeps like the real one."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

import horizon_pipeline.scene.lidar as lidar
from horizon_pipeline.scene.build import build_scene
from horizon_pipeline.scene.frames import Poses
from horizon_pipeline.scene.lidar import LidarRules, Spin, assign_spins, count_in_box, points_to_world, sweep_profile
from test_scene import CLIP, clock, ego_at, mover, write_clip

RULES = LidarRules(lidar_min_points=5, lidar_box_margin_m=0.2, ground_clearance_m=0.2)
SWEEP_START_DEG = 178.0  # the real sweep starts facing backward and turns clockwise (T005)
EXT_XYZ = np.array([1.0, 0.0, 1.9])
EXT_ROT = Rotation.from_euler("z", 2, degrees=True)


def sweep_us(az_deg):
    return ((SWEEP_START_DEG - np.asarray(az_deg)) % 360) / 360 * 100_000


# ---------------------------------------------------------------- counting


def test_box_count_with_margin_and_ground_band():
    center, size = np.array([10.0, 5.0, 1.0]), np.array([4.0, 2.0, 1.5])  # bottom at z = 0.25
    pts = np.array([
        [10.0, 5.0, 1.0],    # center: in
        [11.9, 5.0, 1.0],    # inside the box near its end: in
        [12.15, 5.0, 1.0],   # in the 0.2 m margin: in
        [12.25, 5.0, 1.0],   # beyond the margin: out
        [10.0, 6.15, 1.0],   # side margin: in
        [10.0, 5.0, 1.9],    # in the top margin (top at 1.75): in
        [10.0, 5.0, 2.0],    # above the top margin: out
        [10.0, 5.0, 0.35],   # inside the box but within 0.2 m of its bottom (road): out
        [10.0, 5.0, 0.5],    # 0.25 m above the bottom: in
        [10.0, 5.0, 0.1],    # below the box: out
    ])  # fmt: skip
    assert count_in_box(pts, center, Rotation.identity(), size, RULES) == 6


def test_rotated_box_counts_its_corners():
    center, size, yaw = np.array([-3.0, 7.0, 0.0]), np.array([4.0, 2.0, 2.0]), np.radians(30)
    rot = Rotation.from_euler("z", yaw)
    hx, hy = size[0] / 2 + 0.2, size[1] / 2 + 0.2
    corners = np.array([[sx * hx, sy * hy, 0.5] for sx in (-1, 1) for sy in (-1, 1)])
    just_in = rot.apply(corners * [0.99, 0.99, 1]) + center
    just_out = rot.apply(corners * [1.01, 1.01, 1]) + center
    # The same points without the rotation would be outside (the corners swing by ~1 m).
    assert count_in_box(just_in, center, rot, size, RULES) == 4
    assert count_in_box(just_out, center, rot, size, RULES) == 0
    assert count_in_box(just_in, center, Rotation.identity(), size, RULES) < 4


# ---------------------------------------------------------------- placement and timing


def test_raw_points_are_placed_with_the_pose_at_their_own_time():
    t = np.arange(0, 200_001, 10_000)
    poses = Poses(t, np.c_[10.0 * t / 1e6, np.zeros((len(t), 2))], Rotation.from_euler("z", (0.3 * t / 1e6)[:, None]).as_quat())
    wall = np.array([20.0, 4.0, 1.0])  # one static point, seen at 50 times during a spin
    times = np.linspace(20_000, 120_000, 50).astype(np.int64)
    p, r = poses.at(times)
    sensor = EXT_ROT.inv().apply(r.inv().apply(wall - p) - EXT_XYZ)
    spin = Spin(0, 20_000, 120_000, sensor, times)
    world = points_to_world(spin, poses, EXT_XYZ, EXT_ROT)
    assert np.abs(world - wall).max() < 1e-3
    # One pose for the whole spin would smear it by up to v x 100 ms = 1 m (plus the turn).
    p0, r0 = poses.at([spin.start])
    assert np.linalg.norm(r0.apply(EXT_ROT.apply(sensor) + EXT_XYZ) + p0 - wall, axis=1).max() > 0.9


def test_sweep_profile_recovers_a_clockwise_sweep():
    az = np.arange(-180, 180, 0.25)
    xyz = np.c_[np.cos(np.radians(az)), np.sin(np.radians(az)), np.zeros(len(az))] * 20
    spin = Spin(0, 5_000, 105_000, xyz, (5_000 + sweep_us(az)).astype(np.int64))
    profile = sweep_profile(spin)
    centers = np.arange(360) + 0.5
    expected = sweep_us(np.where(centers >= 180, centers - 360, centers))
    seam = np.abs(((centers - SWEEP_START_DEG + 180) % 360) - 180) < 1.5  # bins the seam splits
    assert np.abs(profile - expected)[~seam].max() < 400  # us


def test_assign_spins_takes_the_latest_crossing_at_or_before_the_tick():
    """T021: a sensor reading is causal. Crossings here are at 70, 170, 270 ms, ..."""
    starts = np.arange(0, 1_000_000, 100_000)
    k, crossing = assign_spins(np.array([150_000, 100_000, 170_000, 169_999, 0]), np.full(5, 70_000.0), starts)
    assert k.tolist() == [0, 0, 1, 0, -1]  # 170 ms is nearer to 150 ms, but has not happened yet
    assert crossing.tolist() == [70_000, 70_000, 170_000, 70_000, -1]  # nothing crossed by 0 ms


# ---------------------------------------------------------------- end to end


class FakeLidar:
    """Spins every 100 ms from 13 ms. Each holds a ring of points 30 m away and 5 m up, above every
    box (so the sweep timing can be measured) and 40 points on the "seen" object, taken when the sweep crosses it;
    the object moves, so a box at the wrong time would miss them."""

    N_OBJECT_POINTS = 40

    def __init__(self, path, seen):
        self.seen = seen if isinstance(seen, list) else [seen]  # each: world center fn(t_s), yaw, size
        self.starts = np.arange(13_000, 2_850_000, 100_000)
        self.ends = self.starts + 100_000

    def __len__(self):
        return len(self.starts)

    def _to_sensor(self, world, t_us):
        p, yaw = ego_at(t_us / 1e6)
        r = Rotation.from_euler("z", yaw[0])
        return EXT_ROT.inv().apply(r.inv().apply(world - p[0]) - EXT_XYZ)

    def spin(self, i):
        start = int(self.starts[i])
        az = np.arange(-180, 180, 0.5)
        ring = np.c_[np.cos(np.radians(az)), np.sin(np.radians(az)), np.full(len(az), 5.0)] * [30, 30, 1]
        ring_t = (start + sweep_us(az)).astype(np.int64)
        xyz, t = [ring], [ring_t]
        rng = np.random.default_rng(i)
        for center_fn, yaw, size in self.seen:
            # When does the sweep cross the object? Its bearing from the sensor barely moves in 0.1 s.
            az_obj = np.degrees(np.arctan2(*self._to_sensor(center_fn((start + 50_000) / 1e6), start + 50_000)[[1, 0]]))
            t_hit = int(start + sweep_us(az_obj))
            local = (rng.random((self.N_OBJECT_POINTS, 3)) - 0.5) * size * [0.8, 0.8, 0.5] + [0, 0, 0.1]
            world = Rotation.from_euler("z", yaw).apply(local) + center_fn(t_hit / 1e6)
            xyz.append(self._to_sensor(world, t_hit))
            t.append(np.full(self.N_OBJECT_POINTS, t_hit))
        return Spin(i, start, start + 100_000, np.concatenate(xyz), np.concatenate(t))


class TruncatedLidar(FakeLidar):
    """FakeLidar with every point recorded after `cut_us` removed: what a live car has at `cut_us`."""

    def __init__(self, path, seen, cut_us):
        super().__init__(path, seen)
        self.cut_us = cut_us

    def spin(self, i):
        s = super().spin(i)
        keep = s.t <= self.cut_us
        return Spin(s.index, s.start, s.end, s.xyz[keep], s.t[keep])


def lidar_clip(tmp_path, tracks):
    d = write_clip(tmp_path, tracks)
    ext = pd.DataFrame(
        [list(EXT_ROT.as_quat()) + list(EXT_XYZ)], columns=["qx", "qy", "qz", "qw", "x", "y", "z"],
        index=pd.MultiIndex.from_tuples([(CLIP, "lidar_top_360fov")], names=["clip_id", "sensor_name"]),
    )  # fmt: skip
    ext.to_parquet(d / "sensor_extrinsics.offline.parquet")
    (d / "lidar_top_360fov").mkdir()
    (d / "lidar_top_360fov" / "pointclouds.parquet").write_bytes(b"")  # presence only; the reader is faked
    return d


def test_readings_use_no_point_after_the_tick(tmp_path, monkeypatch):
    """T021: removing every point recorded after 2.0 s changes no reading at ticks up to 2.0 s.
    Objects all around the car are crossed at every phase of the spin, so some crossings fall just
    after a tick (which the old nearest-crossing rule would have used)."""
    size = np.array([4.0, 1.8, 1.5])
    seen = [
        (mover(v=(6.0, 3.0, 0.0), c0=(15 * np.cos(a), 15 * np.sin(a), 0.8), yaw=0.4)[0], 0.4, size)
        for a in np.radians(np.arange(0, 360, 30))
    ]
    tracks = {f"o{n}": ("automobile", clock(37_000, 2_937_000), c, lambda t: 0.4, size) for n, (c, _, _) in enumerate(seen)}
    cols = ["lidar_points", "observed", "lidar_spin", "lidar_dt_s"]
    out = {}
    for name, reader in {"all": lambda p: FakeLidar(p, seen), "cut": lambda p: TruncatedLidar(p, seen, 2_000_000)}.items():
        monkeypatch.setattr(lidar, "SpinReader", reader)
        out[name] = build_scene(lidar_clip(tmp_path / name, tracks), "offline").objects.set_index(["tick", "track_id"])
    past = out["all"]["t_s"] <= 2.0 + 1e-9
    pd.testing.assert_frame_equal(out["all"].loc[past, cols], out["cut"].loc[past, cols])
    assert (out["cut"].loc[~past, "lidar_points"] == 0).any()  # and the cut does bite later


def test_observe_end_to_end(tmp_path, monkeypatch):
    seen_yaw, hidden_yaw = 0.4, 0.0
    seen_center, _, _ = mover(v=(6.0, 3.0, 0.0), c0=(15.0, -8.0, 0.8), yaw=seen_yaw)
    hidden_center, _, _ = mover(v=(-3.0, 0.0, 0.0), c0=(-20.0, 12.0, 0.8), yaw=hidden_yaw)
    size = np.array([4.0, 1.8, 1.5])
    d = lidar_clip(tmp_path, {
        "seen": ("automobile", clock(37_000, 2_937_000), seen_center, lambda t: seen_yaw, size),
        "hidden": ("person", clock(61_000, 2_861_000), hidden_center, lambda t: hidden_yaw, (0.6, 0.6, 1.7)),
        "early": ("person", clock(0, 500_000), hidden_center, lambda t: hidden_yaw, (0.6, 0.6, 1.7)),
    })  # fmt: skip
    monkeypatch.setattr(lidar, "SpinReader", lambda path: FakeLidar(path, (seen_center, seen_yaw, size)))

    scene = build_scene(d, "offline")
    o = scene.objects
    # Causal (T021): a row's reading is the latest crossing at or before the tick. At tick 0 no spin
    # (the first starts at 13 ms) has crossed "early" yet, so it has no reading and is not observed.
    read = o["lidar_spin"].notna()
    unread = o[~read]
    assert unread[["track_id", "tick"]].values.tolist() == [["early", 0]]
    assert not unread["observed"].any() and unread["lidar_points"].isna().all() and unread["lidar_dt_s"].isna().all()
    seen, hidden = o[o["track_id"] == "seen"], o[read & (o["track_id"] != "seen")]
    assert (seen["lidar_points"] == FakeLidar.N_OBJECT_POINTS).all()
    assert seen["observed"].all() and seen["in_scene"].all()
    assert (hidden["lidar_points"] == 0).all() and not hidden["in_scene"].any()
    assert o.loc[read, "lidar_dt_s"].between(-0.1, 0).all()
    assert scene.meta["rules"]["lidar_min_points"] == 5
    assert scene.report["lidar"]["observed_share"] == pytest.approx(len(seen) / len(o))
    assert scene.report["lidar"]["rows_without_reading"] == 1
