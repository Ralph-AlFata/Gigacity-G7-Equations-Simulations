"""T010 Rerun view: f-theta projection, boxes drawn into a camera, interpolation between ticks, and
an end-to-end recording of a synthetic clip read back from the .rrd file."""

import json
import os
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest
from physical_ai_av.utils.camera_models import FThetaCameraModel
from rerun.chunk import RrdReader
from scipy.spatial.transform import Rotation

from horizon_pipeline.scene.build import build_scene, write_scene
from horizon_pipeline.scene.camera import Camera, FTheta
from horizon_pipeline.scene.view import (
    CLASS_COLORS, FAR_ALPHA, UNSEEN_COLOR, ViewOptions, box_strips, ego_label, is_current, object_color, recording_path, state_at,
    write_recording,
)  # fmt: skip
from test_scene import CLIP, clock, mover, write_clip

# The front-wide camera of b789fab8 (camera_intrinsics.offline).
FRONT_WIDE = {
    "resolution": [1920, 1080], "principal_point": [957.8038940429688, 756.29541015625],
    "angle_to_pixeldist_poly": [0.0, 946.8442993164062, -16.557571411132812, 36.97359085083008, -77.86387634277344, 32.331783294677734],
    "pixeldist_to_angle_poly": [0.0, 0.0010545405093580484, 3.539619441994546e-08, -9.60976090369492e-11, 1.6493639678864802e-13, -6.889097631031536e-17],
    "max_angle": 1.3493140447168162, "linear_cde": [1.0, 0.0, 0.0],
}  # fmt: skip
# Camera axes in the rig frame: z (optical axis) = rig +x, x (image right) = rig -y, y (image down) = rig -z.
FORWARD = Rotation.from_matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
CAM_XYZ = np.array([1.7, 0.0, 1.2])


def front_camera(params=FRONT_WIDE) -> Camera:
    return Camera("camera_front_wide_120fov", FTheta.from_parameters(params), CAM_XYZ, FORWARD, None, np.array([0]))


def box_row(center, size=(4.0, 2.0, 1.5), yaw=0.0, track="a", in_scene=True):
    q = Rotation.from_euler("z", yaw).as_quat()
    return pd.DataFrame([{
        "track_id": track, "class_group": "vehicle", "in_scene": in_scene,
        "x": center[0], "y": center[1], "z": center[2], "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3],
        "length": size[0], "width": size[1], "height": size[2],
    }])  # fmt: skip


# ---------------------------------------------------------------- projection


def test_ftheta_matches_nvidia_sdk():
    sdk = FThetaCameraModel(
        width=1920, height=1080, principal_point=np.array(FRONT_WIDE["principal_point"]),
        th2r=np.polynomial.Polynomial(FRONT_WIDE["angle_to_pixeldist_poly"]),
        r2th=np.polynomial.Polynomial(FRONT_WIDE["pixeldist_to_angle_poly"]),
    )  # fmt: skip
    rays = np.random.default_rng(0).normal(size=(500, 3))
    rays[:, 2] = np.abs(rays[:, 2]) + 0.2
    uv, _ = FTheta.from_parameters(FRONT_WIDE).project(rays)
    np.testing.assert_allclose(uv, sdk.ray2pixel(rays), atol=1e-9)


def test_ftheta_axes_and_field_of_view():
    model = FTheta.from_parameters(FRONT_WIDE)
    pp = np.array(FRONT_WIDE["principal_point"])
    uv, ok = model.project(np.array([[0, 0, 1.0], [1, 0, 3.0], [0, 1, 3.0], [1, 0, 0.1], [0, 0, -1.0]]))
    np.testing.assert_allclose(uv[0], pp)
    assert uv[1, 0] > pp[0] and uv[1, 1] == pytest.approx(pp[1])  # camera x is image right
    assert uv[2, 1] > pp[1] and uv[2, 0] == pytest.approx(pp[0])  # camera y is image down
    assert ok[:3].all() and not ok[3:].any()  # 84 deg and 180 deg are outside max_angle (77 deg)


def test_box_ahead_projects_around_the_image_center():
    cam = front_camera()
    ahead = box_row(CAM_XYZ + [10.0, 0.0, 0.0])
    strips, colors, labels = box_strips(cam, np.zeros(3), Rotation.identity(), ahead, scale=1.0)
    assert len(strips) == 12  # every edge visible and unbroken
    pts = np.concatenate(strips)
    assert (pts[:, 0] > 0).all() and (pts[:, 0] < 1920).all() and (pts[:, 1] > 0).all() and (pts[:, 1] < 1080).all()
    np.testing.assert_allclose(pts.mean(0), FRONT_WIDE["principal_point"], atol=5)
    assert labels[0] == "a" and labels[1:] == [None] * 11  # one label per box
    assert box_strips(cam, np.zeros(3), Rotation.identity(), box_row(CAM_XYZ - [12.0, 0, 0]), 1.0)[0] == []


def test_boxes_use_the_ego_pose():
    """The same box straight ahead of an ego that is moved and turned by 90 degrees."""
    cam = front_camera()
    ego_xyz, ego_rot = np.array([100.0, 50.0, 2.0]), Rotation.from_euler("z", 90, degrees=True)
    world = ego_rot.apply(CAM_XYZ + [10.0, 0, 0]) + ego_xyz
    strips, _, _ = box_strips(cam, ego_xyz, ego_rot, box_row(world, yaw=np.pi / 2), scale=0.5)
    np.testing.assert_allclose(np.concatenate(strips).mean(0), np.array(FRONT_WIDE["principal_point"]) * 0.5, atol=3)


def test_box_crossing_the_field_of_view_is_cut_not_wrapped():
    """A long box beside the camera: edges leaving the 77 deg field of view are cut into pieces."""
    cam = front_camera()
    side = box_row(CAM_XYZ + [1.0, -3.0, 0.0], size=(12.0, 2.0, 1.5))
    strips, _, _ = box_strips(cam, np.zeros(3), Rotation.identity(), side, scale=1.0)
    assert 0 < len(strips)
    pts = np.concatenate(strips)
    assert (pts[:, 0] > FRONT_WIDE["principal_point"][0]).all()  # all on the right half (rig -y)


def test_colors_fade_beyond_the_chosen_distance():
    near, far = box_row([5, 0, 0]).assign(distance=8.0), box_row([50, 0, 0]).assign(distance=45.0)
    unseen_far = box_row([50, 0, 0], in_scene=False).assign(distance=45.0)
    color = lambda df, fade: object_color(next(df.itertuples()), fade)  # noqa: E731
    assert color(near, 30.0) == color(far, None) == CLASS_COLORS["vehicle"]
    assert color(far, 30.0) == (*CLASS_COLORS["vehicle"], FAR_ALPHA)
    assert color(unseen_far, 30.0) == (*UNSEEN_COLOR[:3], FAR_ALPHA)
    _, colors, _ = box_strips(front_camera(), np.zeros(3), Rotation.identity(), box_row(CAM_XYZ + [40.0, 0, 0]).assign(distance=41.7), 1.0, 30.0)
    assert {c[3] for c in colors} == {FAR_ALPHA}


# ---------------------------------------------------------------- between ticks


def test_state_at_interpolates_between_ticks():
    ego = pd.DataFrame({"tick": [0, 1], "t_s": [0.0, 0.1], "x": [0.0, 1.0], "y": 0.0, "z": 0.0})
    ego[["qx", "qy", "qz", "qw"]] = Rotation.from_euler("z", [[0.0], [0.2]]).as_quat()
    both0, both1 = box_row([10, 0, 0], track="both"), box_row([12, 2, 0], yaw=0.4, track="both")
    first = box_row([5, 5, 0], track="first", in_scene=False)
    second = box_row([7, 7, 0], track="second")
    objects = pd.concat([both0.assign(tick=0), first.assign(tick=0), both1.assign(tick=1), second.assign(tick=1)])
    xyz, rot, boxes = state_at(ego, objects, 0.03)
    np.testing.assert_allclose(xyz, [0.3, 0, 0], atol=1e-12)
    assert rot.as_euler("xyz")[2] == pytest.approx(0.06)
    boxes = boxes.set_index("track_id")
    np.testing.assert_allclose(boxes.loc["both", ["x", "y"]].to_numpy(float), [10.6, 0.6])
    assert Rotation.from_quat(boxes.loc["both", ["qx", "qy", "qz", "qw"]].to_numpy(float)).as_euler("xyz")[2] == pytest.approx(0.12)
    assert set(boxes.index) == {"both", "first", "second"}  # each kept with its only box
    np.testing.assert_allclose(boxes.loc["first", ["x", "y"]].to_numpy(float), [5, 5])


# ---------------------------------------------------------------- end to end


def write_camera(clip_dir, name="camera_front_wide_120fov", n=31, size=(192, 108)):
    """A 10 Hz test video, its frame timestamps, and calibration scaled to its size."""
    (clip_dir / name).mkdir()
    with av.open(str(clip_dir / name / "video.mp4"), "w") as out:
        stream = out.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        for i in range(n):
            img = np.full((size[1], size[0], 3), (i * 8) % 256, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    pd.DataFrame({"timestamp": np.arange(n) * 100_000 + 1_500}, index=pd.Index(range(n), name="frame_index")).to_parquet(
        clip_dir / name / "frame_timestamps.parquet"
    )
    params = {**FRONT_WIDE, "resolution": list(size), "principal_point": [size[0] / 2, size[1] / 2],
              "angle_to_pixeldist_poly": [0.0, 90.0]}  # fmt: skip
    index = pd.MultiIndex.from_tuples([(CLIP, name)], names=["clip_id", "camera_name"])
    pd.DataFrame({"model_type": ["ftheta"], "model_parameters": [json.dumps(params)]}, index=index).to_parquet(
        clip_dir / "camera_intrinsics.offline.parquet"
    )
    q = FORWARD.as_quat()
    ext_index = pd.MultiIndex.from_tuples([(CLIP, name)], names=["clip_id", "sensor_name"])
    pd.DataFrame([[*q, *CAM_XYZ]], columns=["qx", "qy", "qz", "qw", "x", "y", "z"], index=ext_index).to_parquet(
        clip_dir / "sensor_extrinsics.offline.parquet"
    )
    return name


def test_log_scene_recording(tmp_path):
    long_center, long_yaw, _ = mover(v=(9.0, -1.5, 0.0), c0=(15.0, 0.0, 0.8), yaw=0.0)
    short_center, short_yaw, _ = mover(v=(0.0, 0.0, 0.0), c0=(30.0, 8.0, 0.8), yaw=1.0)
    clip_dir = write_clip(tmp_path / "raw", {
        "long": ("automobile", clock(0, 3_000_000), long_center, long_yaw, (4.0, 1.8, 1.5)),
        "short": ("person", clock(61_000, 1_061_000), short_center, short_yaw, (0.6, 0.6, 1.7)),
    })  # fmt: skip
    for source in ("offline", "online"):
        write_scene(build_scene(clip_dir, source, lidar=False), tmp_path / "scene")
    camera = write_camera(clip_dir)

    path = tmp_path / "recordings" / "view.rrd"
    progress = []
    meta = write_recording(path, tmp_path / "scene" / CLIP / "offline", clip_dir,
                           ViewOptions(lidar=False, cameras=[camera], compare=True), progress.append)  # fmt: skip
    assert [p.split()[0] for p in progress] == ["ego", camera]  # one line per source
    assert sorted(f.name for f in path.parent.iterdir()) == ["view.rrd"]  # no .partial left behind

    store = RrdReader(path).stream().collect()
    entities = set(store.schema().entity_paths())
    assert {
        "/world/ego", "/world/ego/box", "/world/ego/heading", "/world/ego/rear_axle", "/world/ego_path",
        "/world/ego_online", "/world/ego_online_path", "/world/obstacles/long", "/world/obstacles/short",
        f"/cameras/{camera}/image", f"/cameras/{camera}/boxes",
        *(f"/plots/{e}/{q}" for e in ("ego", "ego_online") for q in ("speed", "a_long", "a_lat", "yaw_rate")),
    } <= entities  # fmt: skip
    summary = {}
    for line in store.summary().splitlines():
        entity, rows = line.split()[0], int(line.split("rows=")[1].split()[0])
        summary.setdefault(entity, []).append((rows, line))
    rows = lambda e: sum(r for r, _ in summary[e])  # noqa: E731
    assert rows("/world/ego") == rows("/world/ego_online") == meta["n_ticks"]  # one transform per tick
    # T022: a label per tick (plus the static box), and plots that skip the ticks without a value
    # yet (T009): speed and yaw rate from tick 1, accelerations from tick 2.
    assert rows("/world/ego/box") == rows("/world/ego_online/box") == meta["n_ticks"] + 1
    for e in ("ego", "ego_online"):  # each also has one static row: the series style
        assert rows(f"/plots/{e}/speed") == rows(f"/plots/{e}/yaw_rate") == (meta["n_ticks"] - 1) + 1
        assert rows(f"/plots/{e}/a_long") == rows(f"/plots/{e}/a_lat") == (meta["n_ticks"] - 2) + 1
    assert any("Clear" in line for _, line in summary["/world/obstacles/short"])  # removed when its track ends
    assert not any("Clear" in line for _, line in summary["/world/obstacles/long"])
    # One frame per tick, except the last: that frame (3.0015 s) is after the scene ends (3.0 s).
    assert rows(f"/cameras/{camera}/image") == rows(f"/cameras/{camera}/boxes") == meta["n_ticks"] - 1


def test_ego_label_shows_speed_and_acceleration():
    row = pd.Series({"speed": 7.0987, "a_long": 0.657}).to_frame().T.itertuples().__next__()
    assert ego_label(row) == "ego 7.1 m/s, a_long +0.7 m/s²"
    first = pd.Series({"speed": np.nan, "a_long": np.nan}).to_frame().T.itertuples().__next__()
    assert ego_label(first, "ego online") == "ego online ? m/s, a_long ? m/s²"


def test_recording_path_changes_with_the_recording_version(monkeypatch):
    from horizon_pipeline.scene import view

    a = recording_path(Path("r"), CLIP, "offline", ViewOptions())
    monkeypatch.setattr(view, "RECORDING_VERSION", view.RECORDING_VERSION + 1)
    assert recording_path(Path("r"), CLIP, "offline", ViewOptions()) != a  # old recordings are not reused


def test_recording_path_depends_on_the_options():
    a = recording_path(Path("r"), CLIP, "offline", ViewOptions())
    assert a == recording_path(Path("r"), CLIP, "offline", ViewOptions())
    assert a.parent == Path("r") / CLIP and a.name.startswith("offline-") and a.suffix == ".rrd"
    assert a != recording_path(Path("r"), CLIP, "offline", ViewOptions(fade_beyond_m=30.0))
    assert a != recording_path(Path("r"), CLIP, "online", ViewOptions())


def test_a_recording_is_stale_once_the_scene_is_rebuilt(tmp_path):
    scenes = tmp_path / "scene" / CLIP
    for source in ("offline", "online"):
        (scenes / source).mkdir(parents=True)
        (scenes / source / "objects.parquet").write_bytes(b"")
    rrd = tmp_path / "x.rrd"
    assert not is_current(rrd, scenes / "offline", ViewOptions())
    rrd.write_bytes(b"")
    os.utime(scenes / "offline" / "objects.parquet", (1_000, 1_000))
    os.utime(scenes / "online" / "objects.parquet", (1_000, 1_000))
    os.utime(rrd, (2_000, 2_000))
    assert is_current(rrd, scenes / "offline", ViewOptions(compare=True))
    os.utime(scenes / "online" / "objects.parquet", (3_000, 3_000))  # the other source was rebuilt
    assert is_current(rrd, scenes / "offline", ViewOptions())
    assert not is_current(rrd, scenes / "offline", ViewOptions(compare=True))
    os.utime(scenes / "offline" / "objects.parquet", (3_000, 3_000))
    assert not is_current(rrd, scenes / "offline", ViewOptions())
