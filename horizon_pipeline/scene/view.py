"""Rerun view of a built scene (T010): the scene in 3D next to the real camera videos, with the
scene's boxes drawn on the video.

Reads only the scene files, plus the raw lidar, videos and calibration for display. Entities:

    world/ego                     ego pose per tick (transform); box, heading arrow, rear-axle point
    world/ego_path                the whole ego path (static)
    world/ego_<source>            with --compare: the other ego source's box and path
    world/obstacles/<track_id>    one box per existing object; colored by class group when in_scene,
                                  gray and faded when it exists but the lidar does not see it, and
                                  faded beyond `fade_beyond_m` if set; labels shown on selection
    world/lidar                   the spin nearest each tick, placed point by point (T008)
    cameras/<camera>/image        video frames on the same timeline
    cameras/<camera>/boxes        the same boxes, projected into that camera

Timeline: `clip_time`, in seconds from the scene's t0.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation

from horizon_pipeline.scene.build import load_ego
from horizon_pipeline.scene.camera import ALPAMAYO_CAMERAS, Camera, load_camera, video_frames
from horizon_pipeline.scene.frames import QUAT, XYZ, slerp_pairs
from horizon_pipeline.scene.lidar import POINTCLOUDS, SpinReader, lidar_extrinsic, points_to_world

TIMELINE = "clip_time"
CLASS_COLORS = {"vehicle": (60, 140, 255), "vru": (255, 160, 20), "static": (40, 200, 120), "unknown": (220, 60, 220)}
UNSEEN_COLOR = (150, 150, 150, 90)  # exists, but the lidar does not see it
FAR_ALPHA = 50  # objects beyond `fade_beyond_m`
EGO_COLOR = (255, 255, 255)
COMPARE_COLOR = (255, 60, 60)
SIZE = ["length", "width", "height"]
CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
EDGE_SAMPLES = 16  # points per projected edge, so the fisheye curvature shows


@dataclass
class ViewOptions:
    lidar: bool = True
    lidar_every: int = 10  # keep every n-th lidar point (~34 000 of ~335 000 per spin)
    cameras: list[str] = field(default_factory=lambda: list(ALPAMAYO_CAMERAS))
    all_frames: bool = False  # every video frame (30 Hz) instead of the one nearest each tick
    image_scale: float = 0.5
    compare: bool = False  # also draw the other ego source's box and path
    labels: bool = False  # draw every box's label (otherwise shown when a box is selected)
    fade_beyond_m: float | None = None  # fade objects farther than this from the ego rear axle


def load_scene(scene_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    scene_dir = Path(scene_dir)
    meta = json.loads((scene_dir / "meta.json").read_text())
    return meta, pd.read_parquet(scene_dir / "ego.parquet"), pd.read_parquet(scene_dir / "objects.parquet")


def clip_time(rec: rr.RecordingStream, t_s: float) -> None:
    rec.set_time(TIMELINE, duration=float(t_s))


def object_color(row, fade_beyond_m: float | None = None) -> tuple:
    """Class color when in the scene, or when the scene was built without lidar (`in_scene` empty);
    faded when farther than `fade_beyond_m`."""
    seen = pd.isna(row.in_scene) or bool(row.in_scene)
    color = CLASS_COLORS.get(row.class_group, CLASS_COLORS["unknown"]) if seen else UNSEEN_COLOR
    if fade_beyond_m is not None and row.distance > fade_beyond_m:
        return (*color[:3], FAR_ALPHA)
    return color


def object_label(row) -> str:
    points = "?" if pd.isna(row.lidar_points) else int(row.lidar_points)
    return f"{row.track_id} {row.label_class} {row.speed:.1f} m/s {points} pts"


# ---------------------------------------------------------------- 3D


def log_ego(rec: rr.RecordingStream, meta: dict, ego: pd.DataFrame, path: str = "world/ego", color=EGO_COLOR) -> None:
    fp = meta["ego"]["footprint"]
    lo = np.array([fp["x_min"], fp["y_min"], fp["z_min"]])
    hi = np.array([fp["x_max"], fp["y_max"], fp["z_max"]])
    rec.log(f"{path}/box", rr.Boxes3D(mins=[lo], sizes=[hi - lo], colors=[color]), static=True)
    rec.log(f"{path}/heading", rr.Arrows3D(origins=[[0, 0, 0.1]], vectors=[[hi[0] + 1.0, 0, 0]], colors=[color]), static=True)
    rec.log(f"{path}/rear_axle", rr.Points3D([[0, 0, 0]], radii=[0.08], colors=[color]), static=True)
    rec.log(f"{path}_path", rr.LineStrips3D([ego[XYZ].to_numpy()], colors=[color]), static=True)
    for row in ego.itertuples():
        clip_time(rec, row.t_s)
        rec.log(path, rr.Transform3D(translation=[row.x, row.y, row.z], quaternion=[row.qx, row.qy, row.qz, row.qw]))


def log_objects(rec: rr.RecordingStream, objects: pd.DataFrame, ticks: pd.Series, opts: "ViewOptions") -> None:
    """One entity per track, cleared at ticks where the track does not exist."""
    by_tick = dict(tuple(objects.groupby("tick")))
    alive: set[str] = set()
    for tick, t_s in zip(ticks.index, ticks):
        clip_time(rec, t_s)
        rows = by_tick.get(tick, objects.iloc[:0])
        for gone in alive - set(rows["track_id"]):
            rec.log(f"world/obstacles/{gone}", rr.Clear(recursive=False))
        for row in rows.itertuples():
            rec.log(
                f"world/obstacles/{row.track_id}",
                rr.Boxes3D(
                    centers=[[row.x, row.y, row.z]],
                    sizes=[[row.length, row.width, row.height]],
                    quaternions=[[row.qx, row.qy, row.qz, row.qw]],
                    colors=[object_color(row, opts.fade_beyond_m)],
                    labels=[object_label(row)],
                    show_labels=opts.labels,
                ),
            )
        alive = set(rows["track_id"])


def height_colors(z: np.ndarray) -> np.ndarray:
    """Dark blue at the road to pale yellow at 4 m above it."""
    f = np.clip((z - np.percentile(z, 5)) / 4.0, 0, 1)[:, None]
    return ((1 - f) * [30, 60, 160] + f * [250, 240, 160]).astype(np.uint8)


def log_lidar(rec: rr.RecordingStream, clip_dir: Path, ego_source: str, t0_us: int, ticks: pd.Series, every: int) -> None:
    """The spin whose middle is nearest each tick, every point placed with the ego pose at its own
    time (the same placement T008 counts with)."""
    reader = SpinReader(clip_dir / POINTCLOUDS)
    poses = load_ego(clip_dir, ego_source)
    ext_xyz, ext_rot = lidar_extrinsic(clip_dir)
    usable = np.flatnonzero((reader.starts >= poses.start) & (reader.ends <= poses.end))
    middles = (reader.starts[usable] + reader.ends[usable]) / 2
    for t_s in ticks:
        s = int(usable[np.abs(middles - (t0_us + t_s * 1e6)).argmin()])
        world = points_to_world(reader.spin(s), poses, ext_xyz, ext_rot)[::every]
        clip_time(rec, t_s)
        rec.log("world/lidar", rr.Points3D(world, colors=height_colors(world[:, 2]), radii=[0.03]))


# ---------------------------------------------------------------- cameras


def state_at(ego: pd.DataFrame, objects: pd.DataFrame, t_s: float):
    """Ego pose and object boxes at a time between ticks: position linear, rotation slerp, as in the
    scene build. Objects in only one of the two ticks keep that tick's box; `in_scene` comes from
    the nearer tick."""
    k = int(np.clip(np.floor(t_s * 10 + 1e-9), 0, len(ego) - 1))
    k1 = min(k + 1, len(ego) - 1)
    frac = 0.0 if k1 == k else float(np.clip((t_s - ego["t_s"].iloc[k]) / (ego["t_s"].iloc[k1] - ego["t_s"].iloc[k]), 0, 1))
    e0, e1 = ego.iloc[k], ego.iloc[k1]
    ego_xyz = (1 - frac) * e0[XYZ].to_numpy(float) + frac * e1[XYZ].to_numpy(float)
    r0 = Rotation.from_quat(e0[QUAT].to_numpy(float)[None])
    ego_rot = slerp_pairs(r0, Rotation.from_quat(e1[QUAT].to_numpy(float)[None]), np.array([frac]))[0]

    a = objects[objects["tick"] == k].set_index("track_id")
    b = objects[objects["tick"] == k1].set_index("track_id")
    near, far = (a, b) if frac < 0.5 else (b, a)
    boxes = near.astype({c: float for c in XYZ + QUAT})
    both = a.index.intersection(b.index)
    if len(both) and k1 != k:
        boxes.loc[both, XYZ] = (1 - frac) * a.loc[both, XYZ].to_numpy() + frac * b.loc[both, XYZ].to_numpy()
        rot = slerp_pairs(Rotation.from_quat(a.loc[both, QUAT].to_numpy()), Rotation.from_quat(b.loc[both, QUAT].to_numpy()), np.full(len(both), frac))
        boxes.loc[both, QUAT] = rot.as_quat()
    only_far = far.index.difference(near.index)
    boxes = pd.concat([boxes, far.loc[only_far]])
    return ego_xyz, ego_rot, boxes.reset_index()


def box_strips(cam: Camera, ego_xyz, ego_rot, boxes: pd.DataFrame, scale: float, fade_beyond_m: float | None = None):
    """Projected box edges as 2D line strips, cut where they leave the camera's field of view."""
    strips, colors, labels = [], [], []
    t = np.linspace(0, 1, EDGE_SAMPLES)[:, None]
    for row in boxes.itertuples():
        corners = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).apply(CORNER_SIGNS * np.array([row.length, row.width, row.height]) / 2)
        corners += [row.x, row.y, row.z]
        edges = np.concatenate([corners[i] + t * (corners[j] - corners[i]) for i, j in BOX_EDGES])
        ray = cam.rig_to_camera(ego_rot.inv().apply(edges - ego_xyz))
        uv, ok = cam.model.project(ray)
        ok &= ray[:, 2] > 0.1
        uv = uv * scale
        drawn = False
        for e in range(len(BOX_EDGES)):
            sl = slice(e * EDGE_SAMPLES, (e + 1) * EDGE_SAMPLES)
            for run in np.split(np.arange(EDGE_SAMPLES), np.flatnonzero(np.diff(ok[sl].astype(int))) + 1):
                if ok[sl][run[0]] and len(run) > 1:
                    strips.append(uv[sl][run])
                    colors.append(object_color(row, fade_beyond_m))
                    labels.append(None if drawn else str(row.track_id))
                    drawn = True
    return strips, colors, labels


def log_camera(rec: rr.RecordingStream, cam: Camera, t0_us: int, ego: pd.DataFrame, objects: pd.DataFrame, opts: ViewOptions) -> None:
    frame_s = (cam.frame_us - t0_us) / 1e6
    inside = (frame_s >= ego["t_s"].iloc[0]) & (frame_s <= ego["t_s"].iloc[-1])
    if opts.all_frames:
        wanted = set(np.flatnonzero(inside))
    else:
        wanted = {int(np.abs(frame_s - t).argmin()) for t in ego["t_s"]} & set(np.flatnonzero(inside))
    for i, image in video_frames(cam.video, scale=opts.image_scale):
        if i not in wanted:
            continue
        clip_time(rec, frame_s[i])
        rec.log(f"cameras/{cam.name}/image", rr.Image(image).compress(jpeg_quality=80))
        strips, colors, labels = box_strips(cam, *state_at(ego, objects, frame_s[i]), opts.image_scale, opts.fade_beyond_m)
        boxes = rr.LineStrips2D(strips, colors=colors, labels=labels, show_labels=opts.labels, radii=[1.0])
        rec.log(f"cameras/{cam.name}/boxes", boxes if strips else rr.Clear(recursive=False))


# ---------------------------------------------------------------- layout and entry point


def blueprint(cameras: list[str]) -> rrb.Blueprint:
    world = rrb.Spatial3DView(
        origin="world", name="scene", eye_controls=rrb.EyeControls3D(tracking_entity="world/ego"),
    )  # fmt: skip
    if not cameras:
        return rrb.Blueprint(world, collapse_panels=True)
    views = [rrb.Spatial2DView(origin=f"cameras/{c}", name=c) for c in cameras]
    front = [v for v, c in zip(views, cameras) if "front" in c]
    sides = [v for v, c in zip(views, cameras) if "front" not in c]
    column = rrb.Vertical(*front, rrb.Horizontal(*sides)) if front and sides else rrb.Vertical(*views)
    return rrb.Blueprint(rrb.Horizontal(world, column, column_shares=[1, 1]), collapse_panels=True)


def log_scene(rec: rr.RecordingStream, scene_dir: Path, clip_dir: Path, opts: ViewOptions = ViewOptions()) -> dict:
    """Logs everything to `rec`; returns the scene's meta."""
    meta, ego, objects = load_scene(scene_dir)
    ego = ego.set_index("tick", drop=False)
    log_ego(rec, meta, ego)
    if opts.compare:
        other = next(s for s in ("offline", "online") if s != meta["ego_source"])
        other_meta, other_ego, _ = load_scene(Path(scene_dir).parent / other)
        log_ego(rec, other_meta, other_ego, path=f"world/ego_{other}", color=COMPARE_COLOR)
    log_objects(rec, objects, ego["t_s"], opts)
    if opts.lidar:
        log_lidar(rec, clip_dir, meta["ego_source"], meta["t0_us"], ego["t_s"], opts.lidar_every)
    for name in opts.cameras:
        log_camera(rec, load_camera(clip_dir, name), meta["t0_us"], ego, objects, opts)
    return meta
