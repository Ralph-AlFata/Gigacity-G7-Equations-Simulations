"""Cameras (T010): f-theta projection, camera poses and video frames, to draw the scene on the
real video.

The projection follows NVIDIA's SDK (`physical_ai_av.utils.camera_models.FThetaCameraModel`):
a ray at angle theta from the optical axis lands at radius `angle_to_pixeldist_poly(theta)` from
the principal point, in the direction of its x, y. The camera frame has z forward, x right and
y down; `sensor_extrinsics` gives it in the rig frame (checked on our clips: the front camera's
z axis is the rig's +x, its x axis the rig's -y, its y axis the rig's -z).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from horizon_pipeline.scene.lidar import EXTRINSICS

INTRINSICS = "camera_intrinsics.offline"
ALPAMAYO_CAMERAS = ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov"]


@dataclass
class FTheta:
    width: int
    height: int
    principal_point: np.ndarray
    angle_to_pixeldist: np.polynomial.Polynomial
    max_angle: float
    linear_cde: np.ndarray

    @classmethod
    def from_parameters(cls, params: dict) -> "FTheta":
        return cls(
            width=params["resolution"][0],
            height=params["resolution"][1],
            principal_point=np.array(params["principal_point"], float),
            angle_to_pixeldist=np.polynomial.Polynomial(params["angle_to_pixeldist_poly"]),
            max_angle=float(params["max_angle"]),
            linear_cde=np.array(params["linear_cde"], float),
        )

    def project(self, ray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (u right, v down, origin top-left) of camera-frame points, and whether each lies
        within the calibrated field of view (in front of the camera, angle <= max_angle)."""
        norm_xy = np.linalg.norm(ray[:, :2], axis=1)
        theta = np.arctan2(norm_xy, ray[:, 2])
        r = self.angle_to_pixeldist(theta)
        direction = ray[:, :2] / np.maximum(norm_xy, 1e-12)[:, None]
        x, y = r * direction[:, 0], r * direction[:, 1]
        c, d, e = self.linear_cde
        uv = np.c_[c * x + d * y, e * x + y] + self.principal_point
        return uv, theta <= self.max_angle


@dataclass
class Camera:
    name: str
    model: FTheta
    xyz: np.ndarray  # camera origin in the rig frame
    rot: Rotation  # camera frame -> rig frame
    video: Path
    frame_us: np.ndarray  # timestamp of each video frame

    def rig_to_camera(self, p_rig: np.ndarray) -> np.ndarray:
        return self.rot.inv().apply(p_rig - self.xyz)


def load_camera(clip_dir: Path, name: str) -> Camera:
    clip_dir = Path(clip_dir)
    intr = pd.read_parquet(clip_dir / f"{INTRINSICS}.parquet").xs(name, level="camera_name").iloc[0]
    if intr["model_type"] != "ftheta":
        raise ValueError(f"{name}: camera model {intr['model_type']!r} is not supported")
    ex = pd.read_parquet(clip_dir / f"{EXTRINSICS}.parquet").xs(name, level="sensor_name").iloc[0]
    frames = pd.read_parquet(clip_dir / name / "frame_timestamps.parquet")
    return Camera(
        name=name,
        model=FTheta.from_parameters(json.loads(intr["model_parameters"])),
        xyz=ex[["x", "y", "z"]].to_numpy(float, copy=True),
        rot=Rotation.from_quat(ex[["qx", "qy", "qz", "qw"]].to_numpy(float, copy=True)),
        video=clip_dir / name / "video.mp4",
        frame_us=frames["timestamp"].to_numpy(np.int64, copy=True),
    )


def video_frames(path: Path, every: int = 1, scale: float = 1.0):
    """Yields (frame_index, RGB array) for every `every`-th frame, resized by `scale`."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i % every:
                continue
            if scale != 1.0:
                frame = frame.reformat(width=round(frame.width * scale), height=round(frame.height * scale))
            yield i, frame.to_ndarray(format="rgb24")
