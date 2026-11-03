"""Poses and frame transforms.

Conventions (T004-T006):
- quaternions are `(qx, qy, qz, qw)`, scalar last;
- a pose `(p, R)` maps a point from its local frame to the parent frame: `x_parent = R @ x_local + p`;
- the rig origin is the rear-axle midpoint on the ground, x forward, y left, z up;
- "world" is the ego source's anchor frame (the rig at clip time 0).
"""

import numpy as np
from scipy.spatial.transform import Rotation

QUAT = ["qx", "qy", "qz", "qw"]
XYZ = ["x", "y", "z"]


def wrap(angle):
    """Angle in [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def yaw_of(rot: Rotation) -> np.ndarray:
    """Heading in the parent x-y plane: the angle of the local x axis projected onto it."""
    m = rot.as_matrix()
    return np.arctan2(m[..., 1, 0], m[..., 0, 0])


def to_local(points: np.ndarray, origin: np.ndarray, rot: Rotation) -> np.ndarray:
    """Parent-frame points into the frame of the pose `(origin, rot)`: `R^T (x - p)`."""
    return rot.inv().apply(np.array(points, float) - origin)


def to_parent(points: np.ndarray, origin: np.ndarray, rot: Rotation) -> np.ndarray:
    """Points in the frame of the pose `(origin, rot)` into its parent frame: `R x + p`."""
    return rot.apply(np.array(points, float)) + origin


def slerp_pairs(r0: Rotation, r1: Rotation, frac: np.ndarray) -> Rotation:
    """Element-wise slerp between two equally long rotation arrays, at fractions in [0, 1]."""
    delta = (r0.inv() * r1).as_rotvec()
    return r0 * Rotation.from_rotvec(delta * np.asarray(frac, float)[:, None])


def dense_end(t_us: np.ndarray) -> int:
    """Index of the last sample before the first gap longer than twice the median step.

    Online egomotion is 100 Hz over the clip and then a sparse tail of 0.1-2 s steps (T004);
    poses are only interpolated over the dense part.
    """
    steps = np.diff(t_us)
    if len(steps) == 0:
        return 0
    long = np.flatnonzero(steps > 2 * np.median(steps))
    return int(long[0]) if len(long) else len(t_us) - 1


class Poses:
    """A time series of poses, interpolated linearly in position and by slerp in orientation.

    Never extrapolates: asking for a time outside [first, last] raises.
    """

    def __init__(self, t_us, xyz, quat):
        self.t = np.asarray(t_us, dtype=np.int64)
        if len(self.t) < 2 or np.any(np.diff(self.t) <= 0):
            raise ValueError("poses need at least two strictly increasing timestamps")
        self.xyz = np.array(xyz, dtype=float)
        self.rot = Rotation.from_quat(np.array(quat, dtype=float))

    @classmethod
    def from_frame(cls, df) -> "Poses":
        return cls(df["timestamp"].to_numpy(np.int64), df[XYZ].to_numpy(float, copy=True), df[QUAT].to_numpy(float, copy=True))

    @property
    def start(self) -> int:
        return int(self.t[0])

    @property
    def end(self) -> int:
        return int(self.t[-1])

    def at(self, t_us) -> tuple[np.ndarray, Rotation]:
        t = np.atleast_1d(np.asarray(t_us, dtype=np.int64))
        outside = (t < self.t[0]) | (t > self.t[-1])
        if outside.any():
            raise ValueError(
                f"{outside.sum()} times outside the poses' range [{self.t[0]}, {self.t[-1]}] us, "
                f"e.g. {t[outside][:3].tolist()}"
            )
        i = np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 2)
        frac = (t - self.t[i]) / (self.t[i + 1] - self.t[i])
        xyz = self.xyz[i] + (self.xyz[i + 1] - self.xyz[i]) * frac[:, None]
        return xyz, slerp_pairs(self.rot[i], self.rot[i + 1], frac)
