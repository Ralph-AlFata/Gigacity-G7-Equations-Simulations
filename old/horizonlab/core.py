"""Disk-tube equations and continuous-interval checking for a polyline reference.

All positions are in a common fixed 2D frame; units are metres and seconds.
The interval checker returns a conservative prefix, not a formal numerical proof.
It never declares a whole interval clear solely because its endpoints are clear.
"""
from dataclasses import dataclass
import numpy as np


def nonnegative(name, values):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    return x


def q_from_tracking_bounds(speed_error_mps, heading_error_rad, ref_speed_max_mps):
    nonnegative("tracking bounds", [speed_error_mps, heading_error_rad, ref_speed_max_mps])
    if heading_error_rad > np.pi:
        raise ValueError("heading error bound must be at most pi")
    return float(speed_error_mps + 2 * ref_speed_max_mps * np.sin(heading_error_rad / 2))


def rear_axle_radius(length_m, width_m, rear_axle_to_center_m):
    nonnegative("vehicle dimensions", [length_m, width_m])
    if not np.isfinite(rear_axle_to_center_m):
        raise ValueError("rear axle offset must be finite")
    return float(np.hypot(length_m / 2 + abs(rear_axle_to_center_m), width_m / 2))


@dataclass
class Obstacle:
    track_id: str
    position_m: np.ndarray  # At its last observation, BEFORE age propagation.
    velocity_mps: np.ndarray
    radius_m: float
    eps_position_m: float
    eps_velocity_mps: float
    acceleration_bound_mps2: float
    age_s: float = 0.0
    motion_model: str = "constant_velocity"

    def __post_init__(self):
        if self.motion_model not in {"constant_velocity", "bounded_speed"}:
            raise ValueError("unknown obstacle motion model")
        self.position_m = np.asarray(self.position_m, dtype=float)
        self.velocity_mps = np.asarray(self.velocity_mps, dtype=float)
        if self.position_m.shape != (2,) or self.velocity_mps.shape != (2,):
            raise ValueError("obstacle position and velocity must be 2D")
        if not np.isfinite([self.position_m, self.velocity_mps]).all():
            raise ValueError("nonfinite obstacle state")
        nonnegative("obstacle bounds", [self.radius_m, self.eps_position_m,
                    self.eps_velocity_mps, self.acceleration_bound_mps2, self.age_s])
        if self.motion_model == "bounded_speed" and (np.any(self.velocity_mps != 0) or self.acceleration_bound_mps2 != 0):
            raise ValueError("bounded-speed envelope requires zero nominal velocity and acceleration term")

    def center(self, tau_s):
        return self.position_m + self.velocity_mps * (self.age_s + tau_s)

    def beta(self, tau_s):
        elapsed = self.age_s + tau_s
        return (self.eps_position_m + self.eps_velocity_mps * elapsed
                + 0.5 * self.acceleration_bound_mps2 * elapsed ** 2)


@dataclass
class HorizonResult:
    H_s: float
    interval_end_s: float
    status: str
    limiting_track_id: str | None
    initial_margin_m: float
    interval_checks: int


def clearance_horizon(tau_s, reference_xy_m, obstacles, ego_radius_m, rho0_m,
                      qbar_mps, cap_s, time_tolerance_s=0.001, margin_tolerance_m=1e-8):
    """Largest continuously certified prefix found, on [0, cap_s].

    On each linear reference segment, relative centers are affine. The exact
    minimum center distance on an interval is the distance to that line segment.
    Separation S is nondecreasing, so min(D) - S(interval_end) lower-bounds m
    throughout the interval. Unresolved intervals are recursively subdivided.
    A remaining unresolved interval returns its LEFT endpoint conservatively.

    status=resolution_limit does NOT assert a real boundary in that interval.
    status=boundary_bracket means a nonpositive margin was observed at its end.
    cap_reached means clearance was certified through cap, not that it ends there.
    """
    t = np.asarray(tau_s, dtype=float)
    xy = np.asarray(reference_xy_m, dtype=float)
    nonnegative("solver parameters", [ego_radius_m, rho0_m, qbar_mps, cap_s,
                                      time_tolerance_s, margin_tolerance_m])
    if time_tolerance_s <= 0:
        raise ValueError("time_tolerance_s must be positive")
    if (t.ndim != 1 or len(t) < 2 or xy.shape != (len(t), 2) or not np.isfinite(xy).all()
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
            or abs(t[0]) > 1e-9 or cap_s > t[-1] + 1e-9):
        raise ValueError("reference must start at 0, have increasing times, and cover cap")
    if not obstacles:
        return HorizonResult(cap_s, cap_s, "cap_reached", None, float("inf"), 0)
    pos = np.array([o.position_m for o in obstacles])
    vel = np.array([o.velocity_mps for o in obstacles])
    age = np.array([o.age_s for o in obstacles])
    base = np.array([ego_radius_m + rho0_m + o.radius_m + o.eps_position_m for o in obstacles])
    ev = np.array([o.eps_velocity_mps for o in obstacles])
    acc = np.array([o.acceleration_bound_mps2 for o in obstacles])

    def required(tau):
        elapsed = age + tau
        return base + qbar_mps * tau + ev * elapsed + 0.5 * acc * elapsed ** 2

    def relative(tau, point):
        return point - (pos + vel * (age + tau)[:, None])

    initial = np.linalg.norm(relative(0, xy[0]), axis=1) - required(0)
    j = int(np.argmin(initial))
    if initial[j] <= margin_tolerance_m:
        return HorizonResult(0.0, 0.0, "initial_overlap_or_tolerance",
                             obstacles[j].track_id, float(initial[j]), 0)
    checks = 0

    def certify(left, right, p_left, p_right):
        nonlocal checks
        checks += 1
        a = relative(left, p_left)
        b = relative(right, p_right)
        delta = b - a
        dd = np.einsum("ij,ij->i", delta, delta)
        fraction = np.divide(-np.einsum("ij,ij->i", a, delta), dd,
                             out=np.zeros_like(dd), where=dd > 0)
        fraction = np.clip(fraction, 0, 1)
        d_min = np.linalg.norm(a + fraction[:, None] * delta, axis=1)
        lower = d_min - required(right)
        if np.all(lower > margin_tolerance_m):
            return None
        if right - left <= time_tolerance_s:
            end_margin = np.linalg.norm(b, axis=1) - required(right)
            k = int(np.argmin(end_margin if end_margin.min() <= 0 else lower))
            status = "boundary_bracket" if end_margin.min() <= 0 else "resolution_limit"
            return left, right, status, obstacles[k].track_id
        mid = (left + right) / 2
        p_mid = (p_left + p_right) / 2
        event = certify(left, mid, p_left, p_mid)
        return event if event is not None else certify(mid, right, p_mid, p_right)

    for j in range(len(t) - 1):
        if t[j] >= cap_s:
            break
        right = min(float(t[j + 1]), cap_s)
        fraction = (right - t[j]) / (t[j + 1] - t[j])
        p_right = xy[j] + fraction * (xy[j + 1] - xy[j])
        event = certify(float(t[j]), right, xy[j], p_right)
        if event is not None:
            left, right, status, blocker = event
            return HorizonResult(left, right, status, blocker, float(initial.min()), checks)
    return HorizonResult(cap_s, cap_s, "cap_reached", None, float(initial.min()), checks)
