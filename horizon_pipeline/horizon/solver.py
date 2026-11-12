"""Baseline H solver (T013): the PDF's equations, for one plan, from numbers only.

For a plan received at tick t, with the objects in the scene at t (E03 README, T023):
- p_ref(τ): the plan's rear-axle position, linear between steps; planar (x, y) only.
- q̄ = ē_v + 2·v̄_ref·sin(ē_θ/2), with v̄_ref the plan's max speed over [0, T_cap].
- Dₖ(τ) = ‖p_ref(τ) − (ôₖ + ŵₖτ)‖.
- Sₖ(τ) = r_veh + rₖ + ρ0 + ε_p + (q̄ + ε_v)·τ + ½·ā·τ², rₖ = half the box's planar diagonal.
- H = inf{τ ∈ [0, T_cap] : minₖ (Dₖ − Sₖ)(τ) ≤ 0}, T_cap = min(plan duration, t_valid). No such τ:
  H = T_cap, `capped`.

The first grid point (every `grid_s`) with min margin ≤ 0 is found, then the crossing is bisected
to `time_tolerance_s`. H is the last time known clear: it is never later than the exact crossing,
and at most the tolerance earlier. Reads no files and keeps no state.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from horizon_pipeline.horizon.config import HorizonConfig


@dataclass(frozen=True)
class Plan:
    """One sample of a reference at one tick: `tau_s` from 0, increasing; planar positions; speeds."""

    tau_s: np.ndarray  # (n,)
    xy: np.ndarray  # (n, 2), world frame
    speed: np.ndarray  # (n,)

    @classmethod
    def from_rows(cls, rows: pd.DataFrame) -> "Plan":
        """From the reference rows of one (tick, sample), in step order (T011 format)."""
        return cls(rows["tau_s"].to_numpy(float), rows[["x", "y"]].to_numpy(float), rows["speed"].to_numpy(float))


@dataclass(frozen=True)
class Objects:
    """The objects in the scene at the plan's tick: position, velocity, box size."""

    track_id: np.ndarray  # (m,)
    xy: np.ndarray  # (m, 2), world frame
    vxy: np.ndarray  # (m, 2)
    length: np.ndarray  # (m,)
    width: np.ndarray  # (m,)

    @classmethod
    def from_rows(cls, rows: pd.DataFrame) -> "Objects":
        """From `objects.parquet` rows (one tick; the caller keeps only `in_scene` ones)."""
        return cls(rows["track_id"].to_numpy(), rows[["x", "y"]].to_numpy(float), rows[["vx", "vy"]].to_numpy(float),
                   rows["length"].to_numpy(float), rows["width"].to_numpy(float))  # fmt: skip

    @property
    def radius(self) -> np.ndarray:
        return 0.5 * np.hypot(self.length, self.width)


@dataclass(frozen=True)
class Horizon:
    H_s: float
    capped: bool
    limiting_track_id: object  # None when capped
    margin_at_0_m: float  # minₖ mₖ(0); NaN without objects
    T_cap_s: float
    q_bar_mps: float
    v_ref_bar_mps: float
    r_veh_m: float


def ego_radius(footprint: dict) -> float:
    """Smallest disk around the rear axle (the origin) that covers the footprint: its farthest corner."""
    return float(max(np.hypot(x, y) for x in (footprint["x_min"], footprint["x_max"])
                     for y in (footprint["y_min"], footprint["y_max"])))  # fmt: skip


def q_bar(cfg: HorizonConfig, v_ref_bar: float) -> float:
    return cfg.e_v_bar_mps + 2 * v_ref_bar * np.sin(cfg.e_theta_bar_rad / 2)


def margins(tau: np.ndarray, plan: Plan, objects: Objects, r_veh: float, qb: float, cfg: HorizonConfig) -> np.ndarray:
    """mₖ(τ) for every τ (rows) and object (columns)."""
    tau = np.atleast_1d(np.asarray(tau, float))
    p = np.column_stack([np.interp(tau, plan.tau_s, plan.xy[:, 0]), np.interp(tau, plan.tau_s, plan.xy[:, 1])])
    o = objects.xy[None, :, :] + objects.vxy[None, :, :] * tau[:, None, None]
    D = np.linalg.norm(p[:, None, :] - o, axis=2)
    S = (r_veh + objects.radius[None, :] + cfg.rho0_m + cfg.eps_p_m
         + ((qb + cfg.eps_v_mps) * tau + 0.5 * cfg.a_bar_mps2 * tau**2)[:, None])  # fmt: skip
    return D - S


def solve(plan: Plan, objects: Objects, r_veh: float, cfg: HorizonConfig) -> Horizon:
    if len(plan.tau_s) == 0 or plan.tau_s[0] != 0 or np.any(np.diff(plan.tau_s) <= 0):
        raise ValueError("plan tau_s must start at 0 and increase")
    if not all(np.isfinite(a).all() for a in (plan.xy, plan.speed)):
        raise ValueError("plan has non-finite positions or speeds")
    # A NaN margin compares False with 0, so a NaN object would be silently ignored: refuse it.
    bad = ~np.isfinite(np.column_stack([objects.xy, objects.vxy, objects.length, objects.width])).all(axis=1)
    if bad.any():
        raise ValueError(f"objects with non-finite position, velocity or size: {list(objects.track_id[bad])}")
    T_cap = float(min(plan.tau_s[-1], cfg.t_valid_s))
    inside = plan.tau_s <= T_cap
    v_ref_bar = float(max(plan.speed[inside].max(), np.interp(T_cap, plan.tau_s, plan.speed)))
    qb = float(q_bar(cfg, v_ref_bar))
    result = dict(T_cap_s=T_cap, q_bar_mps=qb, v_ref_bar_mps=v_ref_bar, r_veh_m=float(r_veh))
    if len(objects.xy) == 0:
        return Horizon(H_s=T_cap, capped=True, limiting_track_id=None, margin_at_0_m=float("nan"), **result)

    grid = np.append(np.arange(0.0, T_cap, cfg.grid_s), T_cap)
    m = margins(grid, plan, objects, r_veh, qb, cfg)
    worst = m.min(axis=1)
    margin_at_0 = float(worst[0])
    hit = np.flatnonzero(worst <= 0)
    if len(hit) == 0:
        return Horizon(H_s=T_cap, capped=True, limiting_track_id=None, margin_at_0_m=margin_at_0, **result)
    i = int(hit[0])
    if i == 0:
        return Horizon(H_s=0.0, capped=False, limiting_track_id=objects.track_id[int(np.argmin(m[0]))],
                       margin_at_0_m=margin_at_0, **result)  # fmt: skip
    lo, hi = float(grid[i - 1]), float(grid[i])  # clear at lo, not clear at hi
    while hi - lo > cfg.time_tolerance_s:
        mid = 0.5 * (lo + hi)
        if margins(mid, plan, objects, r_veh, qb, cfg).min() <= 0:
            hi = mid
        else:
            lo = mid
    limiting = objects.track_id[int(np.argmin(margins(hi, plan, objects, r_veh, qb, cfg)[0]))]
    return Horizon(H_s=lo, capped=False, limiting_track_id=limiting, margin_at_0_m=margin_at_0, **result)
