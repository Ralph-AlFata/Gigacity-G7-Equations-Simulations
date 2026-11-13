"""T013 horizon config and baseline H solver."""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from horizon_pipeline.horizon.config import ConfigError, HorizonConfig, load_config, parse_config
from horizon_pipeline.horizon.solver import Objects, Plan, ego_radius, margins, q_bar, solve

REPO = Path(__file__).resolve().parents[1]
TOL = 1e-3

CFG = HorizonConfig(r_veh_m=2.0, rho0_m=0.2, e_v_bar_mps=0.3, e_theta_bar_rad=0.1, eps_p_m=0.3, eps_v_mps=0.5,
                    a_bar_mps2=2.0, t_valid_s=6.4, grid_s=0.01, time_tolerance_s=TOL)  # fmt: skip


def straight_plan(speed=10.0, duration=6.4):
    tau = np.round(np.arange(0, duration + 1e-9, 0.1), 10)
    return Plan(tau, np.column_stack([speed * tau, np.zeros_like(tau)]), np.full_like(tau, speed))


def objects(*rows):
    """rows: (track_id, x, y, vx, vy, length, width)."""
    a = np.array([r[1:] for r in rows], float).reshape(-1, 6)
    return Objects(np.array([r[0] for r in rows], dtype=object), a[:, 0:2], a[:, 2:4], a[:, 4], a[:, 5])


def static_crossing(L, v, c, b, a):
    """Exact τ where L − vτ = c + bτ + ½aτ² (a disk ahead on the path, closing at v)."""
    return (-(b + v) + math.sqrt((b + v) ** 2 - 2 * a * (c - L))) / a


# ---------------------------------------------------------------- config


def test_repo_config_loads_and_round_trips():
    cfg = load_config(REPO / "configs" / "horizon.yaml")
    assert cfg.r_veh_m is None and cfg.t_valid_s == 6.4
    assert parse_config(cfg.to_dict()) == cfg


@pytest.mark.parametrize("change, message", [
    (lambda r: r.update(extra={}), "unknown sections"),
    (lambda r: r["objects"].update(eps_q_m=1.0), "unknown keys"),
    (lambda r: r["objects"].pop("a_bar_mps2"), "missing keys"),
    (lambda r: r["objects"].update(eps_v_mps=-0.1), "≥ 0"),
    (lambda r: r["horizon"].update(t_valid_s=0), "> 0"),
    (lambda r: r["ego"].update(r_veh_m=0), "> 0"),
    (lambda r: r["ego"].update(rho0_m="auto"), "finite number"),
    (lambda r: r["ego"].update(rho0_m=True), "finite number"),
    (lambda r: r["tracking"].update(e_theta_bar_rad=3.2), "≤ π"),
    (lambda r: r["solver"].update(time_tolerance_s=0.02), "must not exceed"),
    (lambda r: r["solver"].update(method="fancy"), "must be one of"),
    (lambda r: r["objects"].update(no_velocity="drop"), "must be one of"),
])  # fmt: skip
def test_config_rejects_bad_values(change, message):
    raw = yaml.safe_load((REPO / "configs" / "horizon.yaml").read_text())
    change(raw)
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


# ---------------------------------------------------------------- pieces


def test_ego_radius_reproduces_the_old_hand_checked_value():
    # b789fab8 / b4d979af vehicle: 4.688 × 1.999 m, bbox centre 1.312 m ahead of the rear axle.
    fp = {"x_min": 1.312 - 4.688 / 2, "x_max": 1.312 + 4.688 / 2, "y_min": -1.999 / 2, "y_max": 1.999 / 2}
    assert ego_radius(fp) == pytest.approx(3.790, abs=5e-4)


def test_q_bar_follows_the_formula():
    assert q_bar(CFG, 10.0) == pytest.approx(0.3 + 20 * math.sin(0.05))
    assert q_bar(replace(CFG, e_theta_bar_rad=math.pi), 4.0) == pytest.approx(0.3 + 8.0)  # 2·v·sin(π/2)


# ---------------------------------------------------------------- solver


def test_static_object_ahead_matches_the_analytic_crossing():
    h = solve(straight_plan(), objects(("car", 50.0, 0.0, 0, 0, 4, 2)), 2.0, CFG)
    qb = q_bar(CFG, 10.0)
    c = 2.0 + 0.5 * math.hypot(4, 2) + CFG.rho0_m + CFG.eps_p_m
    exact = static_crossing(50.0, 10.0, c, qb + CFG.eps_v_mps, CFG.a_bar_mps2)
    assert exact - TOL <= h.H_s <= exact  # never later than the exact crossing
    assert not h.capped and h.limiting_track_id == "car"


def test_object_head_on_at_30_mps_is_caught():
    h = solve(straight_plan(), objects(("fast", 60.0, 0.0, -30.0, 0, 0.6, 0.6)), 2.0, CFG)
    c = 2.0 + 0.5 * math.hypot(0.6, 0.6) + CFG.rho0_m + CFG.eps_p_m
    exact = static_crossing(60.0, 40.0, c, q_bar(CFG, 10.0) + CFG.eps_v_mps, CFG.a_bar_mps2)
    assert exact - TOL <= h.H_s <= exact


def test_no_objects_is_capped_and_overlap_at_start_is_zero():
    h = solve(straight_plan(), objects(), 2.0, CFG)
    assert (h.H_s, h.capped, h.limiting_track_id) == (6.4, True, None) and math.isnan(h.margin_at_0_m)
    h = solve(straight_plan(), objects(("near", 1.0, 0.0, 0, 0, 1, 1), ("far", 80.0, 30.0, 0, 0, 1, 1)), 2.0, CFG)
    assert (h.H_s, h.capped, h.limiting_track_id) == (0.0, False, "near") and h.margin_at_0_m < 0


def test_far_objects_leave_h_capped():
    h = solve(straight_plan(), objects(("side", 30.0, 60.0, 0, 0, 4, 2)), 2.0, CFG)
    assert h.capped and h.H_s == 6.4 and h.limiting_track_id is None and h.margin_at_0_m > 0


def test_t_cap_is_the_plan_length_at_most_t_valid():
    assert solve(straight_plan(duration=20.0), objects(), 2.0, CFG).T_cap_s == 6.4
    assert solve(straight_plan(duration=3.0), objects(), 2.0, CFG).T_cap_s == pytest.approx(3.0)


def test_v_ref_bar_is_the_max_speed_up_to_t_cap_only():
    plan = straight_plan(duration=20.0)
    speed = plan.speed.copy()
    speed[plan.tau_s > 6.4 + 1e-9] = 30.0  # faster after T_cap: must not count
    speed[plan.tau_s == 3.0] = 12.0
    h = solve(replace(plan, speed=speed), objects(), 2.0, CFG)
    assert h.v_ref_bar_mps == 12.0 and h.q_bar_mps == pytest.approx(q_bar(CFG, 12.0))


def test_non_finite_object_is_refused_not_ignored():
    with pytest.raises(ValueError, match="new"):
        solve(straight_plan(), objects(("ok", 30, 5, 0, 0, 4, 2), ("new", 20, 0, np.nan, np.nan, 4, 2)), 2.0, CFG)


def test_bad_plans_are_refused():
    p = straight_plan()
    with pytest.raises(ValueError, match="start at 0"):
        solve(replace(p, tau_s=p.tau_s + 0.1), objects(), 2.0, CFG)
    with pytest.raises(ValueError, match="non-finite"):
        solve(replace(p, speed=np.where(p.tau_s == 1.0, np.nan, p.speed)), objects(), 2.0, CFG)


# ---------------------------------------------------------------- hand example (tests/hand_example.md)


HAND_OBJECTS = objects(("A", 40.0, 3.0, 0, 0, 4, 2), ("B", 30.0, -8.0, 0, 1.5, 0.6, 0.6))


def test_hand_example_margins():
    qb = q_bar(CFG, 10.0)
    assert qb == pytest.approx(1.299583, abs=1e-6)
    assert HAND_OBJECTS.radius == pytest.approx([2.236068, 0.424264], abs=1e-6)
    m = margins([0.0, 0.5, 1.0], straight_plan(), HAND_OBJECTS, 2.0, qb, CFG)
    expected = [[35.376274, 28.124085], [29.242476, 21.955975], [22.613976, 15.305893]]
    assert m == pytest.approx(np.array(expected), abs=1e-6)


def test_hand_example_horizon():
    h = solve(straight_plan(), HAND_OBJECTS, 2.0, CFG)
    assert h.H_s == pytest.approx(2.04255, abs=TOL) and h.H_s <= 2.042552
    assert h.limiting_track_id == "B" and not h.capped
    assert h.margin_at_0_m == pytest.approx(28.124085, abs=1e-6) and h.T_cap_s == 6.4
