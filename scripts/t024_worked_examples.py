"""T024 worked examples: the numbers in docs/derivation/horizon_derivation.tex, section "Worked examples".

Four scenes, each solved with
- the baseline (PDF) disks,
- the revised model, physical layer only,
- the revised model, calibrated layer only (provisional p99 curves from scripts/spike_t024_object_motion.py),
- the revised model, both layers (their intersection),
and, for scene E2, the reaction-aware horizon H_r (follow the plan until s, then brake).

Plain numpy on purpose, so that every step can be checked by hand against the document.

    uv run python scripts/t024_worked_examples.py
"""

import numpy as np

DT = 0.001  # integration step for the physical layer (s)
GRID = 0.01  # search grid for H (s)
T_CAP = 6.4

# ---------------------------------------------------------------- parameters (Table 3 of the document)
EGO = dict(x_min=-1.032, x_max=3.656, y_min=-0.9995, y_max=0.9995)  # footprint about the rear axle
R_F = float(np.hypot(EGO["x_max"], EGO["y_max"]))  # farthest corner from the rear axle, 3.790 m
RHO_P, E_THETA = 0.2, 0.05  # tracking tube (m) and heading error bound (rad)
RHO = RHO_P + 2 * R_F * np.sin(E_THETA / 2)  # Lemma 1: 0.389 m
B_EGO = 6.0  # ego fallback braking (m/s²)

# Physical layer: estimate-error bounds set well above the measured p95 (T009, parked tracks), since a
# physical bound must hold almost always; motion limits from physics/literature (Table 3).
# b_max and a_perp: tyre-road friction, ABS cars brake at 8.5-10.6 m/s² on dry roads (Miller et al. 2023).
PHYS_VEH = dict(eps_p=0.5, eps_v=1.5, eps_psi=np.radians(6), a_max=4.0, b_max=10.0, v_max=50.0, kappa=0.2, a_perp=10.0)
PHYS_PED = dict(eps_p=0.5, eps_v=1.5, a=1.5, v_max=3.0)
# Provisional p99 box-overhang curves c0 + c1 τ + c2 τ² (front, back, lateral), by motion state at t.
CAL_VEH = {
    "stopped": dict(front=(0.21, 0.49, 0.0), back=(0.28, 0.42, 0.0), lat=(0.28, 0.15, 0.0)),
    "moving": dict(front=(0.48, 1.39, 0.175), back=(0.74, 1.02, 0.29), lat=(0.0, 1.94, 0.175)),
}
CAL_PED = (0.0, 0.85, 0.025)  # centre residual radius, p99


def poly(c, tau):
    return c[0] + c[1] * tau + c[2] * tau**2


# ---------------------------------------------------------------- geometry
def rect_corners(c, psi, a, b):
    u, n = np.array([np.cos(psi), np.sin(psi)]), np.array([-np.sin(psi), np.cos(psi)])
    return np.array([c + sa * a * u + sb * b * n for sa, sb in [(1, 1), (1, -1), (-1, -1), (-1, 1)]])


def rect_distance(P, Q):
    """Distance between two convex quadrilaterals; 0 if they overlap (separating-axis test)."""
    def separated(A, B):
        for i in range(4):
            e = A[(i + 1) % 4] - A[i]
            ax = np.array([-e[1], e[0]])
            if (A @ ax).max() < (B @ ax).min() or (B @ ax).max() < (A @ ax).min():
                return True
        return False

    if not (separated(P, Q) or separated(Q, P)):
        return 0.0

    def pt_seg(p, a, b):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        return np.linalg.norm(p - (a + t * ab))

    return min(pt_seg(p, B[i], B[(i + 1) % 4]) for A, B in ((P, Q), (Q, P)) for p in A for i in range(4))


def point_rect_distance(p, R):
    """Distance from a point to a convex quadrilateral (0 inside)."""
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    side = [cross(R[(i + 1) % 4] - R[i], p - R[i]) for i in range(4)]
    inside = all(x >= 0 for x in side) or all(x <= 0 for x in side)
    if inside:
        return 0.0
    out = []
    for i in range(4):
        a, b = R[i], R[(i + 1) % 4]
        t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
        out.append(np.linalg.norm(p - (a + t * (b - a))))
    return min(out)


def ego_rect(x, y, psi, inflate):
    """Ego footprint at rear-axle pose (x, y, psi), each side pushed out by `inflate` (⊇ footprint ⊕ disk)."""
    cx = 0.5 * (EGO["x_min"] + EGO["x_max"])
    c = np.array([x, y]) + cx * np.array([np.cos(psi), np.sin(psi)])
    return rect_corners(c, psi, 0.5 * (EGO["x_max"] - EGO["x_min"]) + inflate, EGO["y_max"] + inflate)


# ---------------------------------------------------------------- object occupancy
def rotated_half_dims(l, w, dpsi):
    """Half-dims, along the original axes, of every rotation of an l×w half-box by at most dpsi."""
    diag = np.hypot(l, w)
    lp = diag if dpsi >= np.arctan2(w, l) else l * np.cos(dpsi) + w * np.sin(dpsi)
    wp = diag if dpsi >= np.arctan2(l, w) else w * np.cos(dpsi) + l * np.sin(dpsi)
    return lp, wp


def physical_vehicle_profile(v_hat, p, T):
    """Integrals of the physical layer (Proposition 1) on a fine grid: s⁺, s⁻, d̄, Ω (true heading
    change) and Δ̄ = ε_ψ + Ω (heading change against the estimate)."""
    t = np.arange(0, T + DT / 2, DT)
    v_hi = np.minimum(p["v_max"], v_hat + p["eps_v"] + p["a_max"] * t)
    v_lo = np.maximum(0.0, v_hat - p["eps_v"] - p["b_max"] * t)
    v_star = np.sqrt(p["a_perp"] / p["kappa"])  # speed with the largest possible yaw rate
    omega = np.where(v_hi < v_star, p["kappa"] * v_hi,
                     np.where(v_lo > v_star, p["a_perp"] / np.maximum(v_lo, 1e-9), p["kappa"] * v_star))
    def integ(f):
        return np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * DT)])

    Omega = integ(omega)
    dpsi = p["eps_psi"] + Omega
    # progress along u: v cos α ≥ v⁻ cos Δ̄ while Δ̄ ≤ π/2, and ≥ v⁺ cos Δ̄ (< 0) beyond
    lo_rate = np.where(dpsi <= np.pi / 2, v_lo * np.cos(dpsi), v_hi * np.cos(np.minimum(dpsi, np.pi)))
    return t, integ(v_hi), integ(lo_rate), integ(v_hi * np.sin(np.minimum(dpsi, np.pi / 2))), Omega, dpsi


def vehicle_intervals(obj, tau, layer, prof=None):
    """Per-axis bounds of the predicted occupancy, in the object frame at t (centre at ô):
    longitudinal [lo, hi] of the box extent, lateral half-extent. None for a layer means 'no bound'."""
    l, w, v = obj["length"] / 2, obj["width"] / 2, obj["speed"]
    if layer == "physical":
        t, s_hi, s_lo, d_lat, Omega, dpsi = prof
        k = int(round(tau / DT))
        p = PHYS_VEH
        lp, wp = rotated_half_dims(l, w, min(dpsi[k], np.pi / 2))
        pivot = 2 * l * np.sin(min(Omega[k], np.pi) / 2)  # the box centre swings about the rear axle (≤ l behind it)
        lo = s_lo[k] - pivot - lp - p["eps_p"]
        hi = s_hi[k] + pivot + lp + p["eps_p"]
        lat = d_lat[k] + pivot + wp + p["eps_p"]
        return lo, hi, lat
    c = CAL_VEH["stopped" if v < 0.5 else "moving"]
    s = v * tau  # heading-projected constant velocity
    return s - l - poly(c["back"], tau), s + l + poly(c["front"], tau), w + poly(c["lat"], tau)


def vehicle_rect(obj, tau, layers, prof):
    lo, hi, lat = -np.inf, np.inf, np.inf
    for layer in layers:
        a, b, c = vehicle_intervals(obj, tau, layer, prof)
        lo, hi, lat = max(lo, a), min(hi, b), min(lat, c)
    u = np.array([np.cos(obj["yaw"]), np.sin(obj["yaw"])])
    centre = np.array([obj["x"], obj["y"]]) + 0.5 * (lo + hi) * u
    return rect_corners(centre, obj["yaw"], 0.5 * (hi - lo), lat)


def pedestrian_disks(obj, tau, layers):
    """Sound disks (centre, radius) that each contain the pedestrian's body at t+τ."""
    o, w, r = np.array([obj["x"], obj["y"]]), np.array([obj["vx"], obj["vy"]]), 0.5 * np.hypot(obj["length"], obj["width"])
    disks = []
    if "physical" in layers:
        p = PHYS_PED
        disks.append((o + w * tau, p["eps_p"] + p["eps_v"] * tau + 0.5 * p["a"] * tau**2 + r))
        disks.append((o, p["eps_p"] + p["v_max"] * tau + r))
    if "calibrated" in layers:
        disks.append((o + w * tau, poly(CAL_PED, tau) + r))
    return disks


# ---------------------------------------------------------------- horizons
def plan_pose(plan, tau):
    """Plan pose at τ: (x, y, psi)."""
    return plan(tau)


def margin(plan, obj, tau, layers, inflate=RHO, prof=None, ego_pose=None):
    x, y, psi = ego_pose if ego_pose is not None else plan(tau)
    E = ego_rect(x, y, psi, inflate)
    if obj["kind"] == "vehicle":
        return rect_distance(E, vehicle_rect(obj, tau, layers, prof))
    if obj["kind"] == "pedestrian":
        return max(point_rect_distance(c, E) - r for c, r in pedestrian_disks(obj, tau, layers))
    raise ValueError(obj["kind"])


def first_contact(plan, obj, layers, T=T_CAP):
    prof = physical_vehicle_profile(obj["speed"], PHYS_VEH, T + 3) if obj["kind"] == "vehicle" else None
    for tau in np.arange(0, T + GRID / 2, GRID):
        if margin(plan, obj, tau, layers, prof=prof) <= 0:
            return round(tau, 2)
    return None  # capped


def baseline(plan, obj, r_veh, cfg=dict(rho0=0.2, e_v=0.3, e_th=0.05, eps_p=0.3, eps_v=0.5, a=2.0), v_ref=None):
    """The PDF baseline, as in horizon_pipeline/horizon/solver.py, on the same grid."""
    q = cfg["e_v"] + 2 * v_ref * np.sin(cfg["e_th"] / 2)
    rk = 0.5 * np.hypot(obj["length"], obj["width"])
    for tau in np.arange(0, T_CAP + GRID / 2, GRID):
        x, y, _ = plan(tau)
        D = np.hypot(x - (obj["x"] + obj["vx"] * tau), y - (obj["y"] + obj["vy"] * tau))
        S = r_veh + rk + cfg["rho0"] + cfg["eps_p"] + (q + cfg["eps_v"]) * tau + 0.5 * cfg["a"] * tau**2
        if D - S <= 0:
            return round(tau, 2)
    return None


def reaction_horizon(v_ego, obj, layers, b_ego=B_EGO):
    """H_r for a straight plan at constant speed v_ego along +x: the last switch time s such that
    following the plan until s and then braking at b_ego along the same path touches nothing while
    the ego moves (passive safety)."""
    prof = physical_vehicle_profile(obj["speed"], PHYS_VEH, T_CAP + 3) if obj["kind"] == "vehicle" else None
    for s in np.arange(0, T_CAP + GRID / 2, GRID):
        t_stop = v_ego / b_ego
        for tau in np.arange(0, s + t_stop + GRID / 2, GRID):
            dt = max(0.0, tau - s)
            x = v_ego * min(tau, s) + v_ego * min(dt, t_stop) - 0.5 * b_ego * min(dt, t_stop) ** 2
            if margin(None, obj, tau, layers, prof=prof, ego_pose=(x, 0.0, 0.0)) <= 0:
                return round(max(0.0, s - GRID), 2)
    return None


def straight(v):
    return lambda tau: (v * tau, 0.0, 0.0)


def report(name, plan, obj, v_ref, extra=""):
    b2 = baseline(plan, obj, 2.0, v_ref=v_ref)
    b3 = baseline(plan, obj, R_F, v_ref=v_ref)
    ph = first_contact(plan, obj, ["physical"])
    ca = first_contact(plan, obj, ["calibrated"])
    bo = first_contact(plan, obj, ["physical", "calibrated"])
    fmt = lambda h: "capped (6.4)" if h is None else f"{h:.2f} s"
    print(f"{name}\n  baseline r_veh=2.0: {fmt(b2)} | baseline r_veh=3.79: {fmt(b3)} | revised physical: {fmt(ph)} | "
          f"revised calibrated p99: {fmt(ca)} | revised both: {fmt(bo)}{extra}")


def main():
    print(f"ego: R_F = {R_F:.3f} m, inflation ρ = ρ_p + 2 R_F sin(ē_θ/2) = {RHO_P} + {RHO - RHO_P:.3f} = {RHO:.3f} m")
    car = dict(kind="vehicle", length=3.89, width=1.77, yaw=0.0, vy=0.0)

    # E1: a car in the next lane, same speed, abreast of the ego (centre 3.5 m to the left, level with the ego's box centre)
    e1 = dict(car, x=1.312, y=3.5, vx=10.0, speed=10.0)
    report("E1 next-lane car abreast, both 10 m/s, 3.5 m centre to centre", straight(10.0), e1, 10.0)

    # E2: following a car at the same 4 m/s, 4 m bumper to bumper
    e2 = dict(car, x=EGO["x_max"] + 4.0 + 3.89 / 2, y=0.0, vx=4.0, speed=4.0)
    hr = {k: reaction_horizon(4.0, e2, layers) for k, layers in
          [("physical", ["physical"]), ("calibrated", ["calibrated"]), ("both", ["physical", "calibrated"])]}
    report("E2 following at 4 m/s, 4 m gap", straight(4.0), e2, 4.0,
           extra="\n  reaction-aware H_r (brake at 6 m/s² after s): " + " | ".join(f"{k} {v:.2f} s" for k, v in hr.items()))
    m = RHO + PHYS_VEH["eps_p"]
    vf = 4.0 - PHYS_VEH["eps_v"]
    rss = (4.0 - m + vf**2 / (2 * PHYS_VEH["b_max"]) - 4.0**2 / (2 * B_EGO)) / 4.0
    print(f"  RSS-type closed form (lead brakes at b_max from τ=0, ego at b_e after s; no heading change): H_r = {rss:.3f} s")
    straight_lead = dict(PHYS_VEH, kappa=1e-9, eps_psi=0.0)
    saved = dict(PHYS_VEH)
    PHYS_VEH.update(straight_lead)
    print(f"  numerical, physical layer without heading change: H_r = {reaction_horizon(4.0, e2, ['physical']):.2f} s, "
          f"H_c = {first_contact(straight(4.0), e2, ['physical']):.2f} s (closed form {(4.0 - m + vf**2 / (2 * PHYS_VEH['b_max'])) / 4.0:.3f} s)")
    PHYS_VEH.update(saved)

    # E3: the hand example: pedestrian at (30, −8) walking +y at 1.5 m/s, plan straight at 10 m/s
    e3 = dict(kind="pedestrian", length=0.6, width=0.6, x=30.0, y=-8.0, vx=0.0, vy=1.5, speed=1.5, yaw=np.pi / 2)
    report("E3 hand-example pedestrian crossing (tests/hand_example.md)", straight(10.0), e3, 10.0)

    # E4: passing a parked car at the curb, centre 3.0 m to the right, 15 m ahead, ego at 10 m/s
    e4 = dict(car, x=15.0, y=-3.0, vx=0.0, speed=0.0)
    report("E4 passing a parked car, 3.0 m centre offset, 15 m ahead, ego 10 m/s", straight(10.0), e4, 10.0)


if __name__ == "__main__":
    main()
