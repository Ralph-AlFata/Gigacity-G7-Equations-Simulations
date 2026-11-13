"""T024 worked examples: the numbers in docs/derivation/horizon_derivation.tex, section "Worked examples".

Four scenes, each solved with
- the baseline (PDF) disks,
- the revised model with the PHYSICAL parameter set (what an object can do),
- the revised model with the FORESEEABLE parameter set (what a normal road user does),
plus the reaction horizon H_r (follow the plan until s, then brake along it), and a sensitivity of
scene E1 to the lateral-acceleration assumption.

No parameter is fitted to our data: every value is from physics, regulation or literature, or is an
explicit assumption (Table 3 of the document). Plain numpy on purpose, so that every step can be
checked by hand.

    uv run python scripts/t024_worked_examples.py
"""

import numpy as np

DT = 0.001  # integration step for the object envelopes (s)
GRID = 0.01  # search grid for H (s)
T_CAP = 6.4

# ---------------------------------------------------------------- parameters (Table 3 of the document)
EGO = dict(x_min=-1.032, x_max=3.656, y_min=-0.9995, y_max=0.9995)  # footprint about the rear axle
R_F = float(np.hypot(EGO["x_max"], EGO["y_max"]))  # farthest corner from the rear axle, 3.790 m
RHO_P, E_THETA = 0.2, 0.05  # tracking tube (m) and heading error bound (rad): current config
RHO = RHO_P + 2 * R_F * np.sin(E_THETA / 2)  # Lemma 1: 0.389 m
B_EGO = 4.0  # ego fallback braking (m/s²): UN R157 §5.5.1, a minimum-risk manoeuvre aims at ≤ 4 m/s²

# Estimation errors describe the perception system, not behaviour: the same in both sets
# (current config values; ε_ψ = the ego's heading bound).
EST = dict(eps_p=0.3, eps_v=0.5, eps_psi=0.05)
def bosetti(v):
    """Normal-driving lateral acceleration envelope, Bosetti et al. 2014, Eq. 1: a0/sqrt(1+(v/v0)^4)."""
    return 5.22 / np.sqrt(1.0 + (np.asarray(v, float) / 14.84) ** 4)


# Sideways behaviour of a lane-following vehicle (Proposition 3): lateral speed ramps at `acc` up to
# `cap` for `rho` seconds, then brakes at `brk` (no braking if brk = 0); plus a fluctuation margin `mu`.
LATERAL = {
    "RSS (ad-rss-lib)": dict(rho=2.0, acc=0.2, brk=0.8, cap=np.inf, mu=0.1),
    "IEEE 2846 / UniD": dict(rho=np.inf, acc=1.18, brk=0.0, cap=0.31, mu=0.1),
    "lane change (R79)": dict(rho=np.inf, acc=1.0, brk=0.0, cap=0.7, mu=0.1),
    "cut-in (R157)": dict(rho=np.inf, acc=1.0, brk=0.0, cap=3.0, mu=0.375),
}

PARAMS = {
    # What is possible. Braking and cornering: tyre-road friction (ABS 8.5-10.6 m/s² on dry roads,
    # Miller 2023); acceleration: the observed maximum 4.43 m/s² (Bosetti 2014), rounded up; κ_max an
    # assumption (5 m radius). Pedestrians: sprinters, 6.42 m/s² and 9.46 m/s (di Prampero 2005).
    "physical": dict(
        veh=dict(EST, a_max=5.0, b_max=10.0, v_max=50.0, kappa=0.2, a_perp=lambda v: 10.0),
        ped=dict(EST, a=6.5, v_max=9.5),
    ),
    # What a normal road user does. A lead vehicle may still brake fully (UN R157 §5.2.5.1), so b_max
    # stays; acceleration a_max,accel = 3.5 m/s² (ad-rss-lib); turning within Bosetti's envelope.
    # Vehicles keep their heading line (regime "lane") unless option T lets slow ones turn.
    "foreseeable": dict(
        veh=dict(EST, a_max=3.5, b_max=10.0, v_max=50.0, kappa=0.2, a_perp=bosetti,
                 lat=LATERAL["RSS (ad-rss-lib)"], turning="never", v_turn=5.0, stopped_yield=False),
        ped=dict(EST, a=0.6, v_max=2.0),  # Koschi et al. 2018, raised to the observed speed + 0.1 m/s
    ),
}


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
    if all(x >= 0 for x in side) or all(x <= 0 for x in side):
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


# ---------------------------------------------------------------- object envelopes (Propositions 1 and 2)
def rotated_half_dims(l, w, dpsi):
    """Half-dims, along the original axes, of every rotation of an l×w half-box by at most dpsi (Lemma 3)."""
    diag = np.hypot(l, w)
    lp = diag if dpsi >= np.arctan2(w, l) else l * np.cos(dpsi) + w * np.sin(dpsi)
    wp = diag if dpsi >= np.arctan2(l, w) else w * np.cos(dpsi) + l * np.sin(dpsi)
    return lp, wp


def lane_following(v_hat, p):
    """Is this vehicle held to its heading line (Proposition 3) rather than allowed to turn?"""
    if "lat" not in p:
        return False  # physical set: always the turning envelope
    return p["turning"] == "never" or v_hat >= p["v_turn"]


def vehicle_profile(v_hat, p, T):
    """The integrals of Propositions 1 and 3 on a fine grid: S⁺, S⁻, d̄, Ω, Δ̄."""
    t = np.arange(0, T + DT / 2, DT)
    v_hi = np.minimum(p["v_max"], v_hat + p["eps_v"] + p["a_max"] * t)
    v_lo = np.maximum(0.0, v_hat - p["eps_v"] - p["b_max"] * t)
    # ω̄(s) = max over v in [v⁻, v⁺] of min(κ_max v, a_perp(v)/v), evaluated on 200 speeds per step
    frac = np.linspace(0, 1, 200)[None, :]
    vv = np.maximum(v_lo[:, None] + (v_hi - v_lo)[:, None] * frac, 1e-6)
    omega = np.minimum(p["kappa"] * vv, p["a_perp"](vv) / vv).max(axis=1)

    def integ(f):
        return np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * DT)])

    if lane_following(v_hat, p):
        # Proposition 3: the heading line is kept (Ω = 0); sideways, the RSS lateral response
        # (accelerate at most `acc` for `rho`, then brake at `brk`) plus the margin mu, plus the drift
        # ∫ v⁺ sin ε_ψ that comes from not knowing the heading exactly.
        L = p["lat"]
        peak = min(L["cap"], L["acc"] * L["rho"])
        v_lat = np.where(t <= L["rho"], np.minimum(L["cap"], L["acc"] * t),
                         np.maximum(0.0, peak - L["brk"] * (t - L["rho"])) if L["brk"] > 0 else peak)
        Omega = np.zeros_like(t)
        # without turning faster than κ_max, the sideways offset after a distance S is at most κ S²/2
        behave = np.minimum(integ(v_lat), 0.5 * p["kappa"] * integ(v_hi) ** 2) + L["mu"]
        d = behave + integ(v_hi * np.sin(p["eps_psi"]))
    else:
        Omega = integ(omega)
        d = None
    dpsi = p["eps_psi"] + Omega
    lo_rate = np.where(dpsi <= np.pi / 2, v_lo * np.cos(dpsi), v_hi * np.cos(np.minimum(dpsi, np.pi)))
    S_hi, S_lo = integ(v_hi), integ(lo_rate)
    if d is None:
        d = integ(v_hi * np.sin(np.minimum(dpsi, np.pi / 2)))
    else:  # the sideways motion is along the true normal, which is tilted by ≤ ε_ψ against n
        S_hi, S_lo = S_hi + behave * np.sin(p["eps_psi"]), S_lo - behave * np.sin(p["eps_psi"])
    return dict(S_hi=S_hi, S_lo=S_lo, d=d, Omega=Omega, dpsi=dpsi)


def vehicle_rect(obj, tau, p, prof):
    """Proposition 1: the predicted box at τ, in world coordinates."""
    l, w = obj["length"] / 2, obj["width"] / 2
    k = int(round(tau / DT))
    S_hi, S_lo, d, Om, dp = (prof[x][k] for x in ("S_hi", "S_lo", "d", "Omega", "dpsi"))
    lp, wp = rotated_half_dims(l, w, min(dp, np.pi / 2))
    eps = p["eps_psi"]
    # The box centre is ahead of the rear axle by l_k ≤ l: turning swings it back along u, and sideways.
    piv_back = l * (1 - np.cos(min(dp, np.pi)))
    piv_front = l * (1 - np.cos(eps))
    piv_lat = l * min(2 * np.sin(min(Om, np.pi) / 2), np.sin(min(dp, np.pi / 2)) + np.sin(eps))
    lo = S_lo - piv_back - lp - p["eps_p"]
    hi = S_hi + piv_front + lp + p["eps_p"]
    lat = d + piv_lat + wp + p["eps_p"]
    u = np.array([np.cos(obj["yaw"]), np.sin(obj["yaw"])])
    centre = np.array([obj["x"], obj["y"]]) + 0.5 * (lo + hi) * u
    return rect_corners(centre, obj["yaw"], 0.5 * (hi - lo), lat)


def pedestrian_disks(obj, tau, p):
    """Proposition 2: two disks, each containing the pedestrian's body at t+τ."""
    o, w = np.array([obj["x"], obj["y"]]), np.array([obj["vx"], obj["vy"]])
    r = np.hypot(obj["length"] / 2, obj["width"] / 2)
    v_max = max(p["v_max"], np.linalg.norm(w) + 0.1)  # never below what is observed
    return [(o + w * tau, p["eps_p"] + p["eps_v"] * tau + 0.5 * p["a"] * tau**2 + r),
            (o, p["eps_p"] + v_max * tau + r)]


# ---------------------------------------------------------------- horizons
def margin(obj, tau, p, prof, ego_pose):
    E = ego_rect(*ego_pose, RHO)
    if obj["kind"] == "vehicle" and p["veh"].get("stopped_yield") and obj["speed"] < 0.5:
        # option Y: a stopped vehicle yields; it stays where it is (its box, enlarged by ε_p)
        e = p["veh"]["eps_p"]
        return rect_distance(E, rect_corners(np.array([obj["x"], obj["y"]]), obj["yaw"], obj["length"] / 2 + e, obj["width"] / 2 + e))
    if obj["kind"] == "vehicle":
        return rect_distance(E, vehicle_rect(obj, tau, p["veh"], prof))
    # the ego misses the intersection of the two disks if it misses either
    return max(point_rect_distance(c, E) - r for c, r in pedestrian_disks(obj, tau, p["ped"]))


def profile(obj, p):
    return vehicle_profile(obj["speed"], p["veh"], T_CAP + 4) if obj["kind"] == "vehicle" else None


def contact_horizon(v_ego, obj, p):
    """H_c for a straight plan at constant speed v_ego along +x: first grid τ with possible contact."""
    prof = profile(obj, p)
    for tau in np.arange(0, T_CAP + GRID / 2, GRID):
        if margin(obj, tau, p, prof, (v_ego * tau, 0.0, 0.0)) <= 0:
            return round(tau, 2)
    return None  # capped


def reaction_horizon(v_ego, obj, p, b_ego=B_EGO):
    """H_r: the last switch time s such that following the plan until s, then braking at b_ego along
    it, touches no predicted region while the ego moves (passive safety)."""
    prof = profile(obj, p)
    t_stop = v_ego / b_ego
    for s in np.arange(0, T_CAP + GRID / 2, GRID):
        for tau in np.arange(0, s + t_stop + GRID / 2, GRID):
            dt = min(max(0.0, tau - s), t_stop)
            x = v_ego * min(tau, s) + v_ego * dt - 0.5 * b_ego * dt**2
            if margin(obj, tau, p, prof, (x, 0.0, 0.0)) <= 0:
                return round(max(0.0, s - GRID), 2)
    return None


def baseline(v_ego, obj, r_veh, cfg=dict(rho0=0.2, e_v=0.3, e_th=0.05, eps_p=0.3, eps_v=0.5, a=2.0)):
    """The PDF baseline with the repository's config, on the same grid."""
    q = cfg["e_v"] + 2 * v_ego * np.sin(cfg["e_th"] / 2)
    rk = 0.5 * np.hypot(obj["length"], obj["width"])
    for tau in np.arange(0, T_CAP + GRID / 2, GRID):
        D = np.hypot(v_ego * tau - (obj["x"] + obj["vx"] * tau), -(obj["y"] + obj["vy"] * tau))
        S = r_veh + rk + cfg["rho0"] + cfg["eps_p"] + (q + cfg["eps_v"]) * tau + 0.5 * cfg["a"] * tau**2
        if D - S <= 0:
            return round(tau, 2)
    return None


def fmt(h):
    return "capped" if h is None else f"{h:.2f}"


def main():
    print(f"ego: R_F = {R_F:.3f} m, ρ = ρ_p + 2 R_F sin(ē_θ/2) = {RHO_P} + {RHO - RHO_P:.3f} = {RHO:.3f} m")
    car = dict(kind="vehicle", length=3.89, width=1.77, yaw=0.0, vy=0.0)
    scenes = [
        ("E1 car in the next lane, abreast, both 10 m/s, centres 3.5 m apart", 10.0,
         dict(car, x=1.312, y=3.5, vx=10.0, speed=10.0)),
        ("E2 following at 4 m/s, 4 m bumper gap", 4.0,
         dict(car, x=EGO["x_max"] + 4.0 + 3.89 / 2, y=0.0, vx=4.0, speed=4.0)),
        ("E3 hand-example pedestrian (30, -8) walking +y at 1.5 m/s, plan 10 m/s", 10.0,
         dict(kind="pedestrian", length=0.6, width=0.6, x=30.0, y=-8.0, vx=0.0, vy=1.5, speed=1.5, yaw=np.pi / 2)),
        ("E4 parked car 15 m ahead, centre 3.0 m to the side, plan 10 m/s", 10.0,
         dict(car, x=15.0, y=-3.0, vx=0.0, speed=0.0)),
    ]
    ph, fo = PARAMS["physical"], PARAMS["foreseeable"]
    fo_t = dict(fo, veh=dict(fo["veh"], turning="slow"))
    fo_y = dict(fo, veh=dict(fo["veh"], stopped_yield=True))
    print(f"\n{'scene':72s} baseline | physical H_c H_r | foreseeable H_c H_r | option T H_c H_r | option Y H_c H_r")
    for name, v, obj in scenes:
        b = baseline(v, obj, R_F)
        cells = [f"{fmt(contact_horizon(v, obj, q)):>6s} {fmt(reaction_horizon(v, obj, q)):>6s}" for q in (ph, fo, fo_t, fo_y)]
        print(f"{name:72s} {fmt(b):>8s} | " + " | ".join(cells))

    # Closed-form check (Proposition 4): following, lead keeps its heading, both sets brake at b_max.
    e2 = scenes[1][2]
    straight = {k: dict(v, kappa=1e-9, eps_psi=0.0) if k == "veh" else v for k, v in PARAMS["foreseeable"].items()}
    m = RHO + EST["eps_p"]
    vf = 4.0 - EST["eps_v"]
    bmax = straight["veh"]["b_max"]
    print(f"\ncheck E2 without heading change: H_c numerical {fmt(contact_horizon(4.0, e2, straight))}, "
          f"closed form {(4.0 - m + vf**2 / (2 * bmax)) / 4.0:.3f}; H_r numerical {fmt(reaction_horizon(4.0, e2, straight))}, "
          f"closed form {(4.0 - m + vf**2 / (2 * bmax) - 4.0**2 / (2 * B_EGO)) / 4.0:.3f}")

    # Sensitivity: E1 against the heading-estimate error ε_ψ and the published lateral assumptions.
    e1 = scenes[0][2]
    print("\nE1 sensitivity, H_c (H_r) in s, foreseeable set, by lateral assumption:")
    for eps_psi in (0.0, 0.01, 0.02, 0.05):
        row = []
        for name, lat in LATERAL.items():
            p = dict(PARAMS["foreseeable"], veh=dict(PARAMS["foreseeable"]["veh"], eps_psi=eps_psi, lat=lat))
            row.append(f"{name}: {fmt(contact_horizon(10.0, e1, p))} ({fmt(reaction_horizon(10.0, e1, p))})")
        print(f"  ε_ψ={eps_psi}: " + " | ".join(row))


if __name__ == "__main__":
    main()
