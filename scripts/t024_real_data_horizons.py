"""T024: the revised horizons on the real clips (recorded stand-in, offline), against the baseline.

A prototype of the derivation in docs/derivation/horizon_derivation.tex, vectorised with numpy; not
the pipeline's solver (that is a follow-up ticket). For every plan (tick) of every clip:
- baseline H (the PDF equations, horizon_pipeline/horizon/solver.py, repository config);
- revised H_c and H_r (Sections 3-5 of the document) for several variants of the assumptions.

Known objects at t: in_scene and with a velocity (T014's default). No parameter is fitted to the data:
every value is the document's Table 3.

    uv run python scripts/t024_real_data_horizons.py [--out results/t024-revised]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from horizon_pipeline.horizon.config import load_config
from horizon_pipeline.horizon.solver import Objects, Plan, ego_radius, solve
from horizon_pipeline.reference.format import read_reference

SCENES, REFS = Path("data/scene"), Path("data/reference")
DT = 0.01  # integration step of the object envelopes (s)
G = 0.05  # grid of H_c and of the fallback check (s)
S_STEP = 0.1  # grid of the switch time s for H_r (s)
T_CAP = 6.4
T_MAX = 16.0  # envelopes must cover the plan plus the longest fallback
RHO_P, E_THETA, B_E = 0.2, 0.05, 4.0
INFERENCE_S = 2.0


def bosetti(v):
    return 5.22 / np.sqrt(1.0 + (v / 14.84) ** 4)


def make_sets(eps_psi=0.05):
    est = dict(eps_p=0.3, eps_v=0.5, eps_psi=eps_psi)
    return {
        "physical": dict(est=est, veh=dict(a_max=5.0, b_max=10.0, v_max=50.0, kappa=0.2, a_perp=lambda v: 10.0 + 0 * v, regime="turn"),
                         ped=dict(a=6.5, v_max=9.5)),
        "foreseeable": dict(est=est, veh=dict(a_max=3.5, b_max=10.0, v_max=50.0, kappa=0.2, a_perp=bosetti, regime="lane",
                                              lat=dict(rho=2.0, acc=0.2, brk=0.8, mu=0.1)),
                            ped=dict(a=0.6, v_max=2.0)),
    }


# ---------------------------------------------------------------- envelopes
def integ(f):
    return np.concatenate([np.zeros((f.shape[0], 1)), np.cumsum(0.5 * (f[:, 1:] + f[:, :-1]) * DT, axis=1)], axis=1)


def rot_dims(l, w, phi):
    """Lemma 3 (l ≥ w not required: the formula is symmetric in its saturation)."""
    diag = np.hypot(l, w)
    lp = np.where(phi >= np.arctan2(w, l), diag, l * np.cos(phi) + w * np.sin(phi))
    wp = np.where(phi >= np.arctan2(l, w), diag, w * np.cos(phi) + l * np.sin(phi))
    return lp, wp


def vehicle_envelopes(v_hat, l, w, S):
    """Propositions 1 (regime 'turn') and 3 ('lane'): L⁻, L⁺, Λ, and Λ without sideways behaviour or
    heading drift (for option A3), each (K, N) on the DT grid."""
    e, p = S["est"], S["veh"]
    t = np.arange(0, T_MAX + DT / 2, DT)[None, :]
    v, l, w = v_hat[:, None], l[:, None], w[:, None]
    v_hi = np.minimum(p["v_max"], v + e["eps_v"] + p["a_max"] * t)
    v_lo = np.maximum(0.0, v - e["eps_v"] - p["b_max"] * t)
    if p["regime"] == "lane":
        L = p["lat"]
        nu = np.where(t <= L["rho"], L["acc"] * t, np.maximum(0.0, L["acc"] * L["rho"] - L["brk"] * (t - L["rho"])))
        S_hi0 = integ(v_hi)
        behave = np.minimum(integ(np.broadcast_to(nu, v_hi.shape)), 0.5 * p["kappa"] * S_hi0**2) + L["mu"]
        se, ce = np.sin(e["eps_psi"]), np.cos(e["eps_psi"])
        lp, wp = rot_dims(l, w, np.full_like(v_hi, e["eps_psi"]))
        Lm = ce * integ(v_lo) - behave * se - lp - e["eps_p"]
        Lp = S_hi0 + behave * se + lp + e["eps_p"]
        Lam = se * S_hi0 + behave + wp + e["eps_p"]
        Lam_keep = wp + e["eps_p"] + L["mu"]  # (A3): the neighbour keeps its lane
        return Lm, Lp, Lam, np.broadcast_to(Lam_keep, Lam.shape)
    frac = np.linspace(0, 1, 40)[None, None, :]
    vv = np.maximum(v_lo[..., None] + (v_hi - v_lo)[..., None] * frac, 1e-6)
    omega = np.minimum(p["kappa"] * vv, p["a_perp"](vv) / vv).max(axis=2)
    Om = integ(omega)
    dp = e["eps_psi"] + Om
    lo_rate = np.where(dp <= np.pi / 2, v_lo * np.cos(dp), v_hi * np.cos(np.minimum(dp, np.pi)))
    lp, wp = rot_dims(l, w, np.minimum(dp, np.pi / 2))
    Lm = integ(lo_rate) - l * (1 - np.cos(np.minimum(dp, np.pi))) - lp - e["eps_p"]
    Lp = integ(v_hi) + l * (1 - np.cos(e["eps_psi"])) + lp + e["eps_p"]
    piv = l * np.minimum(2 * np.sin(np.minimum(Om, np.pi) / 2), np.sin(np.minimum(dp, np.pi / 2)) + np.sin(e["eps_psi"]))
    Lam = integ(v_hi * np.sin(np.minimum(dp, np.pi / 2))) + piv + wp + e["eps_p"]
    return Lm, Lp, Lam, Lam


class Scene:
    """The known objects at one tick, with their envelopes for one assumption set."""

    def __init__(self, obj, S, yield_stopped=False):
        self.n = len(obj)
        e = S["est"]
        c = obj[["x", "y"]].to_numpy(float)
        yaw = obj["yaw"].to_numpy(float)
        wv = obj[["vx", "vy"]].to_numpy(float)
        l, w = obj["length"].to_numpy(float) / 2, obj["width"].to_numpy(float) / 2
        grp = obj["class_group"].to_numpy()
        speed = np.linalg.norm(wv, axis=1)
        v_along = wv[:, 0] * np.cos(yaw) + wv[:, 1] * np.sin(yaw)
        yaw = np.where(v_along < 0, yaw + np.pi, yaw)  # a box heading may point backwards
        self.c, self.u = c, np.c_[np.cos(yaw), np.sin(yaw)]
        self.n_ax = np.c_[-self.u[:, 1], self.u[:, 0]]
        static = (grp == "static") | ((grp == "vehicle") & yield_stopped & (speed < 0.5))
        self.veh = (grp == "vehicle") & ~static
        self.disk = (grp == "vru") | (grp == "unknown")
        self.static = static
        N = int(round(T_MAX / DT)) + 1
        self.Lm = np.zeros((self.n, N)); self.Lp = np.zeros((self.n, N)); self.Lam = np.zeros((self.n, N))
        self.Lam_keep = np.zeros((self.n, N))
        if self.veh.any():
            i = self.veh
            out = vehicle_envelopes(np.abs(v_along[i]), l[i], w[i], S)
            self.Lm[i], self.Lp[i], self.Lam[i], self.Lam_keep[i] = out
        for arr, val in ((self.Lm, -(l + e["eps_p"])), (self.Lp, l + e["eps_p"]), (self.Lam, w + e["eps_p"]), (self.Lam_keep, w + e["eps_p"])):
            arr[self.static] = val[self.static, None]
        # disks (pedestrians, riders, unknown): Proposition 2
        a = np.where(grp == "unknown", S["veh"]["a_max"], S["ped"]["a"])
        vmax = np.where(grp == "unknown", S["veh"]["v_max"], np.maximum(S["ped"]["v_max"], speed + 0.1))
        self.wv, self.r = wv, np.hypot(l, w)
        self.disk_par = (a, vmax, e)

    def hit(self, ego_xy, ego_yaw, tau, a_e, b_e, lane_keep=False):
        """(K, P) booleans: does the ego rectangle at each pose meet each object's region at τ?"""
        k = np.clip(np.rint(tau / DT).astype(int), 0, self.Lm.shape[1] - 1)
        ue = np.c_[np.cos(ego_yaw), np.sin(ego_yaw)]
        ne = np.c_[-ue[:, 1], ue[:, 0]]
        out = np.zeros((self.n, len(tau)), bool)
        box = ~self.disk
        if box.any():
            Lm, Lp = self.Lm[box][:, k], self.Lp[box][:, k]
            Lam = (self.Lam_keep if lane_keep else self.Lam)[box][:, k]
            uo, no = self.u[box][:, None, :], self.n_ax[box][:, None, :]
            C = self.c[box][:, None, :] + 0.5 * (Lm + Lp)[..., None] * uo
            A, B = 0.5 * (Lp - Lm), Lam
            d = C - ego_xy[None]
            sep = np.zeros(A.shape, bool)
            for ax in (ue[None], ne[None], np.broadcast_to(uo, C.shape), np.broadcast_to(no, C.shape)):
                re = a_e * np.abs((ue[None] * ax).sum(-1)) + b_e * np.abs((ne[None] * ax).sum(-1))
                ro = A * np.abs((uo * ax).sum(-1)) + B * np.abs((no * ax).sum(-1))
                sep |= np.abs((d * ax).sum(-1)) > re + ro
            out[box] = ~sep
        if self.disk.any():
            a, vmax, e = self.disk_par
            i = self.disk
            T = tau[None, :]
            r1 = e["eps_p"] + e["eps_v"] * T + 0.5 * a[i, None] * T**2 + self.r[i, None]
            r2 = e["eps_p"] + vmax[i, None] * T + self.r[i, None]
            c1 = self.c[i][:, None, :] + self.wv[i][:, None, :] * T[..., None]
            c2 = np.broadcast_to(self.c[i][:, None, :], c1.shape)

            def meets(cc, rr):
                rel = cc - ego_xy[None]
                x, y = (rel * ue[None]).sum(-1), (rel * ne[None]).sum(-1)
                dx, dy = np.maximum(np.abs(x) - a_e, 0), np.maximum(np.abs(y) - b_e, 0)
                return dx**2 + dy**2 <= rr**2

            out[i] = meets(c1, r1) & meets(c2, r2)  # the ego misses the intersection if it misses either disk
        return out


# ---------------------------------------------------------------- horizons for one plan
def horizons(plan, scene, fp, rho, lane_keep_in_fallback=False, brake=B_E):
    tau_p, xy_p = plan["tau_s"].to_numpy(), plan[["x", "y"]].to_numpy()
    yaw_p, v_p = plan["yaw"].to_numpy(), plan["speed"].to_numpy()
    cx = 0.5 * (fp["x_min"] + fp["x_max"])
    a_e, b_e = 0.5 * (fp["x_max"] - fp["x_min"]) + rho, fp["y_max"] + rho

    def ego_at(xy, yaw):
        return xy + cx * np.c_[np.cos(yaw), np.sin(yaw)], yaw

    t_cap = min(tau_p[-1], T_CAP)
    grid = np.arange(0, t_cap + G / 2, G)
    xy = np.c_[np.interp(grid, tau_p, xy_p[:, 0]), np.interp(grid, tau_p, xy_p[:, 1])]
    yaw = np.interp(grid, tau_p, yaw_p)
    if scene.n == 0:
        return t_cap, t_cap, True
    hit = scene.hit(*ego_at(xy, yaw), grid, a_e, b_e).any(axis=0)
    if hit.any():
        j = int(np.argmax(hit))
        H_c, capped = (grid[j - 1] if j > 0 else 0.0), False
    else:
        H_c, capped = t_cap, True
    # H_r: fallback F_s = plan until s, then braking at B_E along the plan's path (arc length σ)
    seg = np.linalg.norm(np.diff(xy_p, axis=0), axis=1)
    sigma = np.concatenate([[0], np.cumsum(seg)]) + 1e-9 * np.arange(len(tau_p))
    H_r = 0.0
    for s in np.arange(0, H_c + 1e-9, S_STEP):
        v_s = float(np.interp(s, tau_p, v_p))
        t_stop = v_s / brake
        tf = s + np.arange(G, t_stop + G / 2, G)
        if len(tf):
            sf = float(np.interp(s, tau_p, sigma)) + v_s * (tf - s) - 0.5 * brake * (tf - s) ** 2
            beyond = np.maximum(sf - sigma[-1], 0)
            fx = np.interp(sf, sigma, xy_p[:, 0]) + beyond * np.cos(yaw_p[-1])
            fy = np.interp(sf, sigma, xy_p[:, 1]) + beyond * np.sin(yaw_p[-1])
            fyaw = np.interp(sf, sigma, yaw_p)
            if scene.hit(*ego_at(np.c_[fx, fy], fyaw), tf, a_e, b_e, lane_keep=lane_keep_in_fallback).any():
                break
        H_r = round(s, 2)
    else:
        H_r = H_c
    return H_c, min(H_r, H_c), capped


def followers(obj, ego_row, fp):
    """(A2): vehicles entirely behind the ego's rear bumper, heading within 45° of the ego's."""
    psi = float(ego_row["yaw"])
    u, n = np.array([np.cos(psi), np.sin(psi)]), np.array([-np.sin(psi), np.cos(psi)])
    rel = obj[["x", "y"]].to_numpy(float) - np.array([ego_row["x"], ego_row["y"]])
    cx = rel @ u
    yaw = obj["yaw"].to_numpy(float)
    reach = np.abs(np.cos(yaw - psi)) * obj["length"].to_numpy() / 2 + np.abs(np.sin(yaw - psi)) * obj["width"].to_numpy() / 2
    same_dir = np.cos(yaw - psi) > np.cos(np.radians(45))
    return (obj["class_group"].to_numpy() == "vehicle") & (cx + reach < fp["x_min"]) & same_dir


VARIANTS = {
    "foreseeable (default)": dict(set="foreseeable", eps_psi=0.05),
    "physical": dict(set="physical", eps_psi=0.05),
    "foreseeable, eps_psi 0.02": dict(set="foreseeable", eps_psi=0.02),
    "foreseeable, lane map (eps_psi 0)": dict(set="foreseeable", eps_psi=0.0),
    "foreseeable + Y (stopped yield)": dict(set="foreseeable", eps_psi=0.05, yield_stopped=True),
    "foreseeable + A3 (neighbours keep lane)": dict(set="foreseeable", eps_psi=0.05, a3=True),
    "foreseeable + Y + A3 + lane map": dict(set="foreseeable", eps_psi=0.0, yield_stopped=True, a3=True),
    "foreseeable, without A2": dict(set="foreseeable", eps_psi=0.05, a2=False),
    "foreseeable, fallback 6 m/s2": dict(set="foreseeable", eps_psi=0.05, b_e=6.0),
    "foreseeable, fallback 8 m/s2": dict(set="foreseeable", eps_psi=0.05, b_e=8.0),
}


def stats(h, capped=None):
    h = np.asarray(h)
    out = {"H=0": np.mean(h == 0), "median": np.median(h), "mean": h.mean(), "p10": np.percentile(h, 10),
           "p90": np.percentile(h, 90), f"<{INFERENCE_S:g} s": np.mean(h < INFERENCE_S)}
    if capped is not None:
        out["capped"] = np.mean(capped)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/t024-revised"))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()
    cfg = load_config()
    rho = RHO_P + 2 * 3.790 * np.sin(E_THETA / 2)
    clips = sorted(p.name for p in SCENES.iterdir() if (REFS / p.name / "offline" / "recorded" / "reference.parquet").is_file())
    rows = []
    for clip in clips:
        sd = SCENES / clip / "offline"
        fp = json.loads((sd / "meta.json").read_text())["ego"]["footprint"]
        r_veh = ego_radius(fp)
        ego = pd.read_parquet(sd / "ego.parquet").set_index("tick")
        objects = pd.read_parquet(sd / "objects.parquet")
        objects = objects[objects["in_scene"].fillna(False).astype(bool) & objects[["vx", "vy"]].notna().all(axis=1)]
        by_tick = dict(tuple(objects.groupby("tick")))
        ref, _ = read_reference(REFS / clip / "offline" / "recorded")
        sets = {v: make_sets(VARIANTS[v]["eps_psi"]) for v in args.variants}
        for tick, plan in ref.groupby("tick"):
            obj = by_tick.get(tick, objects.iloc[:0])
            base = solve(Plan.from_rows(plan), Objects.from_rows(obj), r_veh, cfg)
            row = {"clip": clip[:8], "tick": tick, "n_objects": len(obj), "baseline": base.H_s, "baseline_capped": base.capped}
            fol = followers(obj, ego.loc[tick], fp) if len(obj) else np.zeros(0, bool)
            for name in args.variants:
                v = VARIANTS[name]
                o = obj[~fol] if v.get("a2", True) else obj
                sc = Scene(o, sets[name][v["set"]], yield_stopped=v.get("yield_stopped", False))
                hc, hr, cap = horizons(plan, sc, fp, rho, lane_keep_in_fallback=v.get("a3", False), brake=v.get("b_e", B_E))
                row[f"{name}|H_c"], row[f"{name}|H_r"], row[f"{name}|capped"] = hc, hr, cap
            rows.append(row)
        print(f"{clip[:8]}: {len(ref['tick'].unique())} plans done", flush=True)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "horizons.parquet", index=False)

    print(f"\n{len(df)} plans (5 clips, offline, recorded stand-in); H in s; share below the {INFERENCE_S:g} s inference time")
    table = {"baseline (PDF)": stats(df["baseline"], df["baseline_capped"])}
    for name in args.variants:
        table[f"{name}: H_c"] = stats(df[f"{name}|H_c"], df[f"{name}|capped"])
        table[f"{name}: H_r"] = stats(df[f"{name}|H_r"])
    t = pd.DataFrame(table).T
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t.to_string(float_format=lambda x: f"{x:.2f}"))
    sp = []
    for clip in clips:
        e = pd.read_parquet(SCENES / clip / "offline" / "ego.parquet")[["tick", "speed"]]
        sp.append(e.assign(clip=clip[:8]))
    df = df.merge(pd.concat(sp), on=["clip", "tick"], how="left")
    band = pd.cut(df["speed"].fillna(0), [-1, 2, 8, 15, 40], labels=["<2", "2-8", "8-15", ">15"])
    print("\nH_r by ego speed band [m/s]: median (share 0)")
    cols = [n for n in ("foreseeable (default)", "foreseeable, fallback 8 m/s2", "foreseeable + Y + A3 + lane map") if n in args.variants]
    print(pd.DataFrame({n: df.groupby(band, observed=True)[f"{n}|H_r"].agg(lambda x: f"{x.median():.2f} ({np.mean(x == 0):.0%})")
                        for n in cols}).to_string())
    d = "foreseeable (default)"
    if d in args.variants:
        hc, b = df[f"{d}|H_c"], df["baseline"]
        print(f"\nper plan, H_c against baseline: longer {np.mean(hc > b + 1e-9):.0%}, equal {np.mean(np.isclose(hc, b)):.0%}, "
              f"shorter {np.mean(hc < b - 1e-9):.0%}; baseline H=0 plans: {int((b == 0).sum())}, their median H_c "
              f"{hc[b == 0].median():.2f} s, still 0: {np.mean(hc[b == 0] == 0):.0%}")
    print("\nby clip, median H (baseline | foreseeable H_c | foreseeable H_r):")
    d = "foreseeable (default)"
    if d in args.variants:
        print(df.groupby("clip")[["baseline", f"{d}|H_c", f"{d}|H_r"]].median().to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
