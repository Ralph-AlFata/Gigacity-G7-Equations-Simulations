"""T024 evidence: how objects really move against the constant-velocity prediction, in their own frame.

Prints the numbers quoted in docs/derivation/horizon_derivation.tex (T024). They are used to
DIAGNOSE what is structurally wrong with the baseline and as sanity checks, never to fit a parameter:
five clips are far too few for that (Section 1 of the document).
1. heading of moving vehicles against their velocity direction (do they slide sideways?);
2. prediction residuals o(t+τ) − ô(τ), split along the box heading (longitudinal) and across it
   (lateral), for two nominal predictions: the full velocity ŵτ, and the velocity projected on the
   heading (vehicles do not slide sideways);
3. physical motion bounds from 1 s chords (noise-robust): longitudinal and lateral acceleration,
   curvature, pedestrian speed;
4. the baseline β(τ) = ε_p + ε_v τ + ½ ā τ² against the residuals;
5. the H = 0 plans of a T014 run: disk overlap at τ = 0 against the real box gap.

Uses only offline scenes; the residuals look into the future, which is allowed here: this is
calibration of the environment model, not something the car computes.

    uv run python scripts/spike_t024_object_motion.py [--results results/t014-baseline]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from horizon_pipeline.scene.checks import parked_spread

SCENES = Path("data/scene")
TAUS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.4]
Q = [0.5, 0.9, 0.99, 0.999]
SPEED_BINS = [(0, 0.5, "stopped <0.5"), (0.5, 5, "slow 0.5-5"), (5, 15, "mid 5-15"), (15, 99, "fast >15")]


def load(clip: Path) -> pd.DataFrame:
    o = pd.read_parquet(clip / "offline" / "objects.parquet")
    o["clip"] = clip.name[:8]
    o["in_scene"] = o["in_scene"].fillna(False).astype(bool)
    return o


def qtab(s: pd.Series) -> str:
    return " ".join(f"{s.quantile(q):6.2f}" for q in Q) + f" {s.max():7.2f}"


def residuals(o: pd.DataFrame, tau: float) -> pd.DataFrame:
    """For every in-scene object-tick with a velocity: the residual at t+τ in the frame of the box at t."""
    k = int(round(tau * 10))
    now = o[o["in_scene"] & o["vx"].notna()][["clip", "track_id", "tick", "class_group", "x", "y", "yaw", "vx", "vy", "speed", "length", "width"]]
    later = o[["clip", "track_id", "tick", "x", "y", "yaw", "length", "width"]].rename(
        columns={"x": "x2", "y": "y2", "yaw": "yaw2", "length": "length2", "width": "width2"})
    later["tick"] -= k
    m = now.merge(later, on=["clip", "track_id", "tick"])
    u = np.c_[np.cos(m["yaw"]), np.sin(m["yaw"])]
    n = np.c_[-u[:, 1], u[:, 0]]
    w = m[["vx", "vy"]].to_numpy()
    d = m[["x2", "y2"]].to_numpy() - m[["x", "y"]].to_numpy()
    r_full = d - w * tau  # nominal ô + ŵτ (the baseline)
    v_u = (w * u).sum(1)
    r_head = d - (v_u[:, None] * u) * tau  # nominal ô + (ŵ·u) u τ: no sideways velocity
    m["norm_full"] = np.linalg.norm(r_full, axis=1)
    m["lon_full"], m["lat_full"] = (r_full * u).sum(1), np.abs((r_full * n).sum(1))
    m["lon_head"], m["lat_head"] = (r_head * u).sum(1), np.abs((r_head * n).sum(1))
    m["norm_head"] = np.linalg.norm(r_head, axis=1)
    m["v_lat"] = np.abs((w * n).sum(1))
    # Box overhang: how far the true box at t+τ sticks out of the nominal box (heading-projected
    # centre, heading and size at t), along u (front, back) and across it. This is what a per-axis
    # inflation of the predicted box must cover, rotation included.
    c2, s2 = np.cos(m["yaw2"].to_numpy()), np.sin(m["yaw2"].to_numpy())
    l2, w2 = m["length2"].to_numpy() / 2, m["width2"].to_numpy() / 2
    centre = m[["x2", "y2"]].to_numpy() - (m[["x", "y"]].to_numpy() + v_u[:, None] * u * tau)
    cl, cn = [], []
    for a, b in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        corner = centre + np.c_[a * l2 * c2 - b * w2 * s2, a * l2 * s2 + b * w2 * c2]
        cl.append((corner * u).sum(1))
        cn.append((corner * n).sum(1))
    cl, cn = np.array(cl), np.array(cn)
    l1, w1 = m["length"].to_numpy() / 2, m["width"].to_numpy() / 2
    m["over_front"] = np.clip(cl.max(0) - l1, 0, None)
    m["over_back"] = np.clip(-cl.min(0) - l1, 0, None)
    m["over_lat"] = np.clip(np.abs(cn).max(0) - w1, 0, None)
    return m


def chords(o: pd.DataFrame) -> pd.DataFrame:
    """1 s chords c1 = o(t+1) − o(t), c2 = o(t+2) − o(t+1): mean speeds, their change, heading change."""
    base = o[["clip", "track_id", "tick", "class_group", "x", "y"]]
    p = [base]
    for k in (10, 20):
        nxt = base[["clip", "track_id", "tick", "x", "y"]].rename(columns={"x": f"x{k}", "y": f"y{k}"})
        nxt["tick"] -= k
        p.append(nxt)
    m = p[0].merge(p[1], on=["clip", "track_id", "tick"]).merge(p[2], on=["clip", "track_id", "tick"])
    c1 = m[["x10", "y10"]].to_numpy() - m[["x", "y"]].to_numpy()
    c2 = m[["x20", "y20"]].to_numpy() - m[["x10", "y10"]].to_numpy()
    m["v1"], m["v2"] = np.linalg.norm(c1, axis=1), np.linalg.norm(c2, axis=1)
    m["a_lon"] = m["v2"] - m["v1"]  # m/s², over 1 s
    m["a_vec"] = np.linalg.norm(c2 - c1, axis=1)
    cross = c1[:, 0] * c2[:, 1] - c1[:, 1] * c2[:, 0]
    m["dpsi"] = np.abs(np.arctan2(cross, (c1 * c2).sum(1)))  # rad over 1 s
    vm = 0.5 * (m["v1"] + m["v2"])
    m["a_lat"] = vm * m["dpsi"]
    m["kappa"] = m["dpsi"] / vm.where(vm > 0)
    m["v_mean"] = vm
    return m


def box_gap(ego_xy, ego_yaw, fp, obj):
    """Planar distance between the ego footprint and an object box (0 if they overlap)."""
    def corners(cx, cy, yaw, x0, x1, y0, y1):
        c, s = np.cos(yaw), np.sin(yaw)
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return pts @ np.array([[c, s], [-s, c]]) + [cx, cy]

    A = corners(*ego_xy, ego_yaw, fp["x_min"], fp["x_max"], fp["y_min"], fp["y_max"])
    L, W = obj["length"] / 2, obj["width"] / 2
    B = corners(obj["x"], obj["y"], obj["yaw"], -L, L, -W, W)

    def sep(P, Q):  # separating axis test on P's edges
        for i in range(4):
            e = P[(i + 1) % 4] - P[i]
            ax = np.array([-e[1], e[0]])
            pp, qq = P @ ax, Q @ ax
            if pp.max() < qq.min() or qq.max() < pp.min():
                return True
        return False

    if not (sep(A, B) or sep(B, A)):
        return 0.0

    def pt_seg(p, a, b):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        return np.linalg.norm(p - (a + t * ab))

    return min(min(pt_seg(p, Q[i], Q[(i + 1) % 4]) for p in P for i in range(4)) for P, Q in ((A, B), (B, A)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/t014-baseline"))
    args = ap.parse_args()
    clips = sorted(p for p in SCENES.iterdir() if (p / "offline" / "objects.parquet").is_file())
    o = pd.concat([load(c) for c in clips], ignore_index=True)
    veh, vru = o["class_group"] == "vehicle", o["class_group"] == "vru"
    print(f"clips {len(clips)}, object rows {len(o)}, tracks {o.groupby(['clip', 'track_id']).ngroups} "
          f"(vehicle {o[veh].groupby(['clip','track_id']).ngroups}, vru {o[vru].groupby(['clip','track_id']).ngroups})")

    # 0. label noise: parked tracks
    spread = pd.concat({c: parked_spread(o[o["clip"] == c]) for c in o["clip"].unique()})  # index (clip, track)
    print(f"\n[0] parked tracks {len(spread)}: position RMS about their mean p50 {spread.median():.3f} m, "
          f"p95 {spread.quantile(.95):.3f} m, max {spread.max():.3f} m")

    yaw_rms = []
    for (c, tr), g in o[veh].groupby(["clip", "track_id"]):
        if (c, tr) in spread.index:
            d = np.angle(np.exp(1j * (g["yaw"] - np.angle(np.exp(1j * g["yaw"]).mean()))))
            yaw_rms.append(np.sqrt(np.mean(d**2)))
    yaw_rms = np.degrees(np.array(yaw_rms))
    print(f"    parked vehicles ({len(yaw_rms)}): box-yaw RMS about the track mean p50 {np.median(yaw_rms):.2f}°, p95 {np.quantile(yaw_rms, .95):.2f}°")

    # 1. heading vs velocity direction
    mv = o[veh & o["in_scene"] & (o["speed"] > 8)]
    ang = np.abs(np.angle(np.exp(1j * (np.arctan2(mv["vy"], mv["vx"]) - mv["yaw"]))))
    ang = np.minimum(ang, np.pi - ang)  # a box yaw may point backwards
    print(f"\n[1] moving vehicles (>8 m/s, n={len(mv)}): |velocity direction − box heading| (deg) "
          f"p50 {np.degrees(np.median(ang)):.1f}, p90 {np.degrees(np.quantile(ang, .9)):.1f}, p99 {np.degrees(np.quantile(ang, .99)):.1f}")

    # 2. residuals
    print(f"\n[2] residual quantiles {[f'p{q*100:g}' for q in Q]} + max (m)")
    rows = []
    for tau in TAUS:
        r = residuals(o, tau)
        rows.append(r.assign(tau=tau))
    R = pd.concat(rows, ignore_index=True)
    rv = R[R["class_group"] == "vehicle"]
    r1 = rv[rv["tau"] == 1.0]
    print(f"    |ŵ·n| vehicles at t: {qtab(r1['v_lat'])}")
    for label, col in [("vehicle |r| full-velocity (baseline)", "norm_full"),
                       ("vehicle lateral, full velocity", "lat_full"),
                       ("vehicle lateral, heading-projected", "lat_head"),
                       ("vehicle longitudinal ahead (+), heading-proj.", "lon_head+"),
                       ("vehicle longitudinal behind (−), heading-proj.", "lon_head-")]:
        print(f"  {label}")
        for tau in TAUS:
            g = rv[rv["tau"] == tau]
            s = g["lon_head"].clip(lower=0) if col == "lon_head+" else (-g["lon_head"]).clip(lower=0) if col == "lon_head-" else g[col]
            print(f"    τ={tau:3.1f}  n={len(g):6d} tracks={g.groupby(['clip','track_id']).ngroups:4d}  {qtab(s)}")
    print("  vehicle lateral heading-projected / longitudinal |.| p99.9 by speed at t (m)")
    for lo, hi, name in SPEED_BINS:
        line = []
        for tau in [1.0, 2.0, 4.0, 6.4]:
            g = rv[(rv["tau"] == tau) & (rv["speed"] >= lo) & (rv["speed"] < hi)]
            if len(g):
                line.append(f"τ={tau}: lat {g['lat_head'].quantile(.999):5.2f} lon {g['lon_head'].abs().quantile(.999):5.2f} (n={len(g)})")
        print(f"    {name:13s} " + " | ".join(line))
    rp = R[R["class_group"] == "vru"]
    lab = o.drop_duplicates(["clip", "track_id"]).set_index(["clip", "track_id"])["label_class"]
    rp = rp.assign(label=lab.reindex(pd.MultiIndex.from_frame(rp[["clip", "track_id"]])).to_numpy())
    for name in ["person", "rider"]:
        print(f"  {name} |r| full velocity")
        for tau in TAUS:
            g = rp[(rp["tau"] == tau) & (rp["label"] == name)]
            if len(g):
                print(f"    τ={tau:3.1f}  n={len(g):6d} tracks={g.groupby(['clip','track_id']).ngroups:4d}  {qtab(g['norm_full'])}")

    # 3. physical bounds from 1 s chords
    C = chords(o)
    cv = C[(C["class_group"] == "vehicle") & (C["v1"] > 1)]
    print(f"\n[3] 1 s chords, vehicles moving >1 m/s (n={len(cv)}):")
    print(f"    a_lon (m/s²) p0.1 {cv['a_lon'].quantile(.001):.2f}  p1 {cv['a_lon'].quantile(.01):.2f}  "
          f"p99 {cv['a_lon'].quantile(.99):.2f}  p99.9 {cv['a_lon'].quantile(.999):.2f}  min {cv['a_lon'].min():.2f}  max {cv['a_lon'].max():.2f}")
    cm = cv[(cv["v1"] > 3) & (cv["v2"] > 3)]
    print(f"    a_lat (m/s², >3 m/s) p99 {cm['a_lat'].quantile(.99):.2f}  p99.9 {cm['a_lat'].quantile(.999):.2f}  max {cm['a_lat'].max():.2f}")
    ch = cv[(cv["v1"] > 2) & (cv["v2"] > 2)]
    print(f"    heading change over 1 s (deg, both chords >2 m/s) p99 {np.degrees(ch['dpsi'].quantile(.99)):.1f}  p99.9 {np.degrees(ch['dpsi'].quantile(.999)):.1f}  max {np.degrees(ch['dpsi'].max()):.1f}")
    ck = ch
    print(f"    curvature (1/m, >2 m/s) p99 {ck['kappa'].quantile(.99):.3f}  p99.9 {ck['kappa'].quantile(.999):.3f}  max {ck['kappa'].max():.3f}")
    print(f"    speed (m/s) p99 {cv['v_mean'].quantile(.99):.1f}  max {cv['v_mean'].max():.1f}")
    C = C.assign(label=lab.reindex(pd.MultiIndex.from_frame(C[["clip", "track_id"]])).to_numpy())
    for name in ["person", "rider"]:
        cp = C[C["label"] == name]
        print(f"  {name} (n={len(cp)}): speed p50 {cp['v1'].median():.2f}  p99 {cp['v1'].quantile(.99):.2f}  p99.9 {cp['v1'].quantile(.999):.2f}  max {cp['v1'].max():.2f} m/s; "
              f"|Δv| over 1 s p99 {cp['a_vec'].quantile(.99):.2f}  p99.9 {cp['a_vec'].quantile(.999):.2f} m/s²")
    cp = C[C["class_group"] == "vru"]
    print(f"  vru all (n={len(cp)}): speed p50 {cp['v1'].median():.2f}  p99 {cp['v1'].quantile(.99):.2f}  p99.9 {cp['v1'].quantile(.999):.2f}  max {cp['v1'].max():.2f} m/s; "
          f"|Δv| over 1 s p99 {cp['a_vec'].quantile(.99):.2f}  p99.9 {cp['a_vec'].quantile(.999):.2f} m/s²")

    # 4. baseline β coverage
    print("\n[4] baseline β(τ) = 0.3 + 0.5τ + τ²: share of |r| (full velocity) above β")
    for tau in TAUS:
        beta = 0.3 + 0.5 * tau + tau**2
        gv, gp = rv[rv["tau"] == tau], rp[rp["tau"] == tau]
        print(f"    τ={tau:3.1f}  β={beta:5.2f} m  vehicle {np.mean(gv['norm_full'] > beta):.4f}  vru {np.mean(gp['norm_full'] > beta):.4f}")

    # 5. H = 0 plans: disks against boxes
    if args.results.is_dir():
        gaps, lat = [], []
        for clip in clips:
            f = args.results / clip.name / "offline" / "recorded" / "horizons.parquet"
            if not f.is_file():
                continue
            h = pd.read_parquet(f)
            h = h[(h["H_s"] == 0) & h["limiting_track_id"].notna()]
            fp = json.loads((clip / "offline" / "meta.json").read_text())["ego"]["footprint"]
            ego = pd.read_parquet(clip / "offline" / "ego.parquet").set_index("tick")
            obj = o[o["clip"] == clip.name[:8]].set_index(["tick", "track_id"])
            for _, row in h.iterrows():
                ob = obj.loc[(row["tick"], row["limiting_track_id"])]
                e = ego.loc[row["tick"]]
                gaps.append(box_gap((e["x"], e["y"]), e["yaw"], fp, ob))
                lat.append(abs(ob["rel_y"]))
        g = pd.Series(gaps)
        print(f"\n[5] H = 0 plans ({len(g)}): real box gap to the limiting object at τ = 0: p10 {g.quantile(.1):.2f}  "
              f"p50 {g.median():.2f}  p90 {g.quantile(.9):.2f} m; touching (gap 0): {np.mean(g == 0):.1%}; "
              f"gap > 0.5 m: {np.mean(g > 0.5):.1%}; |rel_y| median {np.median(lat):.2f} m")


if __name__ == "__main__":
    main()
