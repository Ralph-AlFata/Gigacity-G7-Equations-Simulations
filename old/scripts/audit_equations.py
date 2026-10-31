"""Independent numerical/data audit of the existing batch; local files only.

The quartic oracle does not use the production interval checker or its bounds.
It is a floating-point cross-check, not an interval-arithmetic proof.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizonlab.core import Obstacle, clearance_horizon, q_from_tracking_bounds, rear_axle_radius
from horizonlab.dataset import normalize
from horizonlab.replay import estimate_obstacles, load_reference_records, run_replay
from scripts.prepare_runpod_inputs import local_history


def polynomial_horizon(t, xy, obstacles, radius, rho, q, cap):
    """Earliest root of ||relative affine center||^2 - S^2 on each segment.

    Parameterize each segment by u in [0,1] to avoid large absolute-time
    polynomial coefficients. S>=0 makes the squared equation equivalent.
    """
    for left, right, p0, p1 in zip(t[:-1], t[1:], xy[:-1], xy[1:]):
        if left >= cap:
            break
        duration = min(right, cap) - left
        end = p0 + (p1-p0) * duration/(right-left)
        earliest = None
        for o in obstacles:
            a = p0-o.position_m-o.velocity_mps*(o.age_s+left)
            b = end-p0-o.velocity_mps*duration
            age = o.age_s+left
            s = np.array([radius+rho+o.radius_m+o.eps_position_m+q*left
                          +o.eps_velocity_mps*age+.5*o.acceleration_bound_mps2*age**2,
                          (q+o.eps_velocity_mps+o.acceleration_bound_mps2*age)*duration,
                          .5*o.acceleration_bound_mps2*duration**2])
            if np.linalg.norm(a) <= s[0]:
                return float(left)
            # Reverse triangle inequality, independent of production min-distance bound.
            if np.linalg.norm(a)-np.linalg.norm(b) > s.sum()+1e-10:
                continue
            f = -np.convolve(s, s)
            f[:3] += [a@a, 2*(a@b), b@b]
            for root in np.polynomial.polynomial.polyroots(np.trim_zeros(f, 'b')):
                if abs(root.imag) < 1e-7 and -1e-10 <= root.real <= 1+1e-10:
                    u = float(np.clip(root.real, 0, 1))
                    residual = abs(np.linalg.norm(a+b*u)-(s[0]+s[1]*u+s[2]*u*u))
                    if residual < 1e-6:
                        candidate = left+u*duration
                        earliest = candidate if earliest is None else min(candidate, earliest)
        if earliest is not None:
            return float(earliest)
    return float(cap)


def manual_pose(ego, query_us):
    """Direct quaternion SLERP and rotation matrices, independent of normalizer."""
    times = ego.timestamp.to_numpy(np.int64)
    hi = np.searchsorted(times, query_us).clip(0, len(times)-1)
    lo = np.maximum(hi-1, 0)
    fraction = np.divide(query_us-times[lo], times[hi]-times[lo],
                         out=np.zeros(len(query_us)), where=hi!=lo)
    pos = ego[['x', 'y', 'z']].to_numpy(float)
    pos = pos[lo]+fraction[:, None]*(pos[hi]-pos[lo])
    quat = ego[['qx', 'qy', 'qz', 'qw']].to_numpy(float, copy=True)
    quat /= np.linalg.norm(quat, axis=1)[:, None]
    a, b = quat[lo], quat[hi].copy()
    dot = np.einsum('ij,ij->i', a, b)
    b[dot<0] *= -1
    angle = np.arccos(np.clip(abs(dot), 0, 1))
    small = angle < 1e-7
    denom = np.where(small, 1, np.sin(angle))
    w0 = np.where(small, 1-fraction, np.sin((1-fraction)*angle)/denom)
    w1 = np.where(small, fraction, np.sin(fraction*angle)/denom)
    quat = w0[:, None]*a+w1[:, None]*b
    quat /= np.linalg.norm(quat, axis=1)[:, None]
    x,y,z,w = quat.T
    mat = np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                    2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                    2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=1).reshape(-1,3,3)
    return pos, mat, times[hi]


def random_checks(count):
    rng = np.random.default_rng(20261030)
    gaps, statuses = [], Counter()
    max_rotation_difference = 0.
    for i in range(count):
        t = np.r_[0, np.cumsum(rng.uniform(.1, 2, 4))]
        xy = np.vstack([np.zeros(2), np.cumsum(rng.uniform(-10, 10, (4,2)), axis=0)])
        obs = [Obstacle(str(k), rng.uniform(-30, 30, 2), rng.uniform(-5, 5, 2),
                        *rng.uniform(0, 2, 4), age_s=rng.uniform(0, .5))
               for k in range(int(rng.integers(1, 5)))]
        radius, rho, q = rng.uniform(0, 2, 3)
        cap = rng.uniform(.01, t[-1])
        result = clearance_horizon(t, xy, obs, radius, rho, q, cap)
        exact = polynomial_horizon(t, xy, obs, radius, rho, q, cap)
        assert result.H_s <= exact+1e-7, (i, result, exact)
        if result.status == 'boundary_bracket':
            assert exact <= result.interval_end_s+1e-7
        gaps.append(exact-result.H_s)
        statuses[result.status] += 1
        if i < 100:
            angle = rng.uniform(-np.pi, np.pi)
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            offset = rng.uniform(-1000, 1000, 2)
            transformed = [Obstacle(o.track_id, rot@o.position_m+offset, rot@o.velocity_mps,
                o.radius_m, o.eps_position_m, o.eps_velocity_mps, o.acceleration_bound_mps2, o.age_s) for o in obs]
            changed = clearance_horizon(t, xy@rot.T+offset, transformed, radius, rho, q, cap)
            max_rotation_difference = max(max_rotation_difference, abs(result.H_s-changed.H_s))
            assert abs(result.H_s-changed.H_s) < 1e-8
    # Independent velocity-vector inequality and enclosing rear-axle geometry.
    for _ in range(10000):
        ev, eh, vmax = rng.uniform(0, 10), rng.uniform(0, np.pi), rng.uniform(0, 40)
        vref = rng.uniform(0, vmax)
        theta = rng.uniform(-np.pi, np.pi)
        err = rng.uniform(-eh, eh)
        v = vref+rng.uniform(-ev, ev)
        actual = np.linalg.norm(v*np.array([np.cos(theta+err), np.sin(theta+err)])
                             -vref*np.array([np.cos(theta), np.sin(theta)]))
        assert actual <= q_from_tracking_bounds(ev, eh, vmax)+1e-10
        length, width, offset = *rng.uniform(.1, 10, 2), rng.uniform(-5, 5)
        corners = np.array([[offset+s*length/2, r*width/2] for s in [-1,1] for r in [-1,1]])
        assert abs(np.linalg.norm(corners, axis=1).max()-rear_axle_radius(length,width,offset)) < 1e-10
    return dict(random_scenarios=count, failures=0, statuses=dict(statuses),
                max_oracle_minus_solver_s=max(gaps), rotation_translation_checks=100,
                max_rotation_translation_difference_s=max_rotation_difference,
                tracking_inequality_checks=10000, rear_axle_geometry_checks=10000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--random-cases', type=int, default=2000)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'audit_summary.json').exists():
        raise FileExistsError('Preserve the completed audit; choose another output directory')
    inputs = ROOT/'artifacts/runpod_every3s_frontwide'
    source = ROOT/'artifacts/outputs_20261031_003733'
    old = ROOT/'results/alpamayo_every3s_active_scene'
    manifest = json.loads((inputs/'inputs.json').read_text())
    config = json.loads((old/'sample_000/config_used.json').read_text())
    original = pd.read_csv(old/'all_horizons.csv', dtype={'sample_id':str})
    results = dict(random=random_checks(args.random_cases))
    print('Random solver and algebra checks passed', flush=True)
    corrected_root = out/'data_with_pose_support'
    canonical, input_rows, data_rows = {}, [], []
    for clip in sorted({c['clip_id'] for c in manifest['cases']}):
        directory = ROOT/'data/pai_selected_train'/clip
        ego = pd.read_parquet(directory/'raw_ego.parquet').sort_values('timestamp')
        raw = pd.read_parquet(directory/'raw_obstacles.parquet')
        saved = pd.read_csv(directory/'agents.csv', dtype={'track_id':str})
        meta = json.loads((directory/'metadata.json').read_text())
        frames = raw.reference_frame_timestamp_us.to_numpy(np.int64)
        pos, rot, support = manual_pose(ego, frames)
        centers = np.einsum('nij,nj->ni', rot, raw[['center_x','center_y','center_z']].to_numpy())+pos
        independent = pd.DataFrame(dict(track_id=raw.track_id.astype(str),
            t_s=(raw.timestamp_us.to_numpy(np.int64)-meta['time_origin_us'])/1e6,
            x_m=centers[:,0], y_m=centers[:,1],
            radius_m=.5*np.hypot(raw.size_x, raw.size_y))).sort_values(['track_id','t_s']).reset_index(drop=True)
        assert (independent.track_id == saved.track_id).all()
        max_error = float(np.max(np.abs(independent[['x_m','y_m','radius_m']].to_numpy()-saved[['x_m','y_m','radius_m']].to_numpy())))
        assert max_error < 1e-8
        ce, ca, cm = normalize(ego, raw, meta['vehicle_dimensions'])
        cm.update(clip_id=clip, dataset_revision=meta['dataset_revision'], split=meta['split'])
        dest = corrected_root/clip
        dest.mkdir(parents=True)
        ce.to_csv(dest/'ego.csv',index=False)
        ca.to_csv(dest/'agents.csv',index=False)
        (dest/'metadata.json').write_text(json.dumps(cm,indent=2)+'\n')
        canonical[clip] = (pd.read_csv(directory/'ego.csv'), saved, meta)
        data_rows.append(dict(clip_id=clip, raw_obstacle_rows=len(raw), unique_tracks=raw.track_id.nunique(),
                              max_independent_geometry_error_m=max_error,
                              delayed_availability_rows=int((ca.available_s>saved.available_s+1e-10).sum())))
        for case in (c for c in manifest['cases'] if c['clip_id']==clip):
            assert hashlib.sha256((directory/'raw_ego.parquet').read_bytes()).hexdigest()==case['raw_ego_sha256']
            assert hashlib.sha256((inputs/case['input_file']).read_bytes()).hexdigest()==case['input_sha256']
            p,r,s = manual_pose(ego, np.array(case['history_timestamps_us']))
            local_xyz = (p-p[-1])@r[-1]
            local_rot = np.einsum('ij,njk->nik', r[-1].T, r)
            with np.load(inputs/case['input_file']) as a:
                np.testing.assert_allclose(a['ego_history_xyz'][0,0],local_xyz,atol=2e-6)
                np.testing.assert_allclose(a['ego_history_rot'][0,0],local_rot,atol=1e-7)
                assert a['image_frames'].shape[:2]==(4,3)
            old_count = ((saved.t_s<=case['t0_s'])&(saved.available_s<=case['t0_s'])).sum()
            new_count = ((ca.t_s<=case['t0_s'])&(ca.available_s<=case['t0_s'])).sum()
            actual_anchor = int(ego.timestamp[ego.timestamp<=case['t0_us']].iloc[-1])
            local_history(ego, actual_anchor)  # corrected path must have sufficient past support
            input_rows.append(dict(clip_id=clip, t0_s=case['t0_s'],
                history_future_support_s=max(0, float(s.max()-case['t0_us'])/1e6),
                causal_anchor_shift_s=(case['t0_us']-actual_anchor)/1e6,
                premature_obstacle_rows=int(old_count-new_count),
                max_camera_frame_time_s=(max(case['camera_frame_timestamps_us'])-meta['time_origin_us'])/1e6))
    pd.DataFrame(data_rows).to_csv(out/'data_audit.csv', index=False)
    pd.DataFrame(input_rows).to_csv(out/'input_timing_audit.csv', index=False)
    print('Raw geometry and all 12 input histories checked', flush=True)
    oracle_rows, replay_frames = [], []
    for sample in ['0','1','2','recorded']:
        ref_file = old/'recorded_references.jsonl' if sample=='recorded' else source/f'references_sample_{int(sample):03d}.jsonl'
        refs = load_reference_records(ref_file)
        for row in original[original.sample_id==sample].itertuples():
            ego, agents, meta = canonical[row.clip_id]
            exp = next(e for e in config['experiments'] if e['name']==row.experiment)
            obs, error = estimate_obstacles(agents,row.t0_s,exp,config,row.clip_id)
            assert error is None and len(obs)==row.n_obstacles
            ref = refs[(row.clip_id,row.t0_s)]
            t, xy = np.array(ref['tau_s']), np.array(ref['xy_m'])
            live = clearance_horizon(t,xy,obs,row.ego_radius_m,row.rho0_m,row.qbar_mps,row.T_cap_s)
            assert abs(live.H_s-row.H_s) < 1e-12 and live.status==row.status
            exact = polynomial_horizon(t,xy,obs,row.ego_radius_m,row.rho0_m,row.qbar_mps,row.T_cap_s)
            assert row.H_s <= exact+1e-7
            if row.status=='boundary_bracket':
                assert exact <= row.interval_end_s+1e-7
            oracle_rows.append(dict(clip_id=row.clip_id,t0_s=row.t0_s,sample_id=sample,
                experiment=row.experiment,solver_H_s=row.H_s,independent_root_H_s=exact,
                conservatism_s=exact-row.H_s,status=row.status))
        # Sensitivity changes only obstacle support availability, not saved model predictions.
        replay, _ = run_replay(corrected_root,config,out/f'pose_support_{sample}',ref_file)
        replay['sample_id'] = sample
        replay_frames.append(replay)
        print(f'Checked 60 saved horizons and corrected-input replay: {sample}',flush=True)
    oracle = pd.DataFrame(oracle_rows)
    oracle.to_csv(out/'independent_solver_comparison.csv',index=False)
    corrected = pd.concat(replay_frames, ignore_index=True)
    corrected.to_csv(out/'pose_support_sensitivity.csv',index=False)
    results['data'] = data_rows
    results['input_timing'] = input_rows
    results['saved_horizons'] = dict(count=len(oracle), failures=0,
                                    max_conservatism_s=float(oracle.conservatism_s.max()))
    results['baseline'] = {}
    for label, frame in [('original',original),('pose_support_sensitivity',corrected)]:
        base = frame[(frame.experiment=='illustrative_base')&(frame.sample_id!='recorded')]
        results['baseline'][label] = dict(n=len(base),mean_s=float(base.H_s.mean()),
            median_s=float(base.H_s.median()), min_s=float(base.H_s.min()),max_s=float(base.H_s.max()),
            zeros=int(base.H_s.eq(0).sum()), tube_checks=int(base.tube_checked_observations_before_H.sum()),
            tube_violations=int(base.tube_violations_before_H.sum()),shape_violations=int(base.shape_bound_violations_before_H.sum()),
            windows_with_unmodeled_arrival=int(base.future_unmodeled_track_count_before_H.gt(0).sum()))
    files = ['horizonlab/core.py','horizonlab/dataset.py','horizonlab/replay.py',
             'scripts/audit_equations.py','scripts/prepare_runpod_inputs.py','tests/test_audit_regressions.py']
    results['implementation_sha256'] = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    (out/'audit_summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results['baseline'],indent=2),flush=True)


if __name__=='__main__':
    main()
