"""Summarize complete bounded-motion replay results without rerunning inference."""
import argparse
from pathlib import Path
import json
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    h = pd.read_csv(root / 'all_horizons.csv')
    assert len(h) == 240 and np.isfinite(h.H_s).all()
    assert (h.H_s >= 0).all() and (h.H_s <= h.T_cap_s + 1e-9).all()
    assert h.n_unusable_tracks.eq(0).all()
    assert h.n_obstacles.eq(h.n_active_tracks if 'n_active_tracks' in h else h.n_observed_tracks).all()
    if 'n_retired_tracks' in h:
        assert h.n_observed_tracks.eq(h.n_active_tracks + h.n_retired_tracks).all()
    config = json.loads((root / 'sample_000/config_used.json').read_text())
    active = config.get('inventory_policy') == 'active_tracks'
    repo = Path(__file__).resolve().parents[1]
    files = ['horizonlab/core.py', 'horizonlab/replay.py', 'horizonlab/defaults.py', 'scripts/analyze_runpod_results.py',
             'scripts/report_complete_horizons.py', 'tests/test_pipeline.py']
    (root / 'implementation_provenance.json').write_text(json.dumps({
        'sha256': {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in files},
        'config': config}, indent=2) + '\n')
    stats = []
    for (experiment, source), g in h.groupby(['experiment', 'reference_source']):
        stats.append(dict(experiment=experiment, reference_source=source, windows=len(g),
                          calculated=int(g.H_s.notna().sum()), mean_s=g.H_s.mean(), median_s=g.H_s.median(),
                          p05_s=g.H_s.quantile(.05), p10_s=g.H_s.quantile(.1), p90_s=g.H_s.quantile(.9),
                          min_s=g.H_s.min(), max_s=g.H_s.max(), zero_fraction=g.H_s.eq(0).mean(),
                          cap_reached_fraction=g.status.eq('cap_reached').mean(),
                          fraction_H_ge_0_1s=g.H_s.ge(.1).mean(), fraction_H_ge_1s=g.H_s.ge(1).mean(),
                          tube_violations_before_H=int(g.tube_violations_before_H.sum()),
                          tube_checks_before_H=int(g.tube_checked_observations_before_H.sum()),
                          windows_with_new_tracks_before_H=int(g.future_unmodeled_track_count_before_H.gt(0).sum())))
        if 'retrospective_inventory_limited_H_s' in g:
            stats[-1]['retrospective_inventory_limited_mean_s'] = g.retrospective_inventory_limited_H_s.mean()
    stats = pd.DataFrame(stats)
    stats.to_csv(root / 'horizon_statistics.csv', index=False)
    base = h[h.experiment == 'illustrative_base'].copy()
    base['scene'] = base.clip_id.str[:8] + ' / ' + base.t0_s.map(lambda t: f'{t:g} s')
    order = [(clip, t) for clip in sorted(base.clip_id.unique()) for t in sorted(base.t0_s.unique())]
    labels = [f'{clip[:8]} / {t:g} s' for clip, t in order]
    pivot = base.pivot(index='scene', columns='sample_id', values='H_s').reindex(labels)
    pivot = pivot[['recorded', '0', '1', '2']]
    pivot.to_csv(root / 'baseline_horizons_by_timestamp.csv')
    if active:
        base[['clip_id', 't0_s', 'sample_id', 'n_active_tracks', 'n_retired_tracks', 'n_fallback_tracks',
              'future_first_seen_track_count_before_H', 'future_reappearing_track_count_before_H',
              'first_unmodeled_arrival_after_t0_s', 'retrospective_inventory_limited_H_s']].to_csv(
                  root / 'inventory_audit.csv', index=False)
    base[['clip_id', 't0_s', 'sample_id', 'H_s', 'status', 'limiting_track_id',
          'limiting_motion_model', 'limiting_observation_age_s', 'limiting_beta_at_start_m',
          'limiting_center_distance_at_start_m', 'limiting_radius_m', 'initial_margin_m',
          'n_obstacles', 'n_fallback_tracks', 'T_cap_s', 'cap_causes']].to_csv(root / 'limiting_obstacles.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 8), layout='constrained')
    ax.imshow(pivot.to_numpy(), cmap='Blues', vmin=0, vmax=max(1., pivot.max().max()), aspect='auto')
    ax.set_xticks(range(4), ['Recorded', 'Alpamayo 0', 'Alpamayo 1', 'Alpamayo 2'])
    ax.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for col in range(4):
            value = pivot.iloc[row, col]
            ax.text(col, row, f'{value:.3f} s', ha='center', va='center',
                    color='white' if value > .7 else 'black')
    scope = 'Recently observed agents only' if active else 'All previously observed agents'
    ax.set_title(f'Calculated clearance horizons — baseline\n{scope}; all 48 references evaluated\nSingle camera; zero inference delay; uncalibrated uncertainty bounds', pad=15)
    fig.savefig(root / 'horizons_by_timestamp.png', dpi=150)
    plt.close(fig)
    generated = stats[(stats.experiment == 'illustrative_base') & stats.reference_source.str.startswith('alpamayo')].iloc[0]
    lines = ['# Complete clearance-horizon calculation', '',
             '**Start here: [horizons_by_timestamp.png](horizons_by_timestamp.png).** Each cell is a computed horizon, in seconds, starting at the listed dataset timestamp.', '',
             'All 36 generated references and 12 recorded references return numerical horizons under each of five settings: 240 calculations, no unavailable rows. Inventory policy: ' + config.get('inventory_policy', 'all_previously_observed') + '.', '',
             f"Baseline generated references: mean {generated.mean_s:.6f} s; median {generated.median_s:.6f} s; zero fraction {generated.zero_fraction:.1%}; maximum {generated.max_s:.6f} s.", '',
             '## Input rules and mathematical assumptions', '',
             'The core margin equation and continuous-prefix solver are unchanged. Positions, radii, observation age, reference times, and frames are retained. The input adapter has two explicit modes:', '',
             '1. Usable history: endpoint velocity over the recent history. Given position error <= ep at both endpoints and acceleration norm <= A, its terminal velocity error is <= 2 ep / dt + A dt / 2. Use the greater of this bound and the configured velocity-error bound. Constant-velocity tubes still expire at the configured validity time measured from their last observation.', '',
             f"2. Missing/gapped/expired velocity estimate: preserve the last position and actual observation age; use nominal velocity zero, beta(tau) = ep + V(age+tau), with V={config['fallback_speed_bound_mps']:g} m/s. This assumes an absolute center-speed bound over the entire elapsed interval. It does NOT assume the object is stationary. This is a special case of the original tube equation (eps_v=V, A=0). Its separate absolute-speed assumption does not inherit the expired velocity-estimate validity clock. The reference and log caps remain in force.", '',
             '**The speed limit is an explicit exploratory assumption, not a dataset-provided or calibrated physical guarantee.** No future labels were used to choose it. All fallback applications are recorded in track_history_diagnostics.csv. Zero-growth diagnostics retain the fallback speed bound, so they do not remove all uncertainty.', '',
             '## Why many horizons are zero', '',
             'A disappearing object is retained. Its possible-position region grows during the time since it was last seen. If that region overlaps the ego region already at the evaluation time, the equations return H=0. This is a calculated lack of a clearance certificate, not a replacement for a missing result and not evidence of a physical collision. See limiting_obstacles.csv for the responsible track, observation age, radius, and uncertainty.', '',
             '## Retrospective checks', '',
             f"For the 36 baseline model references, there were {int(generated.tube_violations_before_H)} center-tube violations among {int(generated.tube_checks_before_H)} future label checks through H. There were {int(generated.windows_with_new_tracks_before_H)} windows with newly appearing unmodeled track IDs before H. Repeated samples share labels; these are not independent events. Full-cap violations are recorded separately in all_horizons.csv.", '',
             'These checks do not establish coverage for unseen objects or between label samples. Label accuracy, shape bounds, ego tracking bounds, and fallback speed bounds remain assumptions. There are only two hand-selected train clips and overlapping windows; the sample means are descriptive, not population estimates.', '',
             '## Files', '',
             '- baseline_horizons_by_timestamp.csv: all 48 baseline numbers.',
             '- horizon_statistics.csv: all configurations, including zeros and deadline coverage.',
             '- limiting_obstacles.csv: why each baseline horizon ends.',
             '- all_horizons.csv: all 240 calculations and diagnostics.',
             '- sample_000/track_history_diagnostics.csv (and other runs): exact per-object inputs and fallback reasons.',
             '- Each run/config_used.json: exact assumptions.', '',
             'The earlier strict-history results remain in ../alpamayo_every3s and ../alpamayo_every3s_two_observations. Their positive-only mean is not directly comparable to this complete-scene mean: both the velocity bound and the included result population changed.', '']
    if active:
        old = 'A disappearing object is retained. Its possible-position region grows during the time since it was last seen. If that region overlaps the ego region already at the evaluation time, the equations return H=0.'
        new = f"Only objects observed within {config['track_timeout_s']:g} s of the evaluation time enter the horizon. Older tracks are explicitly retired; a returning track starts a new motion-history episode. Newly observed tracks enter immediately with a speed envelope if needed. If an active object's region overlaps the ego region at the starting time, H=0."
        lines = [s.replace(old, new) for s in lines]
        lines += ['## Scene membership and future arrivals', '',
                  f"Timeout is {config['track_timeout_s']:g} s, chosen from the existing gap setting (these labels have about 0.1 s sampling intervals), not optimized against H. Missing labels may mean occlusion or detection failure, not physical departure. The horizons are conditional on the recently observed inventory; they are not full-environment certificates.", '',
                  'Inventory is rebuilt causally at each evaluation timestamp. No future track is inserted into an earlier prediction. A retired track can re-enter when observed again, with its old motion history reset. Retirements and reset reasons are logged per track.', '',
                  'inventory_audit.csv reports future first-seen and returning agents separately. retrospective_inventory_limited_H_s truncates H at the first subsequent unmodeled arrival visible in the recording. This uses hindsight and is only an audit of how long the original inventory remained adequate; it is not an online deadline or a collision horizon. In operation the reference must be rechecked when a new observation changes the scene.', '',
                  'The previous all-seen run remains at ../alpamayo_every3s_complete. Changed inventory scope accounts for much of the difference in horizons; the larger values are not evidence of improved real-world safety.', '']
    (root / 'START_HERE.md').write_text('\n'.join(lines))
    print(pivot.to_string())
    print('\n', stats.to_string(index=False))


if __name__ == '__main__':
    main()
