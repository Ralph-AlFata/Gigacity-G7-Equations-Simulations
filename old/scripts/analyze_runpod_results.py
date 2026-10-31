"""Validate a downloaded batch and compare all samples using local data only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizonlab.replay import load_reference_records, run_replay, recorded_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'examples/active_scene_config.json')
    args = parser.parse_args()
    source, out = args.input.resolve(), args.out.resolve()
    provenance = json.loads((source / 'provenance.json').read_text())
    manifest = provenance['manifest']
    local = json.loads((ROOT / 'artifacts/runpod_every3s_frontwide/inputs.json').read_text())
    if manifest != local:
        raise ValueError('Returned manifest differs from packaged inputs')
    cases = {(c['clip_id'], c['t0_s']): c for c in manifest['cases']}
    samples = provenance['samples']
    expected = len(cases) * samples
    if (source / 'COMPLETE').read_text().strip() != f'{expected} trajectories':
        raise ValueError('Incomplete inference run')
    references = []
    rows = []
    for sample in range(samples):
        refs = load_reference_records(source / f'references_sample_{sample:03d}.jsonl')
        if set(refs) != set(cases):
            raise ValueError('Missing or extra inference timestamps')
        references.append(refs)
        for key, ref in refs.items():
            case = cases[key]
            meta_path = ROOT / 'data/pai_selected_train' / case['clip_id']
            if hashlib.sha256((meta_path / 'raw_ego.parquet').read_bytes()).hexdigest() != case['raw_ego_sha256']:
                raise ValueError('Local ego labels changed')
            np.testing.assert_allclose(ref['tau_s'], np.arange(65) * .1)
            np.testing.assert_allclose(ref['xy_m'][0], case['origin_in_clip_anchor_m'][:2])
            with np.load(source / f"{case['case_id']}_sample_{sample:03d}.npz") as raw:
                xyz = raw['predicted_xyz_local']
                assert xyz.shape == (64, 3) and np.isfinite(xyz).all()
                anchor = xyz @ np.asarray(case['rotation_local_to_clip_anchor']).T + case['origin_in_clip_anchor_m']
                np.testing.assert_allclose(anchor[:, :2], np.asarray(ref['xy_m'])[1:], atol=1e-6)
            truth = np.asarray(case['ground_truth_xyz_clip_anchor'])[:, :2]
            errors = np.linalg.norm(np.asarray(ref['xy_m'])[1:len(truth)+1] - truth, axis=1)
            rows.append(dict(clip_id=key[0], t0_s=key[1], sample_id=sample,
                             duration_s=len(truth)/10, ade_m=errors.mean(),
                             last_compared_error_m=errors[-1]))
    metrics = pd.DataFrame(rows).sort_values(['clip_id', 't0_s', 'sample_id'])
    returned = pd.read_csv(source / 'comparison_metrics.csv').sort_values(['clip_id', 't0_s', 'sample_id'])
    assert list(zip(returned.clip_id, returned.t0_s, returned.sample_id)) == list(zip(metrics.clip_id, metrics.t0_s, metrics.sample_id))
    np.testing.assert_allclose(returned.ade_m, metrics.ade_m, atol=1e-6)
    np.testing.assert_allclose(returned.last_compared_error_m, metrics.last_compared_error_m, atol=1e-6)
    out.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(out / 'verified_comparisons.csv', index=False)
    config = json.loads(args.config.read_text())
    horizon_frames = []
    for sample in range(samples):
        df, _ = run_replay(ROOT / 'data/pai_selected_train', config,
                           out / f'sample_{sample:03d}', source / f'references_sample_{sample:03d}.jsonl')
        df['sample_id'] = sample
        horizon_frames.append(df)
    oracle_file = out / 'recorded_references.jsonl'
    with oracle_file.open('w') as stream:
        for key in cases:
            ego = pd.read_csv(ROOT / 'data/pai_selected_train' / key[0] / 'ego.csv')
            tau, xy = recorded_reference(ego, key[1], 6.4)
            record = dict(references[0][key], tau_s=tau.tolist(), xy_m=xy.tolist(),
                          source='recorded_future_oracle')
            for field in ('sample_id', 'seed', 'inference_latency_s'):
                record.pop(field, None)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
    oracle, _ = run_replay(ROOT / 'data/pai_selected_train', config, out / 'recorded', oracle_file)
    oracle['sample_id'] = 'recorded'
    horizons = pd.concat([*horizon_frames, oracle], ignore_index=True)
    horizons.to_csv(out / 'all_horizons.csv', index=False)
    figures = []
    for clip in sorted({k[0] for k in cases}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ax, key in zip(axes.flat, sorted(k for k in cases if k[0] == clip)):
            case = cases[key]
            origin = np.asarray(case['origin_in_clip_anchor_m'])[:2]
            truth = np.asarray(case['ground_truth_xyz_clip_anchor'])[:, :2] - origin
            ax.plot(*np.vstack([np.zeros(2), truth]).T, color='black', linewidth=2.5, label='Recorded')
            for sample in range(samples):
                ref = references[sample][key]
                xy = np.asarray(ref['xy_m']) - origin
                n = len(truth) + 1
                line, = ax.plot(*xy[:n].T, label=f'Sample {sample}')
                if n < len(xy):
                    ax.plot(*xy[n-1:].T, '--', color=line.get_color())
            ax.scatter(0, 0, color='black', marker='*', s=70)
            ax.set_title(f"t = {key[1]:g} s; observed future {len(truth)/10:g} s")
            ax.set_xlabel('Clip-frame X offset (m)')
            ax.set_ylabel('Clip-frame Y offset (m)')
            ax.axis('equal')
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
        fig.suptitle(f'{clip[:8]} — front-wide camera only\nDashed prediction tails extend beyond the recording; these plots show paths, not clearance', fontsize=13)
        fig.tight_layout()
        name = f'{clip[:8]}_all_timestamps.png'
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        figures.append(name)
    baseline = horizons[horizons.experiment == 'illustrative_base']
    table = metrics.groupby(['clip_id', 't0_s', 'duration_s']).ade_m.agg(['min', 'mean', 'max'])
    lines = ['# Local analysis of Runpod trajectories', '',
             f'Validated {expected} trajectories at {len(cases)} timestamps. All exported frames, timestamps, input-manifest identity, and displacement metrics checked.', '',
             'Single front-wide camera; offline labeled ego history; three samples per anchor. Zero-latency reference comparison. Recorded-path disagreement is neither a safety test nor a measured controller tracking error.', '',
             '## Recorded-path disagreement', '',
             '| Clip | Start (s) | Compared future (s) | ADE min / mean / max over three samples (m) |',
             '|---|---:|---:|---|']
    for (clip, t0, duration), row in table.iterrows():
        lines.append(f"| {clip[:8]} | {t0:g} | {duration:g} | {row['min']:.2f} / {row['mean']:.2f} / {row['max']:.2f} |")
    lines += ['', 'The minimum is descriptive only; no sample was selected using future labels.', '',
              '## Horizon results', '', 'Baseline status counts (36 generated references + 12 matched recorded references):', '',
              '```', baseline.status.value_counts().to_string(), '```', '',
              f"Unavailable horizons remain unavailable, not zero. All five sensitivity settings and all samples are retained in all_horizons.csv. Minimum history samples: {config['min_history_samples']}. Inventory policy: {config.get('inventory_policy', 'all_previously_observed')}; detailed inclusions and retirements are in each run's track_history_diagnostics.csv.", '']
    for name in figures:
        lines += [f'![All timestamps]({name})', '']
    (out / 'README.md').write_text('\n'.join(lines))
    print(table.to_string())
    print('\nBaseline horizon statuses:\n', baseline.status.value_counts().to_string())
    print('\nReport:', out / 'README.md')


if __name__ == '__main__':
    main()
