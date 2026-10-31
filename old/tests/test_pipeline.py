import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from horizonlab.core import Obstacle, clearance_horizon, q_from_tracking_bounds, rear_axle_radius
from horizonlab.dataset import FEATURES, ensure_clip_features, fetch_dataset, normalize
from horizonlab.defaults import default_config
from horizonlab.replay import (estimate_obstacles, load_reference_records, summarize,
                               endpoint_velocity_bound, obstacle_bound_cap, validate_config, run_replay,
                               retrospective_diagnostics)


def legacy_config():
    """Explicitly preserve the original inventory contract in legacy regression tests."""
    config = default_config()
    config.update(inventory_policy='all_previously_observed', obstacle_input_policy='strict_history',
                  min_history_samples=3)
    return config


def obstacle(position, velocity=(0, 0), radius=1, ep=0, ev=0, acc=0, age=0):
    return Obstacle("a", np.array(position, float), np.array(velocity, float), radius, ep, ev, acc, age)


class HorizonTests(unittest.TestCase):
    def test_tracking_bound_and_rear_axle_geometry(self):
        self.assertAlmostEqual(q_from_tracking_bounds(0.2, np.pi, 10), 20.2)
        self.assertAlmostEqual(rear_axle_radius(4.8, 2, 1.2), np.hypot(3.6, 1))
        with self.assertRaises(ValueError):
            q_from_tracking_bounds(0, np.pi + 0.1, 1)

    def test_analytic_closing_case(self):
        r = clearance_horizon([0, 10], [[0, 0], [100, 0]],
                              [obstacle((20, 0), (5, 0), radius=2)], 2, 0, 0, 10, 1e-5)
        self.assertLessEqual(r.H_s, 3.2)
        self.assertLess(3.2 - r.H_s, 1e-5)

    def test_collision_between_clear_endpoints(self):
        r = clearance_horizon([0, 1], [[-2, 0], [2, 0]],
                              [obstacle((0, 0), radius=0.2)], 0.2, 0, 0, 1, 1e-5)
        self.assertLessEqual(r.H_s, 0.4)
        self.assertLess(0.4 - r.H_s, 1e-5)

    def test_tangent_contact_not_missed(self):
        r = clearance_horizon([0, 1], [[-1, 1], [1, 1]],
                              [obstacle((0, 0), radius=0.5)], 0.5, 0, 0, 1, 1e-5)
        self.assertLess(r.H_s, 0.5)
        self.assertGreater(r.H_s, 0.49)
        self.assertNotEqual(r.status, "cap_reached")

    def test_empty_set_and_no_boundary_are_capped(self):
        for obs in [[], [obstacle((10, 0), (10, 0))]]:
            r = clearance_horizon([0, 2], [[0, 0], [2, 0]], obs, 1, 0, 0, 2)
            self.assertEqual(r.H_s, 2)
            self.assertEqual(r.status, "cap_reached")

    def test_initial_overlap_is_zero(self):
        r = clearance_horizon([0, 1], [[0, 0], [1, 0]], [obstacle((0.1, 0))], 1, 0, 0, 1)
        self.assertEqual(r.H_s, 0)

    def test_observation_age_expands_uncertainty(self):
        o = obstacle((1, 0), (2, 0), ep=0.1, ev=0.2, acc=2, age=3)
        np.testing.assert_allclose(o.center(2), [11, 0])
        self.assertAlmostEqual(o.beta(2), 26.1)

    def test_larger_bounds_never_lengthen_horizon(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            pos = rng.uniform([-5, -5], [20, 5])
            vel = rng.uniform(-2, 2, 2)
            r1 = clearance_horizon([0, 2, 4], [[0, 0], [5, 0], [10, 2]],
                                   [obstacle(pos, vel, ep=0.1, ev=0.1, acc=0.1)], 1, 0.1, 0.1, 4)
            r2 = clearance_horizon([0, 2, 4], [[0, 0], [5, 0], [10, 2]],
                                   [obstacle(pos, vel, ep=0.2, ev=0.2, acc=0.2)], 1, 0.2, 0.2, 4)
            self.assertLessEqual(r2.H_s, r1.H_s + 1e-12)

    def test_invalid_reference_rejected(self):
        with self.assertRaises(ValueError):
            clearance_horizon([0, 0], [[0, 0], [1, 0]], [], 1, 0, 0, 1)


class DataTests(unittest.TestCase):
    def test_default_config_enables_logged_scene_lifecycle(self):
        c = default_config()
        validate_config(c)
        self.assertEqual(c['inventory_policy'], 'active_tracks')
        self.assertEqual(c['obstacle_input_policy'], 'bounded_motion')
        self.assertEqual(c['min_history_samples'], 2)
        self.assertEqual(c['track_timeout_s'], .25)

    def fixture(self):
        origin = 1_000_000_000_000
        c = np.sqrt(0.5)
        ego = pd.DataFrame({"timestamp": [origin, origin + 1_000_000],
                            "x": [0, 1], "y": [0, 0], "z": [0, 0],
                            "qx": [0, 0], "qy": [0, 0], "qz": [c, c], "qw": [c, c]})
        obs = pd.DataFrame({"timestamp_us": [origin + 1_000_000], "track_id": ["a"],
                            "reference_frame": ["rig"], "reference_frame_timestamp_us": [origin + 500_000],
                            "center_x": [1], "center_y": [0], "center_z": [0], "size_x": [4], "size_y": [2]})
        dims = {"length": 4.8, "width": 2, "rear_axle_to_bbox_center": 1.2}
        return ego, obs, dims

    def test_shared_time_origin_and_frame_timestamp(self):
        e, o, d = self.fixture()
        ego, agents, meta = normalize(e, o, d)
        self.assertEqual(agents.t_s.iloc[0], 1)
        np.testing.assert_allclose(agents[["x_m", "y_m"]].iloc[0], [0.5, 1])
        self.assertEqual(meta["time_origin_us"], 1_000_000_000_000)

    def test_normalize_with_readonly_pandas_views(self):
        e, o, d = self.fixture()
        original_ego, original_obs = e.copy(deep=True), o.copy(deep=True)
        original_to_numpy = pd.DataFrame.to_numpy

        def readonly_view(frame, *args, **kwargs):
            array = original_to_numpy(frame, *args, **kwargs)
            if not kwargs.get("copy", False):
                array = array.view()
                array.flags.writeable = False
            return array

        with patch.object(pd.DataFrame, "to_numpy", readonly_view):
            ego, agents, meta = normalize(e, o, d)
        np.testing.assert_allclose(agents[["x_m", "y_m"]].iloc[0], [0.5, 1])
        np.testing.assert_allclose(ego.t_s, [0, 1])
        self.assertEqual(agents.t_s.iloc[0], 1)
        self.assertEqual(meta["time_origin_us"], 1_000_000_000_000)
        pd.testing.assert_frame_equal(e, original_ego)
        pd.testing.assert_frame_equal(o, original_obs)

    def test_missing_frame_not_silently_guessed(self):
        e, o, d = self.fixture()
        with self.assertRaises(ValueError):
            normalize(e, o.drop(columns="reference_frame"), d)

    def test_fetch_pins_explicit_revision_before_loading_data(self):
        e, o, dims = self.fixture()
        commit = "a" * 40
        info = Mock(return_value=SimpleNamespace(sha=commit))
        sdk = Mock()
        sdk.repo_id = "nvidia/PhysicalAI-Autonomous-Vehicles"
        sdk.revision = commit
        sdk.features = SimpleNamespace(
            ALL=set(FEATURES),
            get_chunk_feature_filename=lambda chunk, feature: f"{feature}.{chunk}.zip",
        )
        sdk.get_clip_chunk.return_value = 0
        sdk.clip_index = pd.DataFrame(
            {"clip_is_valid": [True], "split": ["test"], "chunk": [0]},
            index=pd.Index(["clip"], name="clip_id"),
        )
        sdk.feature_presence = pd.DataFrame(True, index=sdk.clip_index.index, columns=FEATURES)
        sdk.data_collection = pd.DataFrame(
            {"country": ["Germany"], "month": [1], "radar_config": [np.nan],
             "optional_label": [pd.NA]},
            index=sdk.clip_index.index,
        )
        sdk.get_clip_feature.side_effect = [e, o, dims]
        constructor = Mock(return_value=sdk)
        modules = {
            "physical_ai_av": SimpleNamespace(PhysicalAIAVDatasetInterface=constructor),
            "huggingface_hub": SimpleNamespace(HfApi=lambda: SimpleNamespace(dataset_info=info)),
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", modules), \
                patch("horizonlab.dataset.version", return_value="0.2.3"), \
                patch.object(pd.DataFrame, "to_parquet"):
            manifest = fetch_dataset(tmp, n=1, revision="release-tag")
            saved = json.loads((Path(tmp) / "download_manifest.json").read_text())
            metadata = json.loads((Path(tmp) / "clip" / "metadata.json").read_text())
        info.assert_called_once_with("nvidia/PhysicalAI-Autonomous-Vehicles", revision="release-tag")
        constructor.assert_called_once_with(revision=commit, cache_dir=None, split_name="alpamayo_aligned")
        self.assertEqual(manifest["requested_revision"], "release-tag")
        self.assertEqual(saved["resolved_revision"], commit)
        self.assertEqual(metadata["dataset_revision"], commit)
        self.assertEqual(metadata["collection"], {
            "country": "Germany", "month": 1, "radar_config": None,
            "optional_label": None,
        })
        self.assertEqual(saved["errors"], [])

        # A pinned commit must also work offline, without a Hub revision lookup.
        sdk.get_clip_feature.side_effect = [e, o, dims]
        info.reset_mock()
        info.side_effect = AssertionError("Pinned revision must not query the Hub")
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", modules), \
                patch("horizonlab.dataset.version", return_value="0.2.3"), \
                patch.object(pd.DataFrame, "to_parquet"):
            pinned = fetch_dataset(tmp, n=1, revision=commit)
        self.assertEqual(pinned["resolved_revision"], commit)
        info.assert_not_called()
        sdk.download_clip_features.assert_not_called()

    def test_feature_downloads_use_metadata_and_deduplicate_chunks(self):
        ds = Mock()
        ds.get_clip_chunk.side_effect = lambda clip: {"a": 7, "b": 7, "c": 8}[clip]
        ds.features.get_chunk_feature_filename.side_effect = (
            lambda chunk, feature: f"custom/{feature}/part-{chunk}.parquet"
        )
        ensure_clip_features(ds, ["a", "b", "c"], ["ego", "dimensions"])
        self.assertEqual([call.args[0] for call in ds.download_file.call_args_list], [
            "custom/ego/part-7.parquet", "custom/dimensions/part-7.parquet",
            "custom/ego/part-8.parquet", "custom/dimensions/part-8.parquet",
        ])
        ds.download_clip_features.assert_not_called()

    def test_feature_download_failures_are_not_suppressed(self):
        ds = Mock()
        ds.features.get_chunk_feature_filename.return_value = "required.parquet"
        ds.download_file.return_value = None
        with self.assertRaisesRegex(RuntimeError, "Required feature file"):
            ensure_clip_features(ds, ["a"], ["ego"])
        ds.download_file.side_effect = FileNotFoundError("not cached")
        with self.assertRaisesRegex(FileNotFoundError, "not cached"):
            ensure_clip_features(ds, ["a"], ["ego"])

    def test_anchor_frame_not_transformed_again(self):
        e, o, d = self.fixture()
        o["reference_frame"] = "anchor"
        _, agents, _ = normalize(e, o, d)
        np.testing.assert_allclose(agents[["x_m", "y_m"]].iloc[0], [1, 0])

    def test_velocity_uses_past_only(self):
        a = pd.DataFrame({"t_s": [0, .1, .2, .3], "available_s": [0, .1, .2, .3],
                          "track_id": "a", "x_m": [0, .2, .4, 999], "y_m": 0,
                          "radius_m": 1, "class_name": "car"})
        c = legacy_config()
        obs, error = estimate_obstacles(a, .2, c["experiments"][0], c, "clip")
        self.assertIsNone(error)
        np.testing.assert_allclose(obs[0].velocity_mps, [2, 0])
        obs2, error = estimate_obstacles(a.iloc[:3], 8, c["experiments"][0], c, "clip")
        self.assertEqual(error, "stale_track_bound_expired")
        self.assertFalse(obs2)

    def test_future_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "refs.jsonl"
            path.write_text(json.dumps({"coordinate_frame": "clip_anchor", "reference_point": "rear_axle",
                                        "observation_cutoff_s": 3, "available_time_s": 3, "t0_s": 2}))
            with self.assertRaises(ValueError):
                load_reference_records(path)

    def test_velocity_error_bound_includes_measurement_error_and_acceleration(self):
        # True p(t)=t^2; measured endpoints have opposite 0.3 m errors.
        h = pd.DataFrame(dict(t_s=[0., 1.], x_m=[.3, .7], y_m=[0., 0.]))
        velocity, error = endpoint_velocity_bound(h, .3, 2.)
        self.assertAlmostEqual(error, 1.6)
        self.assertAlmostEqual(np.linalg.norm(velocity - [2., 0.]), error)

    def test_singleton_speed_envelope_has_analytic_horizon(self):
        a = pd.DataFrame(dict(t_s=[0.], available_s=[0.], track_id=['a'],
                              x_m=[10.], y_m=[0.], radius_m=[0.], class_name=['car']))
        c = legacy_config()
        c.update(obstacle_input_policy='bounded_motion', fallback_speed_bound_mps=2.)
        e = dict(c['experiments'][0], eps_position_m=0., object_radius_padding_m=0.)
        diagnostics = []
        obs, err = estimate_obstacles(a, .1, e, c, 'clip', diagnostics)
        self.assertIsNone(err)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].motion_model, 'bounded_speed')
        self.assertAlmostEqual(obs[0].beta(2), 4.2)
        r = clearance_horizon([0, 6], [[0, 0], [0, 0]], obs, 1, 0, 0, 6, 1e-5)
        self.assertLessEqual(r.H_s, 4.4)
        self.assertLess(4.4 - r.H_s, 1e-5)

    def test_stale_fallback_preserves_age_and_does_not_expire_cv_peers(self):
        a = pd.DataFrame(dict(t_s=[0.], available_s=[0.], track_id=['a'],
                              x_m=[100.], y_m=[0.], radius_m=[1.], class_name=['car']))
        c = legacy_config()
        c.update(obstacle_input_policy='bounded_motion', fallback_speed_bound_mps=2.)
        obs, err = estimate_obstacles(a, 8., c['experiments'][0], c, 'clip')
        self.assertIsNone(err)
        self.assertEqual(obs[0].age_s, 8.)
        self.assertAlmostEqual(obs[0].beta(1), .3 + 18.)
        self.assertEqual(obstacle_bound_cap(obs, 6.4), float('inf'))
        self.assertAlmostEqual(obstacle_bound_cap(obs + [obstacle((20, 0), age=1)], 6.4), 5.4)

    def test_bounded_motion_does_not_use_future_or_unavailable_rows(self):
        a = pd.DataFrame(dict(t_s=[0., .1, .2, .3], available_s=[0., .1, 9., .3],
                              track_id='a', x_m=[0., .2, 999., 9999.], y_m=0., radius_m=1., class_name='car'))
        c = legacy_config()
        c.update(min_history_samples=2, obstacle_input_policy='bounded_motion', fallback_speed_bound_mps=40.)
        obs, err = estimate_obstacles(a, .2, c['experiments'][0], c, 'clip')
        self.assertIsNone(err)
        np.testing.assert_allclose(obs[0].velocity_mps, [2., 0.])
        # Two noisy observations 0.1 seconds apart require much more than 0.5 m/s.
        self.assertAlmostEqual(obs[0].eps_velocity_mps, 6.1)

    def test_bounded_motion_requires_explicit_finite_positive_speed_bound(self):
        for speed in [None, 0., -1., float('inf'), float('nan')]:
            c = legacy_config()
            c.update(obstacle_input_policy='bounded_motion', fallback_speed_bound_mps=speed)
            with self.assertRaises(ValueError):
                validate_config(c)

    def active_config(self):
        c = legacy_config()
        c.update(min_history_samples=2, obstacle_input_policy='bounded_motion',
                 fallback_speed_bound_mps=40., inventory_policy='active_tracks', track_timeout_s=.25)
        return c

    def test_track_arrival_and_retirement_follow_observation_availability(self):
        c = self.active_config()
        a = pd.DataFrame(dict(t_s=[0., .3, .2], available_s=[0., .3, 2.],
                              track_id=['old', 'new', 'delayed'], x_m=[0., 10., 99.],
                              y_m=0., radius_m=1., class_name='car'))
        early, _ = estimate_obstacles(a, .2, c['experiments'][0], c, 'clip')
        self.assertEqual([o.track_id for o in early], ['old'])
        audit = []
        current, error = estimate_obstacles(a, .3, c['experiments'][0], c, 'clip', audit)
        self.assertIsNone(error)
        self.assertEqual([o.track_id for o in current], ['new'])
        self.assertEqual(current[0].motion_model, 'bounded_speed')
        self.assertEqual(next(d for d in audit if d['track_id']=='old')['status'], 'retired_timeout')
        empty, error = estimate_obstacles(a, .6, c['experiments'][0], c, 'clip')
        self.assertFalse(empty)
        self.assertEqual(error, 'no_active_agents')

    def test_reappearing_track_resets_velocity_and_shape_episode(self):
        c = self.active_config()
        a = pd.DataFrame(dict(t_s=[0., .1, 1.], available_s=[0., .1, 1.],
                              track_id='a', x_m=[0., 20., 1.], y_m=0.,
                              radius_m=[10., 10., 1.], class_name='car'))
        audit = []
        obs, error = estimate_obstacles(a, 1., c['experiments'][0], c, 'clip', audit)
        self.assertIsNone(error)
        self.assertEqual(obs[0].motion_model, 'bounded_speed')
        np.testing.assert_allclose(obs[0].velocity_mps, [0., 0.])
        self.assertAlmostEqual(obs[0].radius_m, 1.1)
        self.assertTrue(audit[0]['history_reset'])
        self.assertEqual(audit[0]['history_samples'], 1)
        self.assertEqual(audit[0]['episode_start_s'], 1.)

    def test_active_timeout_boundary_and_configuration(self):
        c = self.active_config()
        a = pd.DataFrame(dict(t_s=[0.], available_s=[0.], track_id=['a'],
                              x_m=[10.], y_m=0., radius_m=1., class_name='car'))
        self.assertEqual(len(estimate_obstacles(a, .249, c['experiments'][0], c, 'clip')[0]), 1)
        self.assertEqual(estimate_obstacles(a, .25, c['experiments'][0], c, 'clip')[1], 'no_active_agents')
        for timeout in [None, 0, -1, float('inf'), float('nan')]:
            c['track_timeout_s'] = timeout
            with self.assertRaises(ValueError):
                validate_config(c)

    def test_retrospective_new_and_returning_tracks_do_not_change_snapshot_horizon(self):
        a = pd.DataFrame(dict(t_s=[0., .3, .4, .5], available_s=[0., .3, .6, .5],
                              track_id=['returning', 'known', 'new', 'returning'],
                              x_m=0., y_m=0., radius_m=1.))
        known = obstacle((0., 0.))
        known.track_id = 'known'
        d = retrospective_diagnostics(a, .3, 2., 1., [known])
        self.assertEqual(d['future_first_seen_track_count_before_H'], 1)
        self.assertEqual(d['future_reappearing_track_count_before_H'], 1)
        self.assertAlmostEqual(d['first_unmodeled_arrival_after_t0_s'], .2)
        self.assertAlmostEqual(d['retrospective_inventory_limited_H_s'], .2)

    def test_retrospective_failures_after_horizon_are_reported_separately(self):
        a = pd.DataFrame(dict(t_s=[.1, 1.], track_id=['a', 'a'], x_m=[0., 99.],
                              y_m=[0., 0.], radius_m=[1., 1.]))
        d = retrospective_diagnostics(a, 0., 2., .5, [obstacle((0., 0.))])
        self.assertEqual(d['tube_violations'], 1)
        self.assertEqual(d['tube_violations_before_H'], 0)
        self.assertEqual(d['tube_checked_observations_before_H'], 1)

    def test_gap_uses_reachable_set_and_larger_speed_cannot_increase_horizon(self):
        a = pd.DataFrame(dict(t_s=[0., .4], available_s=[0., .4], track_id=['a', 'a'],
                              x_m=[10., 10.], y_m=[0., 0.], radius_m=[0., 0.], class_name=['car', 'car']))
        c = legacy_config()
        c.update(min_history_samples=2, obstacle_input_policy='bounded_motion', fallback_speed_bound_mps=2.)
        horizons = []
        for speed in [2., 4.]:
            c['fallback_speed_bound_mps'] = speed
            obs, err = estimate_obstacles(a, .5, c['experiments'][0], c, 'clip')
            self.assertIsNone(err)
            self.assertEqual(obs[0].motion_model, 'bounded_speed')
            horizons.append(clearance_horizon([0, 6], [[0, 0], [0, 0]], obs, 1, 0, 0, 6).H_s)
        self.assertLess(horizons[1], horizons[0])

    def test_replay_with_singleton_stale_track_returns_number_and_provenance(self):
        c = legacy_config()
        c.update(obstacle_input_policy='bounded_motion', fallback_speed_bound_mps=2., warmup_s=8.)
        c['experiments'] = c['experiments'][:1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / 'data' / 'clip'
            clip.mkdir(parents=True)
            pd.DataFrame(dict(t_s=[0., 10.], x_m=[0., 0.], y_m=[0., 0.])).to_csv(clip/'ego.csv', index=False)
            pd.DataFrame(dict(t_s=[0.], available_s=[0.], track_id=['a'], x_m=[100.],
                              y_m=[0.], radius_m=[1.], class_name=['car'])).to_csv(clip/'agents.csv', index=False)
            (clip/'metadata.json').write_text(json.dumps(dict(clip_id='clip', kind='synthetic', ego_radius_m=1.)))
            df, _ = run_replay(root/'data', c, root/'out')
            self.assertTrue(df.H_s.notna().all())
            self.assertTrue(df.n_fallback_tracks.eq(1).all())
            self.assertTrue(df.n_unusable_tracks.eq(0).all())
            diagnostics = pd.read_csv(root/'out/track_history_diagnostics.csv')
            self.assertTrue(diagnostics.status.eq('usable_fallback').all())
            self.assertTrue(diagnostics.age_s.ge(8.).all())

    def test_two_point_history_respects_availability(self):
        a = pd.DataFrame(dict(t_s=[0., .1, .2], available_s=[0., .1, 9.],
                              track_id='a', x_m=[0., .2, 999.], y_m=0., radius_m=1., class_name='car'))
        c = legacy_config()
        c['min_history_samples'] = 2
        diagnostics = []
        obs, error = estimate_obstacles(a, .2, c['experiments'][0], c, 'clip', diagnostics)
        self.assertIsNone(error)
        np.testing.assert_allclose(obs[0].velocity_mps, [2., 0.])
        self.assertEqual(diagnostics[0]['past_samples'], 2)
        self.assertEqual(diagnostics[0]['estimator'], 'two_point')

    def test_all_history_failures_are_reported_without_dropping_tracks(self):
        a = pd.DataFrame(dict(t_s=[0., 7.9], available_s=[0., 7.9],
                              track_id=['a', 'b'], x_m=[0., 3.], y_m=0., radius_m=1., class_name='car'))
        c = legacy_config()
        c['min_history_samples'] = 2
        diagnostics = []
        obs, error = estimate_obstacles(a, 8., c['experiments'][0], c, 'clip', diagnostics)
        self.assertFalse(obs)
        self.assertEqual(error, 'stale_track_bound_expired')
        self.assertEqual(len(diagnostics), 2)
        self.assertIn('stale_track_bound_expired', diagnostics[0]['status'])
        self.assertIn('insufficient_track_history', diagnostics[0]['status'])
        self.assertEqual(diagnostics[1]['status'], 'insufficient_track_history')

    def test_invalid_windows_not_removed_from_deadline_denominator(self):
        rows = pd.DataFrame([
            {"clip_id": "c", "experiment": "e", "reference_source": "oracle", "status": "cap_reached",
             "H_s": 2.0, "usable_budget_s": 2.0, "tube_checked_observations": 1, "tube_violations": 0,
             "shape_bound_violations": 0, "future_unmodeled_track_count_before_H": 0, "horizon_compute_ms": 1.0},
            {"clip_id": "c", "experiment": "e", "reference_source": "oracle", "status": "no_observed_agents", "H_s": np.nan},
        ])
        s = summarize(rows, deadlines=[1])["results"][0]
        self.assertEqual(s["deadline_coverage"]["1"]["fraction_all_windows_with_H_ge_deadline"], .5)
        self.assertEqual(s["deadline_coverage"]["1"]["fraction_valid_windows_with_H_ge_deadline"], 1)


if __name__ == "__main__":
    unittest.main()
