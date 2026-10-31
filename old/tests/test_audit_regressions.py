import unittest

import numpy as np
import pandas as pd

from horizonlab.core import clearance_horizon
from horizonlab.dataset import normalize
from scripts.prepare_runpod_inputs import local_history


class AuditRegressions(unittest.TestCase):
    def test_interpolated_obstacle_waits_for_pose_support(self):
        ego = pd.DataFrame(dict(timestamp=[0, 1_000_000], x=[0., 10.], y=[0., 0.],
                                z=[0., 0.], qx=[0., 0.], qy=[0., 0.],
                                qz=[0., 0.], qw=[1., 1.]))
        obs = pd.DataFrame(dict(timestamp_us=[500_000, 500_000, 0],
            reference_frame_timestamp_us=[500_000, 500_000, 0],
            reference_frame=['rig', 'anchor', 'rig'], track_id=['a', 'b', 'c'],
            center_x=[2., 2., 2.], center_y=[0., 0., 0.], center_z=[0., 0., 0.],
            size_x=[2., 2., 2.], size_y=[2., 2., 2.]))
        _, agents, metadata = normalize(ego, obs, dict(length=4., width=2., rear_axle_to_bbox_center=1.))
        agents = agents.set_index('track_id')
        self.assertEqual(agents.loc['a', 'x_m'], 7.)
        self.assertEqual(agents.loc['a', 'available_s'], 1.)
        self.assertEqual(agents.loc['b', 'available_s'], .5)
        self.assertEqual(agents.loc['c', 'available_s'], 0.)
        self.assertTrue(metadata['availability_is_lower_bound'])

    def test_history_rejects_interpolation_beyond_prediction_time(self):
        t = np.arange(0, 2_100_000, 100_000)
        ego = pd.DataFrame(dict(timestamp=t, x=t/1e6, y=0., z=0., qx=0., qy=0., qz=0., qw=1.))
        with self.assertRaisesRegex(ValueError, 'past-only support'):
            local_history(ego, 1_950_000)
        first = local_history(ego, 1_900_000)
        ego.loc[ego.timestamp > 1_900_000, 'x'] = 1e9
        second = local_history(ego, 1_900_000)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_bad_time_array_shapes_raise_value_error(self):
        for times in (0., [[0.], [1.]]):
            with self.assertRaises(ValueError):
                clearance_horizon(times, [[0., 0.], [1., 0.]], [], 1., 0., 0., 1.)


if __name__ == '__main__':
    unittest.main()
