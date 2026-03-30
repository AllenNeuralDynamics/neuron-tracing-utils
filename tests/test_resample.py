import unittest

import numpy as np

from neuron_tracing_utils.resample import (
    _dedupe_consecutive_points,
    _resample_polyline_preserving_anchors,
)


class ResampleHelpersTest(unittest.TestCase):
    def test_preserves_distinct_anchor_nodes(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                [40.0, 0.0, 0.0],
            ]
        )

        resampled, anchor_indices = _resample_polyline_preserving_anchors(
            points,
            node_spacing=20.0,
            degree=1,
            anchor_positions=[11.0, 19.0],
        )

        np.testing.assert_allclose(resampled[anchor_indices[0]], [11.0, 0.0, 0.0])
        np.testing.assert_allclose(resampled[anchor_indices[1]], [19.0, 0.0, 0.0])
        self.assertNotEqual(anchor_indices[0], anchor_indices[1])

    def test_duplicate_anchors_share_the_same_output_node(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
            ]
        )

        resampled, anchor_indices = _resample_polyline_preserving_anchors(
            points,
            node_spacing=20.0,
            degree=1,
            anchor_positions=[11.0, 11.0, 19.0],
        )

        self.assertEqual(anchor_indices[0], anchor_indices[1])
        self.assertNotEqual(anchor_indices[0], anchor_indices[2])
        np.testing.assert_allclose(resampled[anchor_indices[0]], [11.0, 0.0, 0.0])

    def test_keeps_non_consecutive_repeated_points(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        deduped = _dedupe_consecutive_points(points)

        self.assertEqual(len(deduped), 3)
        np.testing.assert_allclose(deduped, points)


if __name__ == "__main__":
    unittest.main()
