import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import scyjava

from neuron_tracing_utils.resample import (
    _dedupe_consecutive_points,
    _resample_polyline_preserving_anchors,
    resample_tree,
)
from neuron_tracing_utils.util import sntutil
from neuron_tracing_utils.util.java import snt


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


class ResampleTreeIntegrationTest(unittest.TestCase):
    DATA_DIR = Path(__file__).parent / "data" / "exaSPIM"
    CASES = (
        ("dense", "N001-685221-dendrite-dense.swc", 10.0),
        ("sparse", "N001-685221-dendrite-sparse.swc", 1.0),
    )

    @classmethod
    def setUpClass(cls):
        scyjava.start_jvm()

    def _load_tree(self, filename):
        return snt.Tree(str(self.DATA_DIR / filename))

    def _graph_rows(self, tree):
        return sntutil.graph_to_ndarray(tree.getGraph())

    def _count_branch_points(self, rows):
        node_ids = rows[:, 0].astype(int)
        parent_ids = rows[:, 6].astype(int)
        child_counts = Counter(parent_ids[parent_ids != -1])
        return sum(1 for node_id in node_ids if child_counts.get(node_id, 0) > 1)

    def _count_terminal_nodes(self, rows):
        node_ids = rows[:, 0].astype(int)
        parent_ids = rows[:, 6].astype(int)
        child_counts = Counter(parent_ids[parent_ids != -1])
        return sum(1 for node_id in node_ids if child_counts.get(node_id, 0) == 0)

    def _branch_count(self, tree):
        return snt.TreeAnalyzer(tree).getNBranches()

    def _anchor_positions(self, tree):
        anchors = []
        for path in list(tree.list()):
            if path.getStartJoins() is None:
                continue
            join = path.getStartJoinsPoint()
            anchors.append(
                (
                    round(join.getX(), 6),
                    round(join.getY(), 6),
                    round(join.getZ(), 6),
                )
            )
        return sorted(anchors)

    def _leaf_endpoint_positions(self, tree):
        endpoints = []
        for path in list(tree.list()):
            if list(path.getChildren()):
                continue
            node = path.getNode(path.size() - 1)
            endpoints.append(
                (round(node.x, 6), round(node.y, 6), round(node.z, 6))
            )
        return sorted(endpoints)

    def _root_paths(self, tree):
        roots = []
        for path in list(tree.list()):
            if path.getStartJoins() is not None:
                continue
            node = path.getNode(0)
            roots.append(
                (
                    round(node.x, 6),
                    round(node.y, 6),
                    round(node.z, 6),
                    path.size(),
                )
            )
        return sorted(roots)

    def _count_consecutive_duplicates(self, tree):
        duplicate_count = 0
        for path in list(tree.list()):
            points = sntutil.path_to_ndarray(path)
            if len(points) < 2:
                continue
            duplicate_count += int(
                np.all(np.diff(points, axis=0) == 0, axis=1).sum()
            )
        return duplicate_count

    def _snapshot_tree(self, tree):
        rows = self._graph_rows(tree)
        root_count = int((rows[:, 6].astype(int) == -1).sum())
        return {
            "graph_nodes": len(rows),
            "path_count": len(list(tree.list())),
            "branch_count": self._branch_count(tree),
            "branch_point_count": self._count_branch_points(rows),
            "terminal_count": self._count_terminal_nodes(rows),
            "root_count": root_count,
            "anchor_positions": self._anchor_positions(tree),
            "leaf_endpoints": self._leaf_endpoint_positions(tree),
            "root_paths": self._root_paths(tree),
            "consecutive_duplicates": self._count_consecutive_duplicates(tree),
        }

    def test_resampling_changes_graph_node_counts_in_expected_direction(self):
        for label, filename, spacing in self.CASES:
            with self.subTest(case=label, filename=filename, spacing=spacing):
                tree = self._load_tree(filename)
                before = self._snapshot_tree(tree)

                resample_tree(tree, spacing)

                after = self._snapshot_tree(tree)

                if label == "dense":
                    self.assertLess(after["graph_nodes"], before["graph_nodes"])
                else:
                    self.assertGreater(after["graph_nodes"], before["graph_nodes"])

    def test_resampling_preserves_real_world_tree_topology_and_anchors(self):
        for label, filename, spacing in self.CASES:
            with self.subTest(case=label, filename=filename, spacing=spacing):
                tree = self._load_tree(filename)
                before = self._snapshot_tree(tree)

                self.assertGreater(before["branch_count"], 1)
                self.assertGreater(before["terminal_count"], 1)
                self.assertGreater(len(before["anchor_positions"]), 0)

                resample_tree(tree, spacing)

                after = self._snapshot_tree(tree)

                self.assertEqual(after["path_count"], before["path_count"])
                self.assertEqual(after["branch_count"], before["branch_count"])
                self.assertEqual(
                    after["branch_point_count"], before["branch_point_count"]
                )
                self.assertEqual(after["terminal_count"], before["terminal_count"])
                self.assertEqual(before["root_count"], 1)
                self.assertEqual(after["root_count"], 1)
                self.assertEqual(
                    after["anchor_positions"], before["anchor_positions"]
                )
                self.assertEqual(after["leaf_endpoints"], before["leaf_endpoints"])
                self.assertEqual(after["root_paths"], before["root_paths"])
                self.assertEqual(after["consecutive_duplicates"], 0)


if __name__ == "__main__":
    unittest.main()
