import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import scyjava
from scipy.interpolate import splprep, splev

from neuron_tracing_utils.util import sntutil, swcutil
from neuron_tracing_utils.util.java import snt


def _point_to_ndarray(point):
    return np.array([point.getX(), point.getY(), point.getZ()], dtype=float)


def _dedupe_consecutive_points(points):
    if len(points) < 2:
        return points.copy()

    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.any(np.diff(points, axis=0) != 0, axis=1)
    return points[keep]


def _cumulative_lengths(points):
    if len(points) < 2:
        return np.array([], dtype=float), np.array([0.0], dtype=float)

    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    return seg_lengths, cumulative


def _point_at_arc_length(points, seg_lengths, cumulative, arc_length):
    if len(points) == 1:
        return points[0].copy()
    if arc_length <= 0:
        return points[0].copy()
    if arc_length >= cumulative[-1]:
        return points[-1].copy()

    seg_idx = int(np.searchsorted(cumulative, arc_length, side="right") - 1)
    seg_len = seg_lengths[seg_idx]
    if seg_len == 0:
        return points[seg_idx].copy()

    offset = arc_length - cumulative[seg_idx]
    t = offset / seg_len
    return points[seg_idx] + t * (points[seg_idx + 1] - points[seg_idx])


def _project_point_onto_polyline(points, query):
    query = np.asarray(query, dtype=float)
    if len(points) == 1:
        return 0.0, points[0].copy(), float(np.linalg.norm(points[0] - query))

    seg_lengths, cumulative = _cumulative_lengths(points)
    best_dist = float("inf")
    best_arc = 0.0
    best_point = points[0].copy()

    for idx, seg_len in enumerate(seg_lengths):
        start = points[idx]
        end = points[idx + 1]
        segment = end - start
        if seg_len == 0:
            proj = start
            t = 0.0
        else:
            t = np.dot(query - start, segment) / np.dot(segment, segment)
            t = float(np.clip(t, 0.0, 1.0))
            proj = start + t * segment

        dist = float(np.linalg.norm(query - proj))
        if dist < best_dist:
            best_dist = dist
            best_arc = cumulative[idx] + t * seg_len
            best_point = proj

    return best_arc, best_point, best_dist


def _slice_polyline(points, seg_lengths, cumulative, start_arc, end_arc):
    start_point = _point_at_arc_length(points, seg_lengths, cumulative, start_arc)
    end_point = _point_at_arc_length(points, seg_lengths, cumulative, end_arc)
    interior_mask = (cumulative > start_arc) & (cumulative < end_arc)
    interior_points = points[interior_mask]
    segment = np.vstack((start_point, interior_points, end_point))
    return _dedupe_consecutive_points(segment)


def _normalize_anchor_positions(anchor_positions, total_length, tol=1e-6):
    if not anchor_positions:
        return [], []

    clipped = [float(np.clip(pos, 0.0, total_length)) for pos in anchor_positions]
    order = sorted(range(len(clipped)), key=lambda idx: clipped[idx])

    unique_positions = []
    input_to_unique = [0] * len(clipped)
    for idx in order:
        position = clipped[idx]
        if not unique_positions or abs(position - unique_positions[-1]) > tol:
            unique_positions.append(position)
        input_to_unique[idx] = len(unique_positions) - 1

    return unique_positions, input_to_unique


def _resample_polyline_preserving_anchors(
    points, node_spacing, degree=1, anchor_positions=None
):
    points = _dedupe_consecutive_points(np.asarray(points, dtype=float))
    if len(points) == 0:
        return points, []

    anchor_positions = [] if anchor_positions is None else list(anchor_positions)
    if len(points) == 1:
        return points.copy(), [0] * len(anchor_positions)

    seg_lengths, cumulative = _cumulative_lengths(points)
    total_length = cumulative[-1]
    unique_anchors, input_to_unique = _normalize_anchor_positions(
        anchor_positions, total_length
    )

    boundaries = [0.0]
    boundaries.extend(
        anchor for anchor in unique_anchors if 0.0 < anchor < total_length
    )
    boundaries.append(total_length)

    output = []
    boundary_output_indices = {}
    for segment_idx, (start_arc, end_arc) in enumerate(zip(boundaries, boundaries[1:])):
        segment = _slice_polyline(
            points, seg_lengths, cumulative, start_arc, end_arc
        )
        resampled = _resample(segment, node_spacing, degree)
        if len(resampled) == 0:
            continue

        # Keep anchor coordinates exact, even when interpolation introduces
        # tiny floating-point drift.
        resampled[0] = segment[0]
        resampled[-1] = segment[-1]

        if segment_idx == 0:
            output = resampled.tolist()
            start_idx = 0
        else:
            start_idx = len(output) - 1
            output.extend(resampled[1:].tolist())

        boundary_output_indices[start_arc] = start_idx
        boundary_output_indices[end_arc] = len(output) - 1

    anchor_output_indices = [
        boundary_output_indices[unique_anchors[unique_idx]]
        for unique_idx in input_to_unique
    ]
    return np.asarray(output, dtype=float), anchor_output_indices


def resample_tree(tree, node_spacing, degree=1):
    """
    Performs in-place resampling of a Tree.
    Each path (3D parametric curve) in the Tree is approximated
    with a B-spline of degree N, which is then
    sampled at evenly spaced intervals along its domain.
    Args:
        tree (snt.Tree): the SNT Tree object to resample
        node_spacing (float): target spacing between consecutive
                              pairs of points
        degree (int): the degree of the fitted spline. Use degree=1
                      for linear interpolation. A higher degree will
                      introduce additional curvature which is not present
                      in the original representation, and may "overshoot"
                      relative to the underlying fluorescent signal.
    Returns:
        None
    """
    # Gather all paths into a Python list,
    # so we don't cause a ConcurrentModificationException
    # when adding and removing paths in the tree.
    paths = list(tree.list())
    for path in paths:
        start_joins = path.getStartJoins()
        children = list(path.getChildren())
        # Get a resampled version of the path
        resampled, child_node_indices = resample_path(
            path, node_spacing, degree, start_joins, children
        )
        # Add it to the tree.
        # Note we have not specified any connections yet,
        # so this is just a single un-branched segment.
        tree.add(resampled)
        # Get the parent of the input path, if any
        if start_joins is not None:
            # Get the point of connection on the parent
            start_joins_point = path.getStartJoinsPoint()
            # Now unlink the input path from parent path.
            # This clears the startJoins and startJoinsPoint fields.
            path.unsetStartJoin()
            # Replace with the resampled version
            resampled.setStartJoin(start_joins, start_joins_point)
        # Now swap the connections for any children of the input path
        for child in children:
            closest_idx = child_node_indices.get(child)
            if closest_idx is None:
                start_joins_point = child.getStartJoinsPoint()
                logging.warning(
                    "Falling back to nearest-node join remap for child %s",
                    child.getName(),
                )
                closest_idx = resampled.indexNearestTo(
                    start_joins_point.getX(),
                    start_joins_point.getY(),
                    start_joins_point.getZ(),
                    float("inf"),
                )
            closest_point = resampled.getNode(closest_idx)
            # Now unlink the child from the input path
            child.unsetStartJoin()
            # and link it to the resampled version
            child.setStartJoin(resampled, closest_point)
        # Remove the input path from the tree
        tree.remove(path)


def resample_swcs(indir, outdir, node_spacing):
    for root, dirs, files in os.walk(indir):
        swcs = [f for f in files if f.endswith(".json") or f.endswith(".swc")]
        for f in swcs:
            swc_path = os.path.join(root, f)
            if not os.path.isfile(swc_path):
                continue
            out_swc = os.path.join(outdir, os.path.relpath(swc_path, indir))
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)
            tree = snt.Tree(swc_path)
            # resample the tree in-place
            resample_tree(tree, node_spacing)
            tree.setRadii(1.0)
            tree.saveAsSWC(out_swc)


def resample_path(path, node_spacing, degree=1, start_joins=None, children=None):
    path_points = sntutil.path_to_ndarray(path).astype(float)
    original_type = path.getSWCType()
    children = [] if children is None else list(children)

    if start_joins is not None:
        # prepend the start joins point to the path points
        path_points = np.vstack((_point_to_ndarray(path.getStartJoinsPoint()), path_points))

    anchor_positions = []
    for child in children:
        start_point = _point_to_ndarray(child.getStartJoinsPoint())
        arc_length, _, distance = _project_point_onto_polyline(path_points, start_point)
        if distance > 1e-3:
            logging.debug(
                "Projected child join %.6f units onto parent path for %s",
                distance,
                child.getName(),
            )
        anchor_positions.append(arc_length)

    resampled, anchor_node_indices = _resample_polyline_preserving_anchors(
        path_points,
        node_spacing,
        degree,
        anchor_positions=anchor_positions,
    )

    if start_joins is not None and len(resampled) > 1:
        # The prepended join point is temporary context for resampling.
        # Keep the parent-child connection represented by setStartJoin()
        # instead of exporting it again as the first node on the child path.
        resampled = resampled[1:]
        anchor_node_indices = [
            max(node_idx - 1, 0) for node_idx in anchor_node_indices
        ]

    respath = path.createPath()
    # createPath() gives us a fresh Path geometry, so reapply the
    # original SWC type explicitly to preserve the compartment label.
    respath.setSWCType(original_type)
    for p in resampled:
        respath.addNode(snt.PointInImage(p[0], p[1], p[2]))

    child_node_map = {
        child: node_idx for child, node_idx in zip(children, anchor_node_indices)
    }
    return respath, child_node_map


def _resample(points, node_spacing, degree=1):
    points = _dedupe_consecutive_points(np.asarray(points, dtype=float))
    if len(points) < 2:
        return points.copy()
    # Determine number of query points and their parameters
    diff = np.diff(points, axis=0)
    ss = np.power(diff, 2).sum(axis=1)
    length = np.sqrt(ss).sum()
    if length == 0:
        return points[[0]].copy()

    quo, rem = divmod(length, node_spacing)
    samples = np.linspace(0, node_spacing * quo, int(quo + 1), endpoint=True)
    if rem != 0:
        samples = np.append(samples, samples[-1] + rem)
    # Queries along the spline must be in range [0, 1]
    query_points = np.clip(samples / max(samples), a_min=0.0, a_max=1.0)
    # Create spline points and evaluate at queries
    spline_degree = min(degree, len(points) - 1)
    tck, _ = splprep(points.T, k=spline_degree)
    result = np.array(splev(query_points, tck)).T
    result[0] = points[0]
    result[-1] = points[-1]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Resample .swc files to have even spacing between consecutive nodes"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="directory of .swc files to resample",
        default=r"C:\Users\cameron.arshadi\Downloads\non_uniform_jsons_for_Tiago\non_uniform_jsons_for_Tiago",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="directory to output resampled .swc files",
        default=r"C:\Users\cameron.arshadi\Downloads\non_uniform_jsons_for_Tiago_resampled_k3",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=10.0,
        help="target spacing between consecutive pairs of points,"
        " in spatial units given by the SWC. For example, "
        "if your SWCs are represented in micrometers,"
        "use micrometers. If they are in pixels, use pixels, etc.",
    )
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "args.json"), "w") as f:
        args.__dict__["script"] = parser.prog
        json.dump(args.__dict__, f, indent=2)

    if args.spacing <= 0:
        raise ValueError("--spacing must be > 0")

    logging.basicConfig(format="%(asctime)s %(message)s")
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    logging.info("Starting resample...")
    resample_swcs(args.input, args.output, args.spacing)
    logging.info("Finished resample.")


if __name__ == "__main__":
    main()
