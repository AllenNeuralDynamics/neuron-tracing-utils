import argparse
import logging
import os
from pathlib import Path

import numpy as np
import scyjava
from scipy.interpolate import splprep, splev

from refinery.util import sntutil, swcutil
from refinery.util.java import snt


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
        # Get a resampled version of the path
        resampled = resample_path(path, node_spacing, degree)
        # Add it to the tree.
        # Note we have not specified any connections yet,
        # so this is just a single un-branched segment.
        tree.add(resampled)
        # Get the parent of the input path, if any
        start_joins = path.getStartJoins()
        if start_joins is not None:
            # Get the point of connection on the parent
            start_joins_point = path.getStartJoinsPoint()
            # Now unlink the input path from parent path.
            # This clears the startJoins and startJoinsPoint fields.
            path.unsetStartJoin()
            # Replace with the resampled version
            resampled.setStartJoin(start_joins, start_joins_point)
        # Now swap the connections for any children of the input path
        # Wrap in a Python list to avoid a ConcurrentModificationException
        children = list(path.getChildren())
        for child in children:
            # Get the point of connection on the input path
            start_joins_point = child.getStartJoinsPoint()
            # Find the closest point on the resampled version, since
            # the node coordinates are different
            closest_idx = resampled.indexNearestTo(
                start_joins_point.getX(),
                start_joins_point.getY(),
                start_joins_point.getZ(),
                float("inf")  # within this distance
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
        swcs = [f for f in files if f.endswith('.swc')]
        for f in swcs:
            swc_path = os.path.join(root, f)
            if not os.path.isfile(swc_path):
                continue
            out_swc = os.path.join(outdir, os.path.relpath(swc_path, indir))
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)
            arr = swcutil.swc_to_ndarray(swc_path, add_offset=True)
            tree = sntutil.ndarray_to_graph(arr).getTree()
            # resample the tree in-place
            resample_tree(tree, node_spacing)
            tree.setRadii(1.0)
            tree.saveAsSWC(out_swc)


def resample_path(path, node_spacing, degree=1):
    path_length = path.getLength()
    if path_length <= node_spacing:
        return path
    path_points = sntutil.path_to_ndarray(path)
    resampled = _resample(path_points, node_spacing, degree)
    respath = path.createPath()
    for p in resampled:
        respath.addNode(snt.PointInImage(p[0], p[1], p[2]))
    return respath


def _resample(points, node_spacing, degree=1):
    # remove duplicate nodes
    _, ind = np.unique(points, axis=0, return_index=True)
    # Maintain input order
    points = points[np.sort(ind)]
    # Determine number of query points and their parameters
    diff = np.diff(points, axis=0, prepend=points[-1].reshape((1, -1)))
    ss = np.power(diff, 2).sum(axis=1)
    length = np.sqrt(ss).sum()
    quo, rem = divmod(length, node_spacing)
    samples = np.linspace(0, node_spacing * quo, int(quo+1))
    if rem != 0:
        samples = np.append(samples, samples[-1] + rem)
    # Queries along the spline must be in range [0, 1]
    query_points = np.clip(samples / max(samples), a_min=0.0, a_max=1.0)
    # Create spline points and evaluate at queries
    tck, _ = splprep(points.T, k=degree)
    return np.array(splev(query_points, tck)).T


def main():
    parser = argparse.ArgumentParser(description="Resample .swc files to have even spacing between consecutive nodes")
    parser.add_argument('--input', type=str, help='directory of .swc files to resample')
    parser.add_argument('--output', type=str, help='directory to output resampled .swc files')
    parser.add_argument(
        "--spacing",
        type=float,
        default=5.0,
        help="target spacing between consecutive pairs of points,"
             " in spatial units given by the SWC. For example, "
             "if your SWCs are represented in micrometers,"
             "use micrometers. If they are in pixels, use pixels, etc."
    )
    parser.add_argument('--log-level', type=int, default=logging.INFO)

    args = parser.parse_args()

    if args.spacing <= 0:
        raise ValueError("--spacing must be > 0")

    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    logging.info("Starting resample...")
    resample_swcs(args.input, args.output, args.spacing)
    logging.info("Finished resample.")


if __name__ == "__main__":
    main()
