import numpy as np

from neuron_tracing_utils.util import imgutil
from neuron_tracing_utils.util.chunkutil import chunk_center, minmax_to_interval
from neuron_tracing_utils.util.java import imglib2
from neuron_tracing_utils.util.java import snt


def tree_to_ndarray(tree):
    arr = []
    nodes = tree.getNodesAsSWCPoints()
    for n in nodes:
        row = [n.id, n.type, n.x, n.y, n.z, n.radius, n.parent]
        arr.append(row)
    return np.array(arr, dtype=float)


def ndarray_to_graph(swc_arr):
    swc_points = []
    for line in swc_arr:
        swc_points.append(
            snt.SWCPoint(
                int(line[0]),
                int(line[1]),
                float(line[2]),
                float(line[3]),
                float(line[4]),
                float(line[5]),
                int(line[6]),
            )
        )
    return snt.DirectedWeightedGraph(swc_points, True)


def path_to_ndarray(path):
    nodes = []
    for i in range(path.size()):
        n = path.getNode(i)
        nodes.append([n.x, n.y, n.z])
    return np.array(nodes)


def swcpoint_to_sphere(img, swcPoint, radius):
    point = imglib2.Point(3)
    point.setPosition(int(swcPoint.x), 0)
    point.setPosition(int(swcPoint.y), 1)
    point.setPosition(int(swcPoint.z), 2)
    return imglib2.HyperSphere(img, point, radius)


def swcpoint_to_block(img, swcpoint, side_lengths):
    return imglib2.Views.interval(
        img,
        minmax_to_interval(
            *chunk_center(
                [swcpoint.x, swcpoint.y, swcpoint.z],
                side_lengths
            )
        ),
    )


def point_neighborhood(img, swcpoint, radius=1, pad=None, shape="sphere"):
    if pad is None:
        pad = [1, 1, 1]
    if shape == "sphere":
        region = swcpoint_to_sphere(img, swcpoint, radius)
    elif shape == "block":
        region = swcpoint_to_block(img, swcpoint, pad)
    else:
        raise ValueError(f"Invalid shape {shape}")
    return imgutil.local_intensities(region)
