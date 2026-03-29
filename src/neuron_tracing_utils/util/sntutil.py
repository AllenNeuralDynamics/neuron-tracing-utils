import numpy as np

from neuron_tracing_utils.util import imgutil, swcutil
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


def _vertex_sort_key(vertex):
    # Keep export order deterministic across SNT graph round-trips by
    # sorting siblings explicitly instead of depending on Java set order.
    return (
        int(vertex.type),
        float(vertex.x),
        float(vertex.y),
        float(vertex.z),
        float(vertex.radius),
    )


def graph_to_ndarray(graph):
    vertices = [v for v in graph.vertexSet()]
    if not vertices:
        return np.empty((0, 7), dtype=float)

    children = {vertex: [] for vertex in vertices}
    roots = []

    for vertex in vertices:
        if graph.inDegreeOf(vertex) == 0:
            roots.append(vertex)
        for edge in graph.outgoingEdgesOf(vertex):
            children[vertex].append(edge.getTarget())

    roots.sort(key=_vertex_sort_key)

    rows = []
    visited = set()
    next_id = 1
    stack = [(root, -1) for root in reversed(roots)]

    while stack:
        vertex, parent_id = stack.pop()
        if vertex in visited:
            continue
        visited.add(vertex)

        current_id = next_id
        next_id += 1
        rows.append(
            [
                current_id,
                int(vertex.type),
                float(vertex.x),
                float(vertex.y),
                float(vertex.z),
                float(vertex.radius),
                parent_id,
            ]
        )

        child_vertices = sorted(children[vertex], key=_vertex_sort_key)
        for child in reversed(child_vertices):
            stack.append((child, current_id))

    if len(visited) != len(vertices):
        raise ValueError("Graph traversal did not visit every SWC vertex")

    return np.array(rows, dtype=float)


def graph_to_swc(graph, out_path, header_lines=None, float_precision=6):
    swcutil.write_swc_rows(
        graph_to_ndarray(graph),
        out_path,
        header_lines=header_lines,
        float_precision=float_precision,
    )


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
