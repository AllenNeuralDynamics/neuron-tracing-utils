import argparse
import ast
import logging
import os
from pathlib import Path

from . import imagej1
from . import snt
from . import imglib2

from . import sntutil
from .transform import WorldToVoxel

import scyjava
import numpy as np


def astar_graph(graph, img, calibration):
    # declare Java classes we will use
    Euclidean = snt.Euclidean
    Reciprocal = snt.Reciprocal
    BiSearch = snt.BiSearch
    SNT = snt.SNT
    ImgUtils = snt.ImgUtils
    Views = imglib2.Views
    DoubleType = imglib2.DoubleType
    ComputeMinMax = imglib2.ComputeMinMax

    # just to avoid a concurrent modification exception
    # when iterating over the edge set
    edges = [e for e in graph.edgeSet()]

    spacing = np.array([calibration.pixelWidth, calibration.pixelHeight, calibration.pixelDepth])

    # A* mode
    # Use heuristic = Dijkstra() instead to default to Dijkstra's algorithm (i.e., h(n) = 0)
    heuristic = Euclidean(calibration)
    for e in edges:
        source = e.getSource()
        target = e.getTarget()

        # these need to be voxel coordinates
        sx = int(round(source.x))
        sy = int(round(source.y))
        sz = int(round(source.z))
        tx = int(round(target.x))
        ty = int(round(target.y))
        tz = int(round(target.z))

        # compute min-max of the subvolume where the start and goal nodes
        # are origin and corner, respectively, plus padding in each dimension
        pad_pixels = 30
        subvolume = ImgUtils.subVolume(img, sx, sy, sz, tx, ty, tz, pad_pixels)
        iterable = Views.iterable(subvolume)
        minmax = ComputeMinMax(iterable, DoubleType(), DoubleType())
        minmax.process()

        # reciprocal of intensity * distance is our cost for moving to a neighboring node
        cost = Reciprocal(minmax.getMin().getRealDouble(), minmax.getMax().getRealDouble())

        search = BiSearch(
            img, calibration,
            sx, sy, sz,
            tx, ty, tz,
            30, 0,  # timeout (s), debug mode reporting interval (ms)
            SNT.SearchImageType.MAP,
            cost, heuristic
        )

        search.run()

        # note the Path result is in world coordinates
        path = search.getResult()
        if path is None:
            logging.warning("Search failed for points {} and {}, skipping edge".format(str(source), str(target)))
            continue

        path_arr = sntutil.path_to_ndarray(path)
        # convert back to voxel coords
        path_arr /= spacing

        assert len(path_arr) > 1

        graph.removeEdge(source, target)
        tmp = graph.addVertex(
            path_arr[0][0],
            path_arr[0][1],
            path_arr[0][2]
        )
        graph.addEdge(source, tmp)
        prev = tmp
        for i in range(1, len(path_arr)):
            tmp = graph.addVertex(
                path_arr[i][0],
                path_arr[i][1],
                path_arr[i][2]
            )
            graph.addEdge(prev, tmp)
            prev = tmp
        graph.addEdge(tmp, target)


def astar_swcs(in_swc_dir, out_swc_dir, imdir, calibration, swc_type="axon", swc_radius=1.0):
    IJLoader = imglib2.IJLoader
    Tree = snt.Tree

    loader = IJLoader()

    for root, dirs, files in os.walk(in_swc_dir):
        for f in files:
            if not f.endswith(".swc"):
                continue
            img_name = os.path.basename(root) + ".tif"
            tiff = os.path.join(imdir, img_name)
            img = loader.get(tiff)

            in_swc = os.path.join(root, f)
            logging.info(f"Running A-star on {in_swc}")

            out_swc = os.path.join(out_swc_dir, os.path.relpath(in_swc, in_swc_dir))
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)

            graph = Tree(in_swc).getGraph()
            astar_graph(graph, img, calibration)
            tree = graph.getTree()
            # Set a non-zero radius.
            # Some programs (JWS) fail to import .swc files with radii == 0
            tree.setSWCType(swc_type)
            tree.setRadii(swc_radius)
            tree.saveAsSWC(out_swc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='directory of .swc files to refine')
    parser.add_argument('--output', type=str,  help='directory to output refined .swc files')
    parser.add_argument('--images', type=str, help='directory of images associated with the .swc files')
    parser.add_argument('--transform', type=str, help='path to the \"transform.txt\" file')
    parser.add_argument('--voxel-size', type=str, help="voxel size for images")
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    calibration = imagej1.Calibration()
    if args.transform is not None:
        voxel_size = WorldToVoxel(args.transform).scale
    elif args.voxel_size is not None:
        voxel_size = ast.literal_eval(args.voxel_size)
    else:
        raise ValueError("Either --transform or --voxel-size must be specified.")
    calibration.pixelWidth = voxel_size[0]
    calibration.pixelHeight = voxel_size[1]
    calibration.pixelDepth = voxel_size[2]

    logging.info("Starting A-star...")
    astar_swcs(args.input, args.output, args.images, calibration)
    logging.info("Finished A-star.")


if __name__ == "__main__":
    main()
