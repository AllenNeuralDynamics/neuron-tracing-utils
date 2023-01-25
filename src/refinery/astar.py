import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from enum import Enum
from pathlib import Path
import time

import scyjava
from jpype import JImplements, JOverride, JArray, JLong
from tqdm import tqdm
import numpy as np

from refinery.util import sntutil, ioutil, imgutil
from refinery.util.ioutil import ImgReaderFactory
from refinery.util.java import snt
from refinery.util.java import imglib2, imagej1
from refinery.transform import WorldToVoxel


class Cost(Enum):
    reciprocal = "reciprocal"
    relative_difference = "relative_difference"


@JImplements("sc.fiji.snt.tracing.cost.Cost", deferred=True)
class RelativeDifference:

    def __init__(self, value_at_start):
        self.value_at_start = value_at_start

    @JOverride
    def costMovingTo(self, new_value):
        if new_value == self.value_at_start:
            return 1.0
        return 1.0 + (abs(new_value - self.value_at_start) / (new_value + self.value_at_start))

    @JOverride
    def minStepCost(self):
        return 1.0


def _astar_edge(edge, img, cost_str, calibration, timeout):
    # declare Java classes we will use
    Euclidean = snt.Euclidean
    Reciprocal = snt.Reciprocal
    BiSearch = snt.BiSearch
    SNT = snt.SNT
    ImgUtils = snt.ImgUtils
    Views = imglib2.Views
    DoubleType = imglib2.DoubleType
    ComputeMinMax = imglib2.ComputeMinMax
    RectangleShape = imglib2.RectangleShape

    source = edge.getSource()
    target = edge.getTarget()
    # these need to be voxel coordinates
    sx = int(round(source.x))
    sy = int(round(source.y))
    sz = int(round(source.z))
    tx = int(round(target.x))
    ty = int(round(target.y))
    tz = int(round(target.z))

    if cost_str == Cost.reciprocal.value:
        # compute min-max of the subvolume where the start and goal nodes
        # are origin and corner, respectively, plus padding in each dimension
        pad_pixels = 20
        subvolume = ImgUtils.subVolume(img, sx, sy, sz, tx, ty, tz, pad_pixels)
        iterable = Views.iterable(subvolume)
        minmax = ComputeMinMax(iterable, DoubleType(), DoubleType())
        minmax.process()
        # reciprocal of intensity * distance is our cost for moving to a neighboring node
        cost = Reciprocal(
            minmax.getMin().getRealDouble(), minmax.getMax().getRealDouble()
        )
    elif cost_str == Cost.relative_difference.value:
        pos = JArray(JLong, 1)(3)
        pos[0] = sx
        pos[1] = sy
        pos[2] = sz
        shape = RectangleShape(1, False)
        nhood_ra = shape.neighborhoodsRandomAccessible(img).randomAccess()
        nhood = nhood_ra.setPositionAndGet(pos)
        start_val = float("-inf")
        for val in nhood:
            start_val = max(start_val, val.get())
        cost = RelativeDifference(start_val)
    else:
        raise Exception(f"Unsupported Cost {cost_str}")

    # A* mode
    # Use heuristic = Dijkstra() instead to default to Dijkstra's algorithm (i.e., h(n) = 0)
    heuristic = Euclidean(calibration)

    search = BiSearch(
        img, calibration, sx, sy, sz, tx, ty, tz, timeout,  -1, SNT.SearchImageType.MAP, cost, heuristic,
    )

    search.run()

    # note the Path result is in world coordinates
    return search.getResult()


def astar_swc(
        in_swc: str,
        out_swc: str,
        img,
        calibration,
        cost_str: str,
        key: str = None,
        timeout: int = 600,  # s
        threads: int = 1
):
    print(f"processing {in_swc}")

    if isinstance(img, (str, Path)):
        reader = ImgReaderFactory.create(img)
        img = imgutil.get_hyperslice(reader.load(img, key=key))

    graph = snt.Tree(in_swc).getGraph()

    spacing = np.array(
        [
            calibration.pixelWidth,
            calibration.pixelHeight,
            calibration.pixelDepth,
        ]
    )

    edges = []
    dfs = graph.getDepthFirstIterator()
    while dfs.hasNext():
        n = dfs.next()
        in_edges = graph.incomingEdgesOf(n)
        if in_edges.isEmpty():
            continue
        edges.append(in_edges.iterator().next())

    # FIXME multithreading is slower than serial
    # paths = []
    # with ThreadPoolExecutor(threads) as executor:
    #     futures = [executor.submit(_astar_edge, edge, img, cost_str, calibration, timeout) for edge in edges]
    #     for fut in tqdm(futures):
    #         try:
    #             paths.append(fut.result())
    #         except Exception as e:
    #             logging.error(f"Exception during astar: {e}")

    for edge in tqdm(edges):
        path = _astar_edge(edge, img, cost_str, calibration, timeout)
        if path is None:
            logging.error(
                "Search failed for {}: points {} and {}".format(
                    in_swc, str(edge.getSource()), str(edge.getTarget())
                )
            )
            continue

        path_arr = sntutil.path_to_ndarray(path)
        # convert back to voxel coords
        path_arr /= spacing
        assert len(path_arr) > 1

        graph.removeEdge(edge)
        tmp = graph.addVertex(path_arr[0][0], path_arr[0][1], path_arr[0][2])
        graph.addEdge(edge.getSource(), tmp)
        prev = tmp
        for i in range(1, len(path_arr)):
            tmp = graph.addVertex(
                path_arr[i][0], path_arr[i][1], path_arr[i][2]
            )
            graph.addEdge(prev, tmp)
            prev = tmp
        graph.addEdge(tmp, edge.getTarget())

    tree = graph.getTree()
    # Set a non-zero radius.
    # Some programs (JWS) fail to import .swc files with radii == 0
    tree.setSWCType("axon")
    tree.setRadii(1.0)
    tree.saveAsSWC(out_swc)


def astar_batch(
        in_swc_dir,
        out_swc_dir,
        im_dir,
        calibration,
        cost,
        key=None,
        threads=1,
):
    im_fmt = ioutil.get_file_format(im_dir)

    in_swcs = []
    out_swcs = []
    im_paths = []

    for root, dirs, files in os.walk(in_swc_dir):
        swcs = [f for f in files if f.endswith(".swc")]
        if not swcs:
            continue

        im_path = os.path.join(im_dir, os.path.basename(root) + im_fmt)
        if not os.path.isfile(im_path):
            raise FileNotFoundError(f"{im_path} does not exist")

        for f in swcs:
            in_swc = os.path.join(root, f)
            in_swcs.append(in_swc)
            logging.info(f"Running A-star on {in_swc}")

            out_swc = os.path.join(
                out_swc_dir, os.path.relpath(in_swc, in_swc_dir)
            )
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)
            out_swcs.append(out_swc)

            im_paths.append(im_path)

    times = len(in_swcs)

    t0 = time.time()
    with ThreadPoolExecutor(threads) as executor:
        futures = []
        for i in range(len(in_swcs)):
            futures.append(executor.submit(astar_swc, in_swcs[i], out_swcs[i], im_paths[i], calibration, cost, key))
        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                logging.error(e)
    t1 = time.time()
    logging.info(f"processed {times} swcs in {t1 - t0}s")


def astar_swcs(
        in_swc_dir,
        out_swc_dir,
        im_path,
        calibration,
        cost,
        key=None,
        threads=1,
):
    reader = ImgReaderFactory.create(im_path)
    img = imgutil.get_hyperslice(reader.load(im_path, key=key))

    in_swcs = []
    out_swcs = []

    for root, dirs, files in os.walk(in_swc_dir):
        swcs = [f for f in files if f.endswith(".swc")]
        if not swcs:
            continue
        for f in swcs:
            in_swc = os.path.join(root, f)
            in_swcs.append(in_swc)
            logging.info(f"Running A-star on {in_swc}")

            out_swc = os.path.join(
                out_swc_dir, os.path.relpath(in_swc, in_swc_dir)
            )
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)
            out_swcs.append(out_swc)

    times = len(in_swcs)

    t0 = time.time()
    for i in range(len(in_swcs)):
        astar_swc(in_swcs[i], out_swcs[i], img, calibration, cost, key, threads=threads)
    t1 = time.time()
    logging.info(f"processed {times} swcs in {t1 - t0}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, help="directory of .swc files to refine"
    )
    parser.add_argument(
        "--output", type=str, help="directory to output refined .swc files"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="image or directory of images associated with the .swc files",
    )
    parser.add_argument(
        "--transform", type=str, help='path to the "transform.txt" file'
    )
    parser.add_argument(
        "--voxel-size",
        type=str,
        help="voxel size for images, as a string in XYZ order, e.g., '0.3,0.3,1.0'",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="key for the N5/Zarr dataset"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="number of threads to use for processing",
    )
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument(
        "--cost",
        type=str,
        choices=[cost.value for cost in Cost],
        default=Cost.reciprocal.value
    )

    args = parser.parse_args()

    scyjava.start_jvm()

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "args.json"), "w") as f:
        args.__dict__["script"] = parser.prog
        json.dump(args.__dict__, f, indent=2)

    logging.basicConfig(format="%(asctime)s %(message)s")
    logging.getLogger().setLevel(args.log_level)

    calibration = imagej1.Calibration()
    if args.transform is not None:
        voxel_size = WorldToVoxel(args.transform).scale
    elif args.voxel_size is not None:
        voxel_size = ast.literal_eval(args.voxel_size)
    else:
        raise ValueError(
            "Either --transform or --voxel-size must be specified."
        )
    logging.info(f"Using voxel size {voxel_size}")
    calibration.pixelWidth = voxel_size[0]
    calibration.pixelHeight = voxel_size[1]
    calibration.pixelDepth = voxel_size[2]

    logging.info("Starting A-star...")
    t0 = time.time()
    if os.path.isdir(args.image):
        astar_batch(
            args.input,
            args.output,
            args.image,
            calibration,
            args.cost,
            args.dataset,
            args.threads,
        )
    else:
        astar_swcs(
            args.input,
            args.output,
            args.image,
            calibration,
            args.cost,
            args.dataset,
            args.threads,
        )
    logging.info(f"Finished A-star. Took {time.time() - t0}s")


if __name__ == "__main__":
    main()
