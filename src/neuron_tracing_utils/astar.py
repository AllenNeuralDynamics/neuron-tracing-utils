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
from jpype import JImplements, JOverride, JArray, JLong, JException
from tqdm import tqdm
import numpy as np
import jpype.imports

from neuron_tracing_utils.util import sntutil, ioutil, imgutil
from neuron_tracing_utils.util.ioutil import ImgReaderFactory, is_n5_zarr
from neuron_tracing_utils.util.java import snt
from neuron_tracing_utils.util.java import imglib2, imagej1
from neuron_tracing_utils.transform import WorldToVoxel


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


@JImplements("java.util.concurrent.Callable", deferred=True)
class _AstarCallable(object):
    def __init__(self, edge, img, cost_str, voxel_size, timeout):
        self.edge = edge
        self.img = img
        self.cost_str = cost_str
        self.voxel_size = voxel_size
        self.timeout = timeout

    @JOverride
    def call(self):
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
        Calibration = imagej1.Calibration

        source = self.edge.getSource()
        target = self.edge.getTarget()
        # these need to be voxel coordinates
        sx = int(round(source.x))
        sy = int(round(source.y))
        sz = int(round(source.z))
        tx = int(round(target.x))
        ty = int(round(target.y))
        tz = int(round(target.z))

        if self.cost_str == Cost.reciprocal.value:
            # compute min-max of the subvolume where the start and goal nodes
            # are origin and corner, respectively, plus padding in each dimension
            pad_pixels = 20
            subvolume = ImgUtils.subVolume(self.img, sx, sy, sz, tx, ty, tz, pad_pixels)
            iterable = Views.iterable(subvolume)
            minmax = ComputeMinMax(iterable, DoubleType(), DoubleType())
            minmax.process()
            # reciprocal of intensity * distance is our cost for moving to a neighboring node
            cost = Reciprocal(
                minmax.getMin().getRealDouble(), minmax.getMax().getRealDouble()
            )
        elif self.cost_str == Cost.relative_difference.value:
            pos = JArray(JLong, 1)(3)

            pos[0] = sx
            pos[1] = sy
            pos[2] = sz
            start_val = _get_max_neighbor(pos, self.img)

            pos[0] = tx
            pos[1] = ty
            pos[2] = tz
            end_val = _get_max_neighbor(pos, self.img)

            target_val = (start_val + end_val) / 2

            cost = RelativeDifference(target_val)
        else:
            raise Exception(f"Unsupported Cost {self.cost_str}")


        calibration = Calibration()
        calibration.pixelWidth = self.voxel_size[0]
        calibration.pixelHeight = self.voxel_size[1]
        calibration.pixelDepth = self.voxel_size[2]

        # A* mode
        # Use heuristic = Dijkstra() instead to default to Dijkstra's algorithm (i.e., h(n) = 0)
        heuristic = Euclidean(calibration)

        search = BiSearch(
            self.img, calibration, sx, sy, sz, tx, ty, tz, self.timeout,  -1, SNT.SearchImageType.MAP, cost, heuristic,
        )

        search.run()

        # note the Path result is in world coordinates
        return search.getResult()


def _get_max_neighbor(pos, img, radius=1) -> float:
    """
    Args:
        pos (JArray): the 3-element position array
        img (RandomAccessibleInterval): the source image
        radius (int): shape radius

    Returns:
        the maximum value in the neighborhood
    """
    DiamondShape = imglib2.DiamondShape

    nhood_ra = DiamondShape(radius).neighborhoodsRandomAccessible(img).randomAccess()
    nhood = nhood_ra.setPositionAndGet(pos)
    maximum = img.randomAccess().setPositionAndGet(pos).get()
    for val in nhood:
        try:
            maximum = max(maximum, float(val.get()))
        except JException as e:
            continue
    return maximum


def astar_swc(
        in_swc: str,
        out_swc: str,
        img,
        voxel_size,
        cost_str: str,
        key: str = None,
        timeout: int = -1,  # s
        threads: int = 1
):
    from java.util.concurrent import Executors

    print(f"processing {in_swc}")

    if isinstance(img, (str, Path)):
        reader = ImgReaderFactory.create(img)
        img = imgutil.get_hyperslice(reader.load(img, key=key))

    graph = snt.Tree(in_swc).getGraph()

    voxel_size = np.array(voxel_size)

    edges = []
    dfs = graph.getDepthFirstIterator()
    while dfs.hasNext():
        n = dfs.next()
        in_edges = graph.incomingEdgesOf(n)
        if in_edges.isEmpty():
            continue
        edges.append(in_edges.iterator().next())

    paths = []
    for edge in tqdm(edges):
        paths.append(_AstarCallable(edge, img, cost_str, voxel_size, timeout).call())

    for edge, path in zip(edges, paths):
        if path is None:
            logging.error(
                "Search failed for {}: points {} and {}".format(
                    in_swc, str(edge.getSource()), str(edge.getTarget())
                )
            )
            continue

        path_arr = sntutil.path_to_ndarray(path)
        # convert back to voxel coords
        path_arr /= voxel_size
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
        voxel_size,
        cost,
        key=None,
        threads=1,
):
    im_fmt = ioutil.get_file_format(im_dir)
    c = 0
    t0 = time.time()
    for root, dirs, files in os.walk(in_swc_dir):
        swcs = [f for f in files if f.endswith(".swc")]
        if not swcs:
            continue

        im_path = os.path.join(im_dir, os.path.basename(root) + im_fmt)
        if not os.path.isfile(im_path):
            raise FileNotFoundError(f"{im_path} does not exist")

        for f in swcs:
            in_swc = os.path.join(root, f)
            logging.info(f"Running A-star on {in_swc}")

            out_swc = os.path.join(
                out_swc_dir, os.path.relpath(in_swc, in_swc_dir)
            )
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)

            astar_swc(in_swc, out_swc, im_path, voxel_size, cost, key, -1, threads)

            c += 1
    t1 = time.time()
    logging.info(f"processed {c} swcs in {t1 - t0}s")


def astar_swcs(
        in_swc_dir,
        out_swc_dir,
        im_path,
        voxel_size,
        cost,
        key=None,
        threads=1,
        filter=None,
        scales=None
):
    reader = ImgReaderFactory.create(im_path)
    img = imgutil.get_hyperslice(reader.load(im_path, key=key))
    if filter is not None:
        img = imgutil.filter(img, scales, voxel_size, filter, lazy=True, threads=threads)

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
        astar_swc(in_swcs[i], out_swcs[i], img, voxel_size, cost, key, threads=threads)
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
    parser.add_argument(
        "--filter",
        type=str,
        choices=["frangi", "tubeness"],
        default=None,
        help="Filter to lazily apply to accessed regions of the volume"
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        help="A sequence of scales to integrate the filter response over. "
             "A scale corresponds to the standard deviation of the "
             "Gaussian kernel used to smooth the image prior to computing the Hessian."
             "Each scale should roughly correspond to the radius of structures you want "
             "to enhance, in physical units.",
        default=None
    )

    args = parser.parse_args()

    scyjava.start_jvm()

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "args.json"), "w") as f:
        args.__dict__["script"] = parser.prog
        json.dump(args.__dict__, f, indent=2)

    logging.basicConfig(format="%(asctime)s %(message)s")
    logging.getLogger().setLevel(args.log_level)

    if args.transform is not None:
        voxel_size = WorldToVoxel(args.transform).scale
    elif args.voxel_size is not None:
        voxel_size = ast.literal_eval(args.voxel_size)
    else:
        raise ValueError(
            "Either --transform or --voxel-size must be specified."
        )
    logging.info(f"Using voxel size {voxel_size}")

    logging.info("Starting A-star...")
    t0 = time.time()
    if os.path.isdir(args.image) and not is_n5_zarr(args.image):
        astar_batch(
            args.input,
            args.output,
            args.image,
            voxel_size,
            args.cost,
            args.dataset,
            args.threads,
        )
    else:
        astar_swcs(
            args.input,
            args.output,
            args.image,
            voxel_size,
            args.cost,
            args.dataset,
            args.threads,
            args.filter,
            args.scales,
        )
    logging.info(f"Finished A-star. Took {time.time() - t0}s")


if __name__ == "__main__":
    main()
