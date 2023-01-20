import argparse
import ast
import itertools
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import imglyb
import numpy as np
import scyjava
import tifffile

from refinery.transform import WorldToVoxel
from refinery.util.ioutil import ImgReaderFactory
from refinery.util.java import imglib2, imagej1
from refinery.util.java import n5
from refinery.util.java import snt

DEFAULT_Z_FUDGE = 0.8


class Cost(Enum):
    """Enum for cost functions a user can select"""

    reciprocal = "reciprocal"
    one_minus_erf = "one-minus-erf"


def fill_paths(pathlist, img, cost, threshold, calibration):
    reporting_interval = 1000  # ms
    thread = snt.FillerThread(
        img, calibration, threshold, reporting_interval, cost
    )
    thread.setSourcePaths(pathlist)
    thread.setStopAtThreshold(True)
    thread.run()
    return thread


def fill_path(path, img, cost, threshold, calibration):
    thread = snt.FillerThread(img, calibration, threshold, 1000, cost)
    thread.setSourcePaths([path])
    thread.setStopAtThreshold(True)
    thread.run()
    return thread


def myRange(start, end, step):
    i = start
    while i < end:
        yield i
        i += step
    yield end


def fill_tree(tree, img, cost, threshold, calibration):
    paths = []
    for b in snt.TreeAnalyzer(tree).getBranches():
        if b.size() > 500:
            vals = list(myRange(0, b.size()-1, 100))
            for i in range(0, len(vals)-1):
                paths.append(b.getSection(vals[i], vals[i+1]))
        else:
            paths.append(b)
    times = len(paths)
    t0 = time.time()
    with ThreadPoolExecutor(16) as executor:
        fillers = executor.map(
            fill_path,
            paths,
            itertools.repeat(img, times),
            itertools.repeat(cost, times),
            itertools.repeat(threshold, times),
            itertools.repeat(calibration, times),
        )
    print(f"Filled {tree.getLabel()} in {time.time() - t0}s")
    converter = snt.FillConverter(list(fillers))
    # Merge fills into a single stack.
    stack = converter.getFillerStack()
    return stack, converter


def get_cost(im, cost_str, z_fudge=DEFAULT_Z_FUDGE):
    Reciprocal = snt.Reciprocal
    OneMinusErf = snt.OneMinusErf

    params = {}

    if cost_str == Cost.reciprocal.value:
        mean = float(np.mean(im))
        maximum = float(np.max(im))
        cost = Reciprocal(mean, maximum)
        params['fill_cost_function'] = {
            'name': Cost.reciprocal.value,
            'args': {
                "min": mean,
                "max": maximum
            }
        }
    elif cost_str == Cost.one_minus_erf.value:
        mean = np.mean(im)
        maximum = np.max(im)
        std = np.std(im)
        cost = OneMinusErf(maximum, mean, std)
        # reduce z-score by a factor,
        # so we can numerically distinguish more
        # very bright voxels
        cost.setZFudge(z_fudge)
        params['fill_cost_function'] = {
            'name': Cost.one_minus_erf.value,
            "args": {
                "max": maximum,
                "average": mean,
                "standardDeviation": std,
                "zFudge": z_fudge
            }
        }
    else:
        raise ValueError(f"Invalid cost {cost_str}")

    return cost, params


def get_cost_global(cost_str, mean, std, cost_min, cost_max, z_fudge=DEFAULT_Z_FUDGE):
    Reciprocal = snt.Reciprocal
    OneMinusErf = snt.OneMinusErf
    params = {}

    if cost_str == Cost.reciprocal.value:
        minimum = max(0, mean + cost_min * std)
        maximum = mean + cost_max * std
        cost = Reciprocal(minimum, maximum)
        params['fill_cost_function'] = {
            'name': Cost.reciprocal.value,
            'args': {
                "min": minimum,
                "max": maximum
            }
        }
    elif cost_str == Cost.one_minus_erf.value:
        maximum = mean + cost_max * std
        cost = OneMinusErf(
            maximum,
            mean,
            std
        )
        # reduce z-score by a factor,
        # so we can numerically distinguish more
        # very bright voxels
        cost.setZFudge(z_fudge)
        params['fill_cost_function'] = {
            'name': Cost.one_minus_erf.value,
            "args": {
                "max": maximum,
                "average": mean,
                "standardDeviation": std,
                "zFudge": z_fudge
            }
        }
    else:
        raise ValueError(f"Invalid cost {cost_str}")

    return cost, params


def fill_swcs(
        swc_dir,
        im_dir,
        out_mask_dir,
        cost_str,
        threshold,
        cal,
        export_labels=True,
        export_gray=True,
        as_n5=False,
        use_global_stats=False,
        cost_min=-2,
        cost_max=10
):
    Tree = snt.Tree
    FillConverter = snt.FillConverter
    DiskCachedCellImgFactory = imglib2.DiskCachedCellImgFactory
    UnsignedShortType = imglib2.UnsignedShortType
    UnsignedByteType = imglib2.UnsignedByteType

    if use_global_stats:
        t0 = time.time()
        mean, std, minimum, maximum = calc_stats(image_dir=im_dir)
        logging.info(f"Computing stats took {time.time() - t0}s")
        logging.info(f"Global mean: {mean}, std: {std}, min: {minimum}, max: {maximum}")

        cost, params = get_cost_global(cost_str, mean, std, cost_min, cost_max)

    for root, dirs, files in os.walk(swc_dir):
        swcs = [os.path.join(root, f) for f in files if f.endswith(".swc")]
        if not swcs:
            continue

        img_name = os.path.basename(root)
        tiff = os.path.join(im_dir, img_name + ".tif")
        logging.info(f"Generating masks for {tiff}")
        try:
            im = tifffile.imread(tiff)
        except Exception as e:
            logging.error(e)
            continue

        if not use_global_stats:
            cost, params = get_cost(im, cost_str)

        logging.info(f"cost params: {params}")

        # Wrap ndarray as imglib2 Img, using shared memory
        # keep the reference in scope until the object is safe to be garbage collected
        img, ref_store = imglyb.as_cell_img(
            im, chunk_shape=im.shape, cache=100
        )

        filler_threads = []
        for f in swcs:
            tree = Tree(os.path.join(root, f))
            thread = fill_paths(tree.list(), img, cost, threshold, cal)
            filler_threads.append(thread)

        converter = FillConverter(filler_threads)

        if export_gray:
            mask = DiskCachedCellImgFactory(UnsignedShortType()).create(
                img.dimensionsAsLongArray()
            )
            converter.convert(img, mask)
            mask_name = img_name + "_Fill_Gray_Mask.n5"
            save_mask(mask, out_mask_dir, mask_name, as_n5)

        if export_labels:
            mask = DiskCachedCellImgFactory(UnsignedByteType()).create(
                img.dimensionsAsLongArray()
            )
            converter.convertLabels(mask)
            mask_name = img_name + "_Fill_Label_Mask.n5"
            save_mask(mask, out_mask_dir, mask_name, as_n5)

        # save cost params
        with open(os.path.join(out_mask_dir, img_name + "_fill_params.json"), 'w') as f:
            json.dump(params, f)


def fill_patches(
        patch_dir: Path,
        cost_str,
        threshold,
        cal,
        structure,
        export_gray=False,
        export_binary=True
):
    cost = snt.Reciprocal(11000, 40000)
    for neuron_dir in patch_dir.iterdir():
        if not neuron_dir.is_dir():
            continue
        swc_dir = neuron_dir / "swcs"
        image_dir = neuron_dir / "images"
        if not swc_dir.is_dir():
            raise Exception(f"Missing swc dir for {neuron_dir}")
        if not image_dir.is_dir():
            raise Exception(f"Missing image dir for {neuron_dir}")
        out_mask_dir = neuron_dir / "masks"
        out_mask_dir.mkdir(exist_ok=True)
        for struct_dir in swc_dir.iterdir():
            struct = struct_dir.name
            if structure != "all" and struct != structure:
                continue
            aligned_swcs = struct_dir / "patch-aligned"
            if not aligned_swcs.is_dir():
                raise Exception(f"Aligned swc dir does not exist: {aligned_swcs}")
            for swc in aligned_swcs.iterdir():
                if not swc.name.endswith(".swc"):
                    continue
                patch_name = swc.stem
                im_path = image_dir / struct / (patch_name + ".tif")
                if not im_path.is_file():
                    raise Exception(f"Missing image for {im_path}")
                img = ImgReaderFactory.create(im_path).load(str(im_path))
                tree = snt.Tree(str(swc))
                filler = fill_path(tree.list()[0], img, cost, threshold, cal)
                converter = snt.FillConverter([filler])
                struct_mask_dir = out_mask_dir / struct
                struct_mask_dir.mkdir(exist_ok=True)
                if export_gray:
                    mask = imglib2.ArrayImgFactory(imglib2.UnsignedShortType()).create(
                        img.dimensionsAsLongArray()
                    )
                    converter.convert(img, mask)
                    mask_name = patch_name + "_gray_mask.tif"
                    logging.info(f"Saving gray mask {mask_name}")
                    save_mask(mask, struct_mask_dir, mask_name, as_n5=False)
                if export_binary:
                    mask = imglib2.ArrayImgFactory(imglib2.BitType()).create(
                        img.dimensionsAsLongArray()
                    )
                    converter.convertBinary(mask)
                    mask_name = patch_name + "_label_mask.tif"
                    logging.info(f"Saving binary mask {mask_name}")
                    save_mask(mask, struct_mask_dir, mask_name, as_n5=False)


def save_mask(mask, mask_dir, mask_name, as_n5=False):
    Views = imglib2.Views
    IJ = imagej1.IJ
    ImageJFunctions = imglib2.ImageJFunctions

    mask_path = os.path.join(mask_dir, mask_name)
    if as_n5:
        save_n5(mask_path, mask)
    else:
        # ImageJ treats the 3rd dimension as channel instead of depth,
        # so add a dummy Z dimension to the end (XYCZ) and swap dimensions 2 and 3 (XYZC).
        # Opening this image in ImageJ will then show the correct axis type (XYZ)
        imp = ImageJFunctions.wrap(
            Views.permute(Views.addDimension(mask, 0, 0), 2, 3), ""
        )
        IJ.saveAsTiff(imp, mask_path)


def save_n5(filepath, img, dataset="volume", block_size=None):
    from java.lang import Runtime
    from java.util.concurrent import Executors

    if block_size is None:
        block_size = [64, 64, 64]
    n5Writer = n5.N5FSWriter(filepath)
    logging.info("Saving N5...")
    exec = Executors.newFixedThreadPool(
        Runtime.getRuntime().availableProcessors()
    )
    n5.N5Utils.save(
        img, n5Writer, dataset, block_size, n5.GzipCompression(6), exec
    )
    exec.shutdown()
    logging.info("Finished saving N5.")


def save_fill(fill_stack, path):
    import gzip

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for plane in fill_stack:
        for node in plane:
            if node is None:
                continue
            s = f"{node.x} {node.y} {node.z}"
            lines.append(s)

    fill_str_bytes = str.encode("\n".join(lines))
    with gzip.open(path, "wb") as f:
        f.write(fill_str_bytes)


def calc_stats(image_dir):
    global_min: float = float("inf")
    global_max: float = float("-inf")
    global_n: int = 0
    global_sum: float = 0
    counts = []
    variances = []
    means = []
    for imfile in Path(image_dir).iterdir():
        im = tifffile.imread(str(imfile))

        global_sum += im.sum(dtype=np.float64)
        global_n += im.size
        global_min = min(global_min, im.min())
        global_max = max(global_max, im.max())

        counts.append(im.size)
        variances.append(im.var(dtype=np.float64))
        means.append(im.mean(dtype=np.float64))

    global_mean: float = global_sum / global_n

    # get the overall standard deviation
    ssq: float = 0
    for i in range(len(variances)):
        ssq += (counts[i] - 1) * variances[i] + (counts[i] - 1) * (means[i] - global_mean) ** 2
    global_std: float = math.sqrt(ssq / (global_n - 1))

    return global_mean, global_std, global_min, global_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, help="directory of .swc files to fill",
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\patches-meanshift"
    )
    parser.add_argument(
        "--output", type=str, help="directory to output mask volumes",
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\endpoint-patches-meanshift\G-115_consensus\labels"
    )
    parser.add_argument(
        "--images",
        type=str,
        help="directory of images associated with the .swc files",
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\endpoint-patches-meanshift\G-115_consensus\images"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.06,
        help="distance threshold for fill algorithm",
    )
    parser.add_argument(
        "--transform", type=str, help='path to the "transform.txt" file',
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\transform.txt"
    )
    parser.add_argument("--voxel-size", type=str, help="voxel size of images")
    parser.add_argument(
        "--cost",
        type=str,
        choices=[cost.value for cost in Cost],
        default=Cost.reciprocal.value,
        help="cost function for the Dijkstra search",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["trees", "patches"],
        default="patches",
        help="task to run"
    )
    parser.add_argument(
        "--structure",
        choices=["soma", "endpoints", "branches", "all"],
        default="endpoints"
    )
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument(
        "--use-global-stats", action="store_true", default=False,
        help="use the statistics of all blocks for the cost function."
             "Otherwise, the statistics for each block are computed individually."
    )
    parser.add_argument(
        "--cost-min", type=float, default=-2, help="the value at which the cost function is maximized, "
                                                   "expressed in number of standard deviations from the mean intensity."
    )
    parser.add_argument(
        "--cost-max", type=float, default=10, help="the value at which the cost function is minimized, expressed in"
                                                   "number of standard deviations from the mean intensity."
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "args.json"), "w") as f:
        args.__dict__["script"] = parser.prog
        json.dump(args.__dict__, f, indent=2)

    logging.basicConfig(format="%(asctime)s %(message)s")
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    os.makedirs(args.output, exist_ok=True)

    calibration = imagej1.Calibration()
    if args.transform is not None:
        voxel_size = WorldToVoxel(args.transform).scale
    elif args.voxel_size is not None:
        voxel_size = ast.literal_eval(args.voxel_size)
    else:
        raise ValueError(
            "Either --transform or --voxel-size must be specified."
        )
    logging.info(f"Using voxel size: {voxel_size}")
    calibration.pixelWidth = voxel_size[0]
    calibration.pixelHeight = voxel_size[1]
    calibration.pixelDepth = voxel_size[2]

    if args.task == "trees":
        fill_swcs(
            args.input,
            args.images,
            args.output,
            args.cost,
            args.threshold,
            calibration,
            export_labels=True,
            export_gray=True,
            as_n5=args.n5,
            use_global_stats=args.use_global_stats,
            cost_min=args.cost_min,
            cost_max=args.cost_max
        )
    elif args.task == "patches":
        fill_patches(
            Path(args.input),
            args.cost,
            args.threshold,
            calibration,
            args.structure
        )
    else:
        raise Exception(f"Invalid task: {args.task}")


if __name__ == "__main__":
    main()
