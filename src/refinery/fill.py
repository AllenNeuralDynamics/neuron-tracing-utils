import argparse
import ast
import itertools
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import numpy as np
import scyjava

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

    if cost_str == Cost.reciprocal.value:
        mean = np.mean(im)
        maxi = np.max(im)
        cost = Reciprocal(mean, maxi)
    elif cost_str == Cost.one_minus_erf.value:
        mean = np.mean(im)
        maxi = np.max(im)
        stddev = np.std(im)
        cost = OneMinusErf(maxi, mean, stddev)
        # reduce z-score by a factor,
        # so we can numerically distinguish more
        # very bright voxels
        cost.setZFudge(z_fudge)
    else:
        raise ValueError(f"Invalid cost {cost_str}")

    return cost


def fill_swcs(
        swc_dir,
        im_dir,
        out_dir,
        cost_str,
        threshold,
        cal,
        export_gray=False,
        export_labels=True
):

    out_dir = Path(out_dir)

    out_mask_dir = out_dir / "masks"
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(swc_dir):
        swcs = [Path(os.path.join(root, f)) for f in files if f.endswith(".swc")]
        if not swcs:
            continue

        cost = snt.Reciprocal(11000, 40000)

        for f in swcs:

            print(f"filling {f}")

            name = f.stem
            im_path = Path(im_dir) / (name + ".tif")

            img = ImgReaderFactory.create(im_path).load(str(im_path))

            fill_dir = out_dir / "fills"
            fill_dir.mkdir(parents=True, exist_ok=True)

            fill_path = fill_dir / f"{name}_Fill.txt.gz"

            tree = snt.Tree(str(f))
            print(tree.getNodes())

            # coords = []
            # nodes = tree.getNodesAsSWCPoints()
            # for n in nodes:
            #     coords.append([n.x, n.y, n.z])
            # coords = np.array(coords, dtype=int)
            # bmin, bmax = coords.min(axis=0), coords.max(axis=0)
            # ival = imglib2.Intervals.createMinMax(bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2])
            # block = imglib2.Views.interval(view, ival)

            filler_stack, converter = fill_tree(tree, img, cost, threshold, cal)

            print("saving fill")
            save_fill(filler_stack, str(fill_path))

            as_n5 = False

            if export_gray:
                mask = imglib2.DiskCachedCellImgFactory(imglib2.UnsignedShortType()).create(
                    img.dimensionsAsLongArray()
                )
                converter.convert(img, mask)
                mask_name = name + "_Fill_Gray_Mask.n5"
                save_mask(mask, out_mask_dir, mask_name, as_n5)

            if export_labels:
                mask = imglib2.DiskCachedCellImgFactory(imglib2.UnsignedByteType()).create(
                    img.dimensionsAsLongArray()
                )
                converter.convertLabels(mask)
                mask_name = name + "_Fill_Label_Mask.n5"
                save_mask(mask, out_mask_dir, mask_name, as_n5)

            del filler_stack


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

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "args.json"), "w") as f:
        args.__dict__["script"] = parser.prog
        json.dump(args.__dict__, f, indent=2)

    logging.basicConfig(format="%(asctime)s %(message)s")
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    if not os.path.isdir(args.output):
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
