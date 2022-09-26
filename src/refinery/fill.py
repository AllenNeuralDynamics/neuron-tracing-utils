import argparse
import ast
import logging
import os
from enum import Enum

from refinery.util.java import n5
from refinery.util.java import snt
from refinery.util.java import imglib2, imagej1
from refinery.transform import WorldToVoxel

import imglyb
import scyjava
import tifffile
import numpy as np


class Cost(Enum):
    """Enum for cost functions a user can select"""

    reciprocal = "reciprocal"
    one_minus_erf = "one-minus-erf"


def fill_paths(pathlist, img, cost, threshold, calibration):
    reporting_interval = 1000  # ms
    thread = snt.FillerThread(img, calibration, threshold, reporting_interval, cost)
    thread.setSourcePaths(pathlist)
    thread.setStopAtThreshold(True)
    thread.run()
    return thread


def save_n5(filepath, img, dataset="volume", block_size=None):
    from java.lang import Runtime
    from java.util.concurrent import Executors

    if block_size is None:
        block_size = [64, 64, 64]
    n5Writer = n5.N5FSWriter(filepath)
    logging.info("Saving N5...")
    exec = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors())
    n5.N5Utils.save(img, n5Writer, dataset, block_size, n5.GzipCompression(6), exec)
    exec.shutdown()
    logging.info("Finished saving N5.")


def get_cost(im, cost_str):
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
    else:
        raise ValueError(f"Invalid cost {cost_str}")

    return cost


def fill_swcs(swc_dir, im_dir, out_mask_dir, cost_str, threshold, cal, export_labels=True, export_gray=True, as_n5=False):
    Tree = snt.Tree
    FillConverter = snt.FillConverter
    DiskCachedCellImgFactory = imglib2.DiskCachedCellImgFactory
    UnsignedShortType = imglib2.UnsignedShortType
    UnsignedByteType = imglib2.UnsignedByteType

    for root, dirs, files in os.walk(swc_dir):
        swcs = [os.path.join(root, f) for f in files if f.endswith('.swc')]
        if not swcs:
            continue

        img_name = os.path.basename(root)
        tiff = os.path.join(im_dir, img_name + ".tif")
        logging.info(f"Generating masks for {tiff}")
        im = tifffile.imread(tiff)

        cost = get_cost(im, cost_str)

        # Wrap ndarray as imglib2 Img, using util memory
        # keep the reference in scope until the object is safe to be garbage collected
        img, ref_store = imglyb.as_cell_img(im, chunk_shape=(64, 64, 64), cache=100)

        filler_threads = []
        for f in swcs:
            tree = Tree(os.path.join(root, f))
            thread = fill_paths(tree.list(), img, cost, threshold, cal)
            filler_threads.append(thread)

        converter = FillConverter(filler_threads)

        if export_gray:
            mask = DiskCachedCellImgFactory(UnsignedShortType()).create(
                img.dimensionsAsLongArray())
            converter.convert(img, mask)
            mask_name = img_name + '_Fill_Gray_Mask.n5'
            save_mask(mask, out_mask_dir, mask_name, as_n5)

        if export_labels:
            mask = DiskCachedCellImgFactory(UnsignedByteType()).create(
                img.dimensionsAsLongArray())
            converter.convertLabels(mask)
            mask_name = img_name + '_Fill_Label_Mask.n5'
            save_mask(mask, out_mask_dir, mask_name, as_n5)


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
        imp = ImageJFunctions.wrap(Views.permute(Views.addDimension(mask, 0, 0), 2, 3), "")
        IJ.saveAsTiff(imp, mask_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='directory of .swc files to fill')
    parser.add_argument('--output', type=str,  help='directory to output mask volumes')
    parser.add_argument('--images', type=str, help='directory of images associated with the .swc files')
    parser.add_argument("--threshold", type=float, default=0.05, help="distance threshold for fill algorithm")
    parser.add_argument('--transform', type=str, help='path to the \"transform.txt\" file')
    parser.add_argument("--voxel-size", type=str, help="voxel size of images")
    parser.add_argument(
        "--cost",
        type=str,
        choices=[Cost.reciprocal.value, Cost.one_minus_erf.value],
        default=Cost.reciprocal.value,
        help="cost function for the Dijkstra search"
    )
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--n5", default=False, action="store_true", help="save masks as n5. Otherwise, save as Tiff.")

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s')
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
        raise ValueError("Either --transform or --voxel-size must be specified.")
    calibration.pixelWidth = voxel_size[0]
    calibration.pixelHeight = voxel_size[1]
    calibration.pixelDepth = voxel_size[2]

    fill_swcs(args.input, args.images, args.output, args.cost, args.threshold, calibration, export_labels=True, export_gray=True,
              as_n5=args.n5)


if __name__ == "__main__":
    main()
