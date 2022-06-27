import argparse
import logging
import os

from . import imagej1
from . import snt
from . import imglib2
from . import n5
from . import java
from .transform import WorldToVoxel

import imglyb
import scyjava
import tifffile
import numpy as np


def fill_paths(pathlist, img, cost, threshold, calibration):
    reporting_interval = 1000  # ms
    thread = snt.FillerThread(img, calibration, threshold, reporting_interval, cost)
    thread.setSourcePaths(pathlist)
    thread.setStopAtThreshold(True)
    thread.run()
    return thread


def save_n5(filepath, img, dataset="volume", block_size=None):
    if block_size is None:
        block_size = [64, 64, 64]
    n5Writer = n5.N5FSWriter(filepath)
    logging.info("Saving N5...")
    exec = java.Executors.newFixedThreadPool(java.Runtime.getRuntime().availableProcessors())
    n5.N5Utils.save(img, n5Writer, dataset, block_size, n5.GzipCompression(6), exec)
    exec.shutdown()
    logging.info("Finished saving N5.")


def fill_swcs(swc_dir, im_dir, out_mask_dir, threshold, transform, export_labels=True, export_gray=True):
    Calibration = imagej1.Calibration
    Tree = snt.Tree
    Reciprocal = snt.Reciprocal
    FillConverter = snt.FillConverter
    DiskCachedCellImgFactory = imglib2.DiskCachedCellImgFactory
    UnsignedShortType = imglib2.UnsignedShortType
    UnsignedByteType = imglib2.UnsignedByteType

    spacing = transform.scale
    cal = Calibration()
    cal.pixelWidth = spacing[0]
    cal.pixelHeight = spacing[1]
    cal.pixelDepth = spacing[2]

    for root, dirs, files in os.walk(swc_dir):
        swcs = [os.path.join(root, f) for f in files if f.endswith('.swc')]
        if not swcs:
            continue

        img_name = os.path.basename(root)
        tiff = os.path.join(im_dir, img_name + ".tif")
        logging.info(f"Generating masks for {tiff}")
        im = tifffile.imread(tiff)

        # Compute statistics over the block to feed cost function
        mean = np.mean(im)
        maxi = np.max(im)
        cost = Reciprocal(mean, maxi)

        # Wrap ndarray as imglib2 Img, using shared memory
        # keep the reference in scope until the object is safe to be garbage collected
        img, ref_store = imglyb.as_cell_img(im, chunk_shape=(64, 64, 64), cache=100)

        filler_threads = []
        for f in swcs:
            tree = Tree(os.path.join(root, f))
            thread = fill_paths(tree.list(), img, cost, threshold, cal)
            filler_threads.append(thread)

        converter = FillConverter(filler_threads)

        if export_gray:
            gray_mask = DiskCachedCellImgFactory(UnsignedShortType()).create(
                img.dimensionsAsLongArray())
            converter.convert(img, gray_mask)
            save_n5(os.path.join(out_mask_dir, img_name + '_Fill_Gray_Mask.n5'), gray_mask)

        if export_labels:
            label_mask = DiskCachedCellImgFactory(UnsignedByteType()).create(
                img.dimensionsAsLongArray())
            converter.convertLabels(label_mask)
            save_n5(os.path.join(out_mask_dir, img_name + '_Fill_Label_Mask.n5'), label_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='directory of .swc files to fill')
    parser.add_argument('--output', type=str,  help='directory to output mask volumes')
    parser.add_argument('--images', type=str, help='directory of images associated with the .swc files')
    parser.add_argument("--threshold", type=float, default=0.05, help="distance threshold for fill algorithm")
    parser.add_argument('--transform', type=str, help='path to the \"transform.txt\" file')
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    um2vx = WorldToVoxel(args.transform)

    fill_swcs(args.input, args.images, args.output, args.threshold, um2vx)


if __name__ == "__main__":
    main()
