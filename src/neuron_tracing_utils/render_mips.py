import argparse
import json
import logging
import os
from pathlib import Path

from neuron_tracing_utils.util.java import snt

import scyjava
import tifffile
import numpy as np
from skimage import exposure
from skimage.io import imsave
from skimage.draw import line
from skimage.color import gray2rgb, label2rgb


def write_orig_mips(swc_dir, im_dir, out_mip_dir, vmin=0.0, vmax=20000.0):
    for root, dirs, files in os.walk(swc_dir):
        swcs = [os.path.join(root, f) for f in files if f.endswith(".swc")]
        if not swcs:
            continue
        img_name = os.path.basename(root)
        img = tifffile.imread(
            os.path.join(im_dir, img_name + ".tif")
        )
        img_rescale = exposure.rescale_intensity(img, in_range=(vmin, vmax))
        mip_rgb = gray2rgb(np.max(img_rescale, axis=0))
        for f in swcs:
            swc = os.path.join(root, f)
            graph = snt.Tree(swc).getGraph()
            for e in graph.edgeSet():
                source = e.getSource()
                target = e.getTarget()
                # draw line
                rr, cc = line(
                    int(round(source.getY())),
                    int(round(source.getX())),
                    int(round(target.getY())),
                    int(round(target.getX())),
                )
                try:
                    mip_rgb[rr, cc, 0] = 255
                except IndexError:
                    continue
        out_tiff = os.path.join(out_mip_dir, img_name + "_astar_MIP.png")
        Path(out_tiff).parent.mkdir(exist_ok=True, parents=True)
        imsave(out_tiff, mip_rgb)


def write_label_mips(im_dir, out_mip_dir):
    labels = [os.path.join(im_dir, f) for f in os.listdir(im_dir) if f.endswith("_Fill_Label_Mask.tif")]
    for l in labels:
        img = tifffile.imread(l)
        mip_rgb = label2rgb(np.max(img, axis=0))

        out_tiff = os.path.join(out_mip_dir, Path(l).name.replace(".tif", "_MIP.png"))
        Path(out_tiff).parent.mkdir(exist_ok=True, parents=True)
        imsave(out_tiff, mip_rgb)


def write_gray_mips(im_dir, out_mip_dir, vmin, vmax):
    labels = [os.path.join(im_dir, f) for f in os.listdir(im_dir) if f.endswith("_Fill_Gray_Mask.tif")]
    for l in labels:
        img = tifffile.imread(l)
        img = exposure.rescale_intensity(img, in_range=(vmin, vmax))
        mip_rgb = gray2rgb(np.max(img, axis=0))

        out_tiff = os.path.join(out_mip_dir, Path(l).name.replace(".tif", "_MIP.png"))
        Path(out_tiff).parent.mkdir(exist_ok=True, parents=True)
        imsave(out_tiff, mip_rgb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="directory to output MIPs")
    parser.add_argument(
        "--images",
        type=str,
        help="directory of images associated with the .swc files",
    )
    parser.add_argument(
        "--swcs", type=str, help="directory of .swc files to render"
    )
    parser.add_argument(
        "--masks",
        type=str,
        help="directory of masks"
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="minimum intensity of the desired display range",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=65535.0,
        help="maximum intensity of the desired display range",
    )
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, 'args.json'), 'w') as f:
        args.__dict__['script'] = parser.prog
        json.dump(args.__dict__, f, indent=2)

    logging.basicConfig(format="%(asctime)s %(message)s")
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    write_orig_mips(args.swcs, args.images, os.path.join(args.output, "orig"), args.vmin, args.vmax)
    write_label_mips(args.masks, os.path.join(args.output, "labels"))
    write_gray_mips(args.masks, os.path.join(args.output, "gray"), args.vmin, args.vmax)


if __name__ == "__main__":
    main()
