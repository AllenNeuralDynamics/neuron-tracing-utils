import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import scyjava
from PIL import Image
from skimage.color import gray2rgb
from skimage.draw import circle_perimeter
from skimage.exposure import rescale_intensity
from skimage.io import imsave
from skimage.util import montage
from tifffile import tifffile

from refinery.util.java import snt


def write_circle_mips(swc_dir, im_dir, out_mip_dir, vmin=0.0, vmax=20000.0):
    mips = []
    for root, dirs, files in os.walk(swc_dir):
        swcs = [os.path.join(root, f) for f in files if f.endswith(".swc")]
        if not swcs:
            continue
        for f in swcs:
            im_name = Path(f).stem
            im_path = os.path.join(im_dir, im_name + ".tif")
            img = tifffile.imread(im_path)
            img_rescale = rescale_intensity(
                img, in_range=(vmin, vmax)
            )
            mip_rgb = gray2rgb(np.max(img_rescale, axis=0))
            graph = snt.Tree(f).getGraph()
            for v in graph.vertexSet():
                # draw circle
                rr, cc = circle_perimeter(
                    int(round(v.getY())), int(round(v.getX())), 4
                )
                try:
                    mip_rgb[rr, cc, 1] = 255
                except IndexError as e:
                    print(e)
                    continue
            mips.append(mip_rgb)
        mt = montage(mips, channel_axis=3)
        out = os.path.join(out_mip_dir, "montage_mip.png")
        Path(out).parent.mkdir(exist_ok=True, parents=True)
        imsave(out, mt)


def write_label_mips(label_dir, im_dir, out_mip_dir, vmin=0.0, vmax=20000.0, alpha=0.5):
    red_multiplier = np.array([1, 0, 0], dtype=np.uint8)
    name_pattern = re.compile(r"patch-(\d+)")
    mips = []
    for root, dirs, files in os.walk(label_dir):
        mask_paths = [Path(os.path.join(root, f)) for f in files]
        for mask_path in mask_paths:
            m = re.search(name_pattern, str(mask_path))
            if not m:
                continue
            name = m.group(0)
            raw = gray2rgb(
                rescale_intensity(
                    tifffile.imread(im_dir / (name + ".tif")).max(axis=0),
                    in_range=(vmin, vmax),
                    out_range=np.uint8
                )
            )
            labels = gray2rgb(tifffile.imread(str(mask_path)).max(axis=0)).astype(np.uint8)
            labels = labels * red_multiplier
            raw_im = Image.fromarray(raw).convert("RGBA")
            label_im = Image.fromarray(labels).convert("RGBA")
            blended = Image.blend(raw_im, label_im, alpha)
            mips.append(np.array(blended))
    mt = montage(mips, channel_axis=3)
    out = out_mip_dir / "montage_mip.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    imsave(out, mt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\patches-meanshift-dog"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\endpoint-patches-meanshift\G-115_consensus",
        help="directory to output MIPs",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=12000.0,
        help="minimum intensity of the desired display range",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=14000.0,
        help="maximum intensity of the desired display range",
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

    indir = Path(args.input)
    for neuron_dir in indir.iterdir():
        if not neuron_dir.is_dir():
            continue
        mip_dir = neuron_dir / "mips"
        mip_dir.mkdir(exist_ok=True)
        swc_dir = neuron_dir / "swcs"
        image_dir = neuron_dir / "images"
        mask_dir = neuron_dir / "masks"
        for struct_dir in swc_dir.iterdir():
            aligned_swcs = struct_dir / "patch-aligned"
            struct_image_dir = image_dir / struct_dir.name
            struct_mip_dir = mip_dir / "points" / struct_dir.name
            struct_mip_dir.mkdir(parents=True, exist_ok=True)
            write_circle_mips(
                aligned_swcs,
                struct_image_dir,
                struct_mip_dir,
                args.vmin,
                args.vmax,
            )
            struct_mip_dir = mip_dir / "labels" / struct_dir.name
            struct_mip_dir.mkdir(parents=True, exist_ok=True)
            struct_mask_dir = mask_dir / struct_dir.name
            write_label_mips(
                struct_mask_dir,
                struct_image_dir,
                struct_mip_dir,
                args.vmin,
                args.vmax,
            )


if __name__ == "__main__":
    main()
