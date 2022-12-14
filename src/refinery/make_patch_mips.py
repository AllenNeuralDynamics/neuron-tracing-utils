import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import scyjava
from PIL import Image
from skimage.exposure import rescale_intensity
from skimage.color import gray2rgb
from skimage.draw import circle_perimeter
from skimage.io import imsave
from tifffile import tifffile

from refinery.util.java import snt


def write_circle_mips(swc_dir, im_dir, out_mip_dir, vmin=0.0, vmax=20000.0):
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
            out = os.path.join(out_mip_dir, im_name + "_mip.png")
            Path(out).parent.mkdir(exist_ok=True, parents=True)
            imsave(out, mip_rgb)


def write_label_mips(label_dir, im_dir, out_mip_dir, vmin=0.0, vmax=20000.0, alpha=0.5):
    red_multiplier = np.array([1, 0, 0], dtype=np.uint8)
    name_pattern = re.compile(r"patch-(\d+)")
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
            labels = gray2rgb(tifffile.imread(str(mask_path)).max(axis=0) * 255).astype(np.uint8)
            labels = labels * red_multiplier
            raw_im = Image.fromarray(raw).convert("RGBA")
            label_im = Image.fromarray(labels).convert("RGBA")
            blended = Image.blend(raw_im, label_im, 0.5)
            out = out_mip_dir / (name + ".png")
            out.parent.mkdir(parents=True, exist_ok=True)
            blended.save(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\endpoint-patches-meanshift\G-115_consensus",
        help="directory to output MIPs",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\endpoint-patches-meanshift\G-115_consensus\images",
        help="directory of images associated with the .swc files",
    )
    parser.add_argument(
        "--swcs",
        type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\endpoint-patches-meanshift\G-115_consensus\swcs",
        help="directory of .swc files to render",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=r"C:\Users\cameron.arshadi\Desktop\2018-08-01\endpoint-patches-meanshift\G-115_consensus\labels\masks"
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

    write_circle_mips(
        args.swcs,
        args.images,
        os.path.join(args.output, "orig"),
        args.vmin,
        args.vmax,
    )

    write_label_mips(
        Path(args.labels),
        Path(args.images),
        Path(args.output) / "orig",
        args.vmin,
        args.vmax,
    )


if __name__ == "__main__":
    main()
