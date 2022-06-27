import argparse
import logging
import os
from pathlib import Path

from javahelpers import snt
import scyjava
import tifffile
import numpy as np
from skimage import exposure
from skimage.io import imsave
from skimage.draw import line
from skimage.color import gray2rgb


def render_swcs(swc_dir, im_dir, out_mip_dir, vmin=12000, vmax=15000):
    for root, dirs, files in os.walk(swc_dir):
        swcs = [os.path.join(root, f) for f in files if f.endswith('.swc')]
        if not swcs:
            continue
        img_name = os.path.basename(root)
        tiff = os.path.join(im_dir, img_name + ".tif")
        img = tifffile.imread(tiff)
        img_rescale = exposure.rescale_intensity(img, in_range=(vmin, vmax))
        img_mip = np.max(img_rescale, axis=0)
        mip_rgb = gray2rgb(img_mip)
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
                    int(round(target.getX()))
                )
                try:
                    mip_rgb[rr, cc, 0] = 255
                except IndexError:
                    continue
        out_tiff = os.path.join(out_mip_dir, img_name + "_astar_MIP.png")
        Path(out_tiff).parent.mkdir(exist_ok=True, parents=True)
        imsave(out_tiff, mip_rgb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='directory of .swc files to render')
    parser.add_argument('--output', type=str,  help='directory to output MIPs')
    parser.add_argument('--images', type=str, help='directory of images associated with the .swc files')
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    render_swcs(args.input, args.images, args.output)


if __name__ == "__main__":
    main()
