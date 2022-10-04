import argparse
import ast
import json
import logging
import re
import os
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import itertools
import time

from refinery.util.java import n5
from refinery.util.java import snt
from refinery.util.java import imglib2, imagej1
from refinery.transform import WorldToVoxel

import imglyb
import scyjava
import tifffile
import numpy as np

import jpype.imports

import zarr


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
    thread = snt.FillerThread(
            img, calibration, threshold, 1000, cost
    )
    thread.setSourcePaths([path])
    thread.setStopAtThreshold(True)
    thread.run()
    return thread


def myRange(start,end,step):
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
            itertools.repeat(calibration, times)
        )
    print(f"Filled {tree.getLabel()} in {time.time() - t0}s")
    converter = snt.FillConverter(list(fillers))
    # Merge fills into a single stack.
    stack = converter.getFillerStack()
    return stack


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
    out_mask_dir,
    cost_str,
    threshold,
    cal,
):
    Tree = snt.Tree

    reader = n5.N5AmazonS3Reader(
        n5.AmazonS3ClientBuilder.defaultClient(), 
        "janelia-mouselight-imagery", 
        "carveouts/2018-08-01/fluorescence-near-consensus.n5"
    )
    img = n5.N5Utils.openWithBoundedSoftRefCache(reader, "volume-rechunked", 2000)
    view = imglib2.Views.hyperSlice(img, 3, 0)
    print(view.dimensionsAsLongArray())
    
    name_pattern = re.compile(r"G-\d+")

    for root, dirs, files in os.walk(swc_dir):
        swcs = [os.path.join(root, f) for f in files if f.endswith(".swc")]
        if not swcs:
            continue

        cost = snt.Reciprocal(11000, 40000)

        for f in swcs:
            if not "consensus" in f:
                continue
                
            print(f"filling {f}")
            
            m = name_pattern.search(f)
            if not m:
                print(f"missing tag for {f}")
                continue
            tag = m.group(0)
            
            fill_path = os.path.join(out_mask_dir, f"{tag}_Fill.txt.gz")
            if os.path.isfile(fill_path):
                continue
            
            tree = Tree(f)
            
            # coords = []
            # nodes = tree.getNodesAsSWCPoints()
            # for n in nodes:
            #     coords.append([n.x, n.y, n.z])
            # coords = np.array(coords, dtype=int)
            # bmin, bmax = coords.min(axis=0), coords.max(axis=0)
            # ival = imglib2.Intervals.createMinMax(bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2])
            # block = imglib2.Views.interval(view, ival)
    
            filler_stack = fill_tree(tree, view, cost, threshold, cal)
        
            print("saving fill")
            save_fill(filler_stack, fill_path)
            
            del filler_stack
        
        
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
    with gzip.open(path, 'wb') as f:
        f.write(fill_str_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, help="directory of .swc files to fill"
    )
    parser.add_argument(
        "--output", type=str, help="directory to output mask volumes"
    )
    parser.add_argument(
        "--images",
        type=str,
        help="directory of images associated with the .swc files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="distance threshold for fill algorithm",
    )
    parser.add_argument(
        "--transform", type=str, help='path to the "transform.txt" file'
    )
    parser.add_argument("--voxel-size", type=str, help="voxel size of images")
    parser.add_argument(
        "--cost",
        type=str,
        choices=[cost.value for cost in Cost],
        default=Cost.reciprocal.value,
        help="cost function for the Dijkstra search",
    )
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument(
        "--n5",
        default=False,
        action="store_true",
        help="save masks as n5. Otherwise, save as Tiff.",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, 'args.json'), 'w') as f:
        args.__dict__['script'] = parser.prog
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

    fill_swcs(
        args.input,
        args.images,
        args.output,
        args.cost,
        args.threshold,
        calibration,
    )


if __name__ == "__main__":
    main()
