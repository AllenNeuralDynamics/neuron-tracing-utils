import json
import os
import shutil
from pathlib import Path

import scyjava

from neuron_tracing_utils.util.java import n5
from neuron_tracing_utils.util.java import imglib2, imagej1
from neuron_tracing_utils.util import chunkutil
from neuron_tracing_utils.util import swcutil
from neuron_tracing_utils.transform import WorldToVoxel

import numpy as np


def crop_img_from_swcs(
    img,
    swc_folder,
    output_image_dir,
    transform,
    add_offset=True,
    output_aligned_swcs=True,
):
    Views = imglib2.Views
    ImageJFunctions = imglib2.ImageJFunctions
    Intervals = imglib2.Intervals
    Calibration = imagej1.Calibration

    arrs = []
    names = []
    for f in os.listdir(swc_folder):
        if os.path.isfile(os.path.join(swc_folder, f)) and f.endswith(".swc"):
            swc_arr = swcutil.swc_to_ndarray(
                os.path.join(swc_folder, f), add_offset
            )
            names.append(swcutil.path_to_name(f))
            arrs.append(swc_arr)

    world_points = np.vstack(arrs)[:, 2:5]
    voxel_points = transform.forward(world_points)

    bbmin, bbmax = chunkutil.bbox(voxel_points)
    bbmin = bbmin.astype(int)
    bbmax = bbmax.astype(int)

    chunk_metadata = {}
    chunk_metadata["chunk_origin"] = bbmin.tolist()
    chunk_metadata["chunk_shape"] = (bbmax - bbmin + 1).tolist()
    chunk_metadata["voxel_spacing"] = transform.scale.tolist()

    interval = Intervals.createMinMax(
        bbmin[0], bbmin[1], bbmin[2], bbmax[0], bbmax[1], bbmax[2]
    )
    block = Views.interval(img, interval)

    imp = ImageJFunctions.wrap(
        Views.permute(Views.addDimension(block, 0, 0), 2, 3), "Img"
    )

    calibration = Calibration()
    calibration.setUnit("um")
    calibration.pixelWidth = transform.scale[0]
    calibration.pixelHeight = transform.scale[1]
    calibration.pixelDepth = transform.scale[2]

    imp.setCalibration(calibration)
    imp.resetDisplayRange()

    imagej1.IJ.saveAsTiff(imp, os.path.join(output_image_dir, "input.tif"))

    if output_aligned_swcs:
        output_swc_folder = os.path.join(swc_folder, "block_aligned_swcs")
        if not os.path.isdir(output_swc_folder):
            os.mkdir(output_swc_folder)
        bbmin_world, _ = chunkutil.bbox(world_points)
        for i, swc_arr in enumerate(arrs):
            swc_arr[:, 2:5] -= bbmin_world
            swcutil.ndarray_to_swc(
                swc_arr,
                os.path.join(output_swc_folder, names[i] + "_aligned.swc"),
            )

    with open(os.path.join(output_image_dir, "block_metadata.json"), "w") as f:
        json.dump(chunk_metadata, f)

    print("Done")


def consolidate_blocks(block_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for d in Path(block_dir).iterdir():
        if not d.name.startswith("block"):
            continue
        block = d / "input.tif"
        if not block.is_file():
            continue
        shutil.copyfile(block, Path(out_dir) / (d.name + ".tif"))


if __name__ == "__main__":
    scyjava.start_jvm()
    swc_folder = r"C:\Users\cameron.arshadi\Desktop\repos\20210812-AG-training-data\swcs\validation\astar-trees"
    transform_path = r"C:\Users\cameron.arshadi\Desktop\repos\20210812-AG-training-data\transform.txt"
    um2vx = WorldToVoxel(transform_path)
    # Open carve-out workspace
    n5_workspace = r"C:\Users\cameron.arshadi\Desktop\repos\20210812-AG-training-data\20210812-AG_full_arbor.n5"
    n5 = n5.N5FSReader(n5_workspace)
    print("datasets: ", n5.list("/"))
    lazyimg = n5.N5Utils.open(n5, "volume")
    print(
        "Volume dimensions: ", imglib2.Intervals.dimensionsAsLongArray(lazyimg)
    )
    lazyimg_ch0 = imglib2.Views.hyperSlice(
        lazyimg, lazyimg.numDimensions() - 1, 0
    )
    output_image_dir = r"C:\Users\cameron.arshadi\Desktop\repos\20210812-AG-training-data\blocks\validation"
    if not os.path.isdir(output_image_dir):
        os.mkdir(output_image_dir)
    crop_img_from_swcs(
        lazyimg_ch0, swc_folder, output_image_dir, um2vx, True, True
    )
