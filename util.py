import json
import os

import scyjava

from javahelpers import imagej1, n5, imglib2
import chunkutil
import swcutil
from transform import WorldToVoxel

import numpy as np
from scipy.interpolate import splprep, splev


def resample(points, node_spacing, degree=1):
    # _, ind = np.unique(points, axis=0, return_index=True)
    # Maintain input order
    # points = points[np.sort(ind)]
    # Determine number of query points and their parameters
    diff = np.diff(points, axis=0, prepend=points[-1].reshape((1, -1)))
    ss = np.power(diff, 2).sum(axis=1)
    length = np.sqrt(ss).sum()
    quo, rem = divmod(length, node_spacing)
    sampl = np.linspace(0, 0 + (node_spacing * (quo + 1)), int(quo + 1), endpoint=False)
    samples = np.append(sampl, sampl[-1] + rem)
    # Queries along the spline must be in range [0, 1]
    query_points = np.clip(samples / max(samples), a_min=0.0, a_max=1.0)
    # Create spline points and evaluate at queries
    tck, _ = splprep(points.T, k=degree)
    return np.array(splev(query_points, tck)).T


def crop_img_from_swcs(img, swc_folder, output_image_dir, transform, add_offset=True, output_aligned_swcs=True):
    Views = imglib2.Views
    ImageJFunctions = imglib2.ImageJFunctions
    Intervals = imglib2.Intervals
    Calibration = imagej1.Calibration

    arrs = []
    names = []
    for f in os.listdir(swc_folder):
        if os.path.isfile(os.path.join(swc_folder, f)) and f.endswith(".swc"):
            swc_arr = swcutil.swc_to_ndarray(os.path.join(swc_folder, f), add_offset)
            names.append(swcutil.path_to_name(f))
            arrs.append(swc_arr)

    world_points = np.vstack(arrs)[:, 2:5]
    voxel_points = transform.forward(world_points)

    bbmin, bbmax = chunkutil.bbox(voxel_points)
    bbmin = bbmin.astype(int)
    bbmax = bbmax.astype(int)

    chunk_metadata = {}
    chunk_metadata['chunk_origin'] = bbmin.tolist()
    chunk_metadata['chunk_shape'] = (bbmax - bbmin + 1).tolist()
    chunk_metadata['voxel_spacing'] = transform.scale.tolist()

    interval = Intervals.createMinMax(
        bbmin[0], bbmin[1], bbmin[2],
        bbmax[0], bbmax[1], bbmax[2]
    )
    block = Views.interval(img, interval)

    imp = ImageJFunctions.wrap(Views.permute(Views.addDimension(block, 0, 0), 2, 3), "Img")

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
            swcutil.ndarray_to_swc(swc_arr, os.path.join(output_swc_folder, names[i] + "_aligned.swc"))

    with open(os.path.join(output_image_dir, "block_metadata.json"), 'w') as f:
        json.dump(chunk_metadata, f)

    print("Done")


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
    print("Volume dimensions: ", imglib2.Intervals.dimensionsAsLongArray(lazyimg))
    lazyimg_ch0 = imglib2.Views.hyperSlice(lazyimg, lazyimg.numDimensions() - 1, 0)
    output_image_dir = r"C:\Users\cameron.arshadi\Desktop\repos\20210812-AG-training-data\blocks\validation"
    if not os.path.isdir(output_image_dir):
        os.mkdir(output_image_dir)
    crop_img_from_swcs(lazyimg_ch0, swc_folder, output_image_dir, um2vx, True, True)
