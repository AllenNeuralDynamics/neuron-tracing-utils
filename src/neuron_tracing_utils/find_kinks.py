import argparse
import glob
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import matplotlib
import numpy as np
import scyjava
from tensorstore import TensorStore

from neuron_tracing_utils.util.ioutil import open_ts
from neuron_tracing_utils.util import sntutil, swcutil

matplotlib.use("TkAgg")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)


def _time(func: Callable) -> Callable:
    """Decorator to measure and log the execution time of a function.

    Parameters
    ----------
    func : Callable
        The function to be timed.

    Returns
    -------
    Callable
        The wrapped function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logging.info(
            f"Execution time for {func.__name__}: {elapsed_time:.6f} seconds"
        )
        return result

    return wrapper


def _vectors(
    A: np.ndarray, B: np.ndarray, C: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Calculate vectors AB and BC for given points A, B, and C.

    Parameters
    ----------
    A, B, C : np.ndarray
        Points (3D) in space.

    Returns
    -------
    tuple of np.ndarray
        Vectors AB and BC.
    """
    return B - A, C - B


def _angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """
    Calculate the angle between three 3D points, with the middle point as the vertex.

    Parameters
    ----------
    A, B, C : np.ndarray
        Points (3D) in the triangle.

    Returns
    -------
    float
        Angle in degrees at point B formed by the three points.
    """
    AB, BC = _vectors(A, B, C)
    mag_prod = np.linalg.norm(AB) * np.linalg.norm(BC)
    if mag_prod == 0:
        logging.debug("Magnitude of vector is zero at point" + str(B))
        # Duplicate nodes are common in SWC files, so this is not necessarily an error.
        return 180
    quo = np.clip(np.dot(AB, BC) / mag_prod, -1, 1)
    return np.degrees(np.arccos(quo))


def _heightt(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """
    Calculate the height of the triangle formed by three 3D points, with the middle point as the tip.

    Parameters
    ----------
    A, B, C : np.ndarray
        Points (3D) in the triangle.

    Returns
    -------
    float
        Height of the triangle from the base (line AC) to the tip (point B).
    """
    AB, BC = _vectors(A, B, C)
    area = 0.5 * np.linalg.norm(np.cross(AB, BC))
    if area == 0:
        # This can happen if the 3 points are collinear,
        # or if one of them is a duplicate.
        return 0
    base = np.linalg.norm(C - A)
    if base == 0:
        logging.warning("Base of triangle is zero.")
        return np.linalg.norm(AB)
    return 2 * area / base


def find_kinks(
    g: Any,
    arr: TensorStore,
    im_mean: float,
    im_std: float,
    n_std: int = 2,
    min_angle: float = 100,
    max_height: float = 20,
    max_workers: int = 8,

):
    """Find kinks in the given graph based on angles, heights, and voxel intensities.

    Parameters
    ----------
    g : DirectedWeightedGraph
        Graph object representing the neuron.
    arr : TensorStore
        Image volume.
    im_mean : float
        Mean intensity value.
    im_std : float
        Standard deviation of intensity values.
    n_std : int, optional
        Number of standard deviations for the intensity threshold. Default is 2.
    min_angle : int, optional
        Minimum angle for detecting kinks. Default is 120.
    max_height : int, optional
        Maximum height for detecting kinks. Default is 20.
    max_workers : int, optional
        Maximum number of threads to use for parallel processing. Default is 8.

    Returns
    -------
    list
        Coordinates of detected kinks.
    """

    int_threshold = im_mean + n_std * im_std

    kink_coords = []

    # Function to check if a vertex is a kink and return the coordinates if true
    def _process_vertex(v):
        if vertex_is_kink(v, g, arr, min_angle, max_height, int_threshold):
            return _asarray(v)
        return None

    # Use a ThreadPoolExecutor to process the vertices concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_process_vertex, list(g.vertexSet()))

    # Collect the results into the kink_coords list
    for result in results:
        if result is not None:
            kink_coords.append(result)

    return kink_coords


def vertex_is_kink(
    v: Any,
    g: Any,
    arr: TensorStore,
    min_angle: float = 100,
    max_height: float = 20,
    int_threshold: float = 60,
):
    """
    Determine if a vertex in the graph represents a kink based on specific geometric and intensity criteria.

    A vertex is considered a kink if it satisfies all the following conditions:
        - It has one incoming and one outgoing edge.
        - The angle between the vectors formed by the incoming and outgoing edges is less than `min_angle`.
        - The height of the triangle formed by the vertex and its neighbors is greater than `max_height`.
        - The median intensities along the vectors are less than `int_threshold`.

    Parameters
    ----------
    v : SWCPoint
        Vertex object representing the point in the graph.
    g : DirectedWeightedGraph
        Graph object representing the neuron.
    arr : TensorStore
        Image volume.
    min_angle : float, optional
        Minimum angle for detecting kinks (in degrees). Default is 100.
    max_height : float, optional
        Maximum height for detecting kinks. Default is 20.
    int_threshold : float, optional
        Intensity threshold for detecting kinks. Default is 60.

    Returns
    -------
    bool
        True if the vertex represents a kink, otherwise False.
    """
    if g.inDegreeOf(v) == 1 and g.outDegreeOf(v) == 1:
        p = next(iter(g.incomingEdgesOf(v))).getSource()
        q = next(iter(g.outgoingEdgesOf(v))).getTarget()
        p = _asarray(p)
        q = _asarray(q)
        v = _asarray(v)
        angle = _angle(p, v, q)
        height = _heightt(p, v, q)
        # Acute angles with large heights are probable kinks
        if angle < min_angle and height > max_height:
            # This is pretty slow for remote volumes,
            # so we only do it if the angle and height are within the thresholds.
            pv_i, vq_i = _get_line_profiles(arr, p, v, q)
            # Low intensity values in the line profiles indicate kinks,
            # since the vectors pv and vq are passing through the image background,
            # which is much darker than the neurites where the 3 points lay.
            if (
                np.median(pv_i) < int_threshold
                and np.median(vq_i) < int_threshold
            ):
                # print(f"Angle: {angle}, Height: {height}")
                # print(pv_i.mean(), vq_i.mean())
                return True
    return False


def _asarray(point: Any) -> np.ndarray:
    """Convert an SNT SWCPoint object to a NumPy array.

    Parameters
    ----------
    point : SWCPoint

    Returns
    -------
    np.ndarray
        3D coordinates of the point as a NumPy array.
    """
    return np.array([point.getX(), point.getY(), point.getZ()])


def _iter_swcs(directory: str):
    """
    Generator function to iterate over all .swc files in a given directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing .swc files.

    Yields
    ------
    str
        Path to each .swc file found in the directory.
    """
    search_pattern = os.path.join(directory, "*.swc")
    for file_path in glob.glob(search_pattern):
        yield file_path


def _bresenham_3d(start: np.ndarray, end: np.ndarray) -> list:
    """3D Bresenham's Line Algorithm.

    Adapted from: https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/

    Parameters
    ----------
    start, end : np.ndarray
        Starting and ending points of the line.

    Returns
    -------
    list
        List of points along the line.
    """
    x1, y1, z1 = start.astype(int)
    x2, y2, z2 = end.astype(int)
    points = []

    dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
    xs, ys, zs = (
        1 if x1 < x2 else -1,
        1 if y1 < y2 else -1,
        1 if z1 < z2 else -1,
    )

    if dx >= dy and dx >= dz:
        e1, e2 = 2 * dy - dx, 2 * dz - dx
        for _ in range(dx + 1):
            points.append([x1, y1, z1])
            if e1 >= 0:
                y1 += ys
                e1 -= 2 * dx
            if e2 >= 0:
                z1 += zs
                e2 -= 2 * dx
            e1 += 2 * dy
            e2 += 2 * dz
            x1 += xs
    elif dy >= dx and dy >= dz:
        e1, e2 = 2 * dx - dy, 2 * dz - dy
        for _ in range(dy + 1):
            points.append([x1, y1, z1])
            if e1 >= 0:
                x1 += xs
                e1 -= 2 * dy
            if e2 >= 0:
                z1 += zs
                e2 -= 2 * dy
            e1 += 2 * dx
            e2 += 2 * dz
            y1 += ys
    else:
        e1, e2 = 2 * dy - dz, 2 * dx - dz
        for _ in range(dz + 1):
            points.append([x1, y1, z1])
            if e1 >= 0:
                y1 += ys
                e1 -= 2 * dz
            if e2 >= 0:
                x1 += xs
                e2 -= 2 * dz
            e1 += 2 * dy
            e2 += 2 * dx
            z1 += zs

    return points


# @_time
def _get_line_profiles(
    arr: TensorStore, A: np.ndarray, B: np.ndarray, C: np.ndarray
) -> (np.ndarray, np.ndarray):
    """Sample the voxel intensities along vectors AB and BC.

    Parameters
    ----------
    arr : array-like
        Image volume.
    A, B, C : np.ndarray
        Points (3D) defining vectors AB and BC.

    Returns
    -------
    np.ndarray, np.ndarray
        Voxel intensities along vectors AB and BC.
    """
    # Get coordinates of voxels along vectors AB and BC
    points_ab = _bresenham_3d(A, B)
    points_bc = _bresenham_3d(B, C)

    # rearrange to ZYX order
    indices_ab = np.array(points_ab)[:, [2, 1, 0]].T
    indices_bc = np.array(points_bc)[:, [2, 1, 0]].T

    # Sample voxel intensities
    intensities_ab = arr[tuple([0] * (len(arr.shape) - 3)) + tuple(indices_ab)]
    intensities_bc = arr[tuple([0] * (len(arr.shape) - 3)) + tuple(indices_bc)]

    return np.array(intensities_ab), np.array(intensities_bc)


@_time
def _sample_cube(
    volume: TensorStore,
    cube_size: int,
    shape_z: int,
    shape_y: int,
    shape_x: int,
) -> (float, float):
    """Randomly sample a cube from the volume and return the sum of intensities and squared intensities.

    Parameters
    ----------
    volume : array-like
        Image volume.
    cube_size : int
        Size of the cube to sample.
    shape_z, shape_y, shape_x : int
        Dimensions of the volume.

    Returns
    -------
    float, float
        Sum of intensities and sum of squared intensities in the cube.
    """
    # Randomly select the starting coordinates for the cube
    start_z = np.random.randint(0, shape_z - cube_size)
    start_y = np.random.randint(0, shape_y - cube_size)
    start_x = np.random.randint(0, shape_x - cube_size)

    # Extract the cube from the volume
    cube = np.array(
        volume[
            ...,
            start_z : start_z + cube_size,
            start_y : start_y + cube_size,
            start_x : start_x + cube_size,
        ]
    ).astype(np.float64)

    # Return the sum of intensities and sum of squared intensities in the cube
    return cube.sum(), (cube**2).sum()


def _calc_intensity_stats(
    volume: TensorStore,
    cube_size: int = 64,
    num_samples: int = 1,
    max_workers: int = 4,
) -> (float, float):
    """
    Estimate the mean and standard deviation of voxel intensity by randomly sampling subvolumes.

    Parameters:
    volume (array-like): Image volume.
    cube_size (int): Size of the cube to sample (in voxels along each edge).
    num_samples (int): Number of cubes to sample.

    Returns:
    float, float: Estimated mean voxel intensity, estimated standard deviation of voxel intensity.
    """
    # Get the shape of the volume
    shape_t, shape_c, shape_z, shape_y, shape_x = volume.shape

    # Ensure the cube size is smaller than the volume dimensions
    cube_size = min(cube_size, shape_x, shape_y, shape_z)

    # Initialize variables to store the sum of intensities and sum of squared intensities
    sum_int = 0
    sum_sq_int = 0

    # Use a ThreadPoolExecutor to sample cubes in parallel
    with ThreadPoolExecutor(max_workers) as executor:
        futures = [
            executor.submit(
                _sample_cube, volume, cube_size, shape_z, shape_y, shape_x
            )
            for _ in range(num_samples)
        ]
        for future in futures:
            sum_cube, sum_sq_cube = future.result()
            sum_int += sum_cube
            sum_sq_int += sum_sq_cube

    # Compute the estimated mean intensity
    total_voxels = cube_size**3 * num_samples
    mean_int = sum_int / total_voxels
    # Compute the estimated standard deviation
    std_int = np.sqrt(
        (sum_sq_int - (sum_int**2) / total_voxels) / total_voxels
    )

    return mean_int, std_int


def parse_arguments():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process .swc files in a directory."
    )
    parser.add_argument(
        "--swc-dir",
        default=r"C:\Users\cameron.arshadi\Downloads\exaSPIM_653158_2023-06-01_20-41-38\Complete\all_voxel",
        help="Path to the directory containing .swc files.",
    )
    parser.add_argument(
        "--zarr-path",
        default=r"https://aind-open-data.s3.amazonaws.com/exaSPIM_653158_2023-06-01_20-41-38_fusion_2023-06-12_11-58-05/fused.zarr",
        help="Path to the zarr volume.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=[0.748, 0.748, 1.0],
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
    )
    args = parser.parse_args()
    return args


def main():
    scyjava.start_jvm()

    args = parse_arguments()
    swc_dir = args.swc_dir

    scale = np.array(args.voxel_size)
    print(scale)

    arr = open_ts(args.zarr_path, dataset="0", total_bytes_limit=int(1e9))
    logging.info(arr)

    t0 = time.time()
    mean, std = _calc_intensity_stats(arr, cube_size=256, num_samples=100)
    # mean = 50
    # std = 10
    logging.info(f"estimating intensity stats took {time.time() - t0} seconds")
    logging.info(f"estimated mean: {mean}, std: {std}")

    all_kinks = []
    for file_path in _iter_swcs(swc_dir):
        logging.info(f"Processing file: {file_path}")
        a = swcutil.swc_to_ndarray(file_path, add_offset=True)
        g = sntutil.ndarray_to_graph(a)
        all_kinks.extend(find_kinks(g, arr, mean, std, n_std=2))

    logging.info(f"Found {len(all_kinks)} kinks.")
    for k in all_kinks:
        print(k * scale)

    with open(args.output, "w") as f:
        for k in all_kinks:
            f.write(f"{k * scale}\n")


if __name__ == "__main__":
    main()
