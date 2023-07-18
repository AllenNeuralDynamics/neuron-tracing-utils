import multiprocessing
import os
import glob
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
import argparse
from typing import Any

import scyjava
from neuron_tracing_utils.resample import resample_tree
from neuron_tracing_utils.util.java import snt
from neuron_tracing_utils.util import swcutil, sntutil


def unzip_file(zip_path: str, extract_path: str = None) -> None:
    """
    Unzips a given zip file.

    Parameters:
        zip_path (str): The path of the zip file to be unzipped.
        extract_path (str, optional): The path where the unzipped files should be placed.
            If not provided, the files will be unzipped to the same directory as the zip file.

    Returns:
        None
    """
    if not extract_path:
        extract_path = os.path.dirname(zip_path)

    if not os.path.isfile(zip_path):
        print(f"No file found at {zip_path}")
        return

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


def process_swc_file(swc_path: str, length_threshold: int = 200, voxel_size: tuple = (1, 1, 1),
                     node_spacing: int = 20, radius: float = 1.0) -> Any:
    """
    Processes an SWC file, performs resampling and filtering based on cable length.

    Parameters:
        swc_path (str): The path of the SWC file to be processed.
        length_threshold (int, optional): The minimum length of cable to consider, default is 200.
        voxel_size (tuple, optional): The scaling factor for each dimension, default is (1, 1, 1).
        node_spacing (int, optional): The spacing between nodes after resampling, default is 20.
        radius (float, optional): The radius to be set for the resampled tree, default is 1.0.

    Returns:
        snt.Tree: The processed Tree object if it passes the length threshold, else None.
    """
    arr = swcutil.swc_to_ndarray(swc_path)
    arr[:, 1] = 2

    g = sntutil.ndarray_to_graph(arr)

    tree = g.getTree()
    tree.scale(*voxel_size)
    resample_tree(tree, node_spacing)
    tree.setRadii(radius)

    analyzer = snt.TreeAnalyzer(tree)
    if analyzer.getCableLength() < length_threshold:
        return None

    return tree


def process_zip_file(zip_path: str, output_dir: str) -> None:
    """
    Processes the files in a given zip file and saves the processed files to the output directory.

    Parameters:
        zip_path (str): The path of the zip file.
        output_dir (str): The directory where the processed files will be saved.

    Returns:
        None
    """
    scyjava.start_jvm()

    with tempfile.TemporaryDirectory() as temp_dir:
        unzip_file(zip_path, temp_dir)

        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)

                tree = process_swc_file(file_path)
                if tree is not None:
                    tree.saveAsSWC(os.path.join(output_dir, file))


def process_all_zip_files(input_dir: str, output_dir: str, length_threshold: int, voxel_size: tuple,
                          node_spacing: int, radius: float, max_workers: int = None) -> None:
    """
    Processes all zip files in the input directory using multiple workers and saves the processed files.

    Parameters:
        input_dir (str): The directory containing the zip files.
        output_dir (str): The directory where the processed files will be saved.
        length_threshold (int): The minimum length of cable to consider.
        voxel_size (tuple): The scaling factor for each dimension.
        node_spacing (int): The spacing between nodes after resampling.
        radius (float): The radius to be set for the resampled tree.
        max_workers (int, optional): The maximum number of workers to use, default is the number of processors.

    Returns:
        None
    """
    zip_files = glob.glob(f"{input_dir}/*.zip")
    total_files = len(zip_files)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for zip_file in zip_files:
            futures.append(executor.submit(process_zip_file, zip_file, output_dir))
        for i, fut in enumerate(futures):
            print(f"Processing zip {i+1}/{total_files}")
            try:
                fut.result()
            except Exception as e:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SWC files in zip format.')
    parser.add_argument('-i', help='The input directory containing zip files.')
    parser.add_argument('-o', help='The output directory to save processed files.')
    parser.add_argument('--length_threshold', type=int, default=200, help='The minimum length of cable to consider.')
    parser.add_argument('--voxel_size', nargs=3, type=float, default=(0.748, 0.748, 1.0), help='The scaling factor for each dimension.')
    parser.add_argument('--node_spacing', type=int, default=20, help='The spacing between nodes after resampling.')
    parser.add_argument('--radius', type=float, default=1.0, help='The radius to be set for the resampled tree.')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), help='The number of worker processes to use.')
    args = parser.parse_args()

    if not os.path.isdir(args.o):
        os.makedirs(args.o)

    process_all_zip_files(args.i, args.o, args.length_threshold, args.voxel_size,
                          args.node_spacing, args.radius, args.workers)
