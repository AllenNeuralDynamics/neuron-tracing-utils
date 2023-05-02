import argparse
import subprocess
from pathlib import Path


def upload_directory_to_s3(local_path, s3_bucket, s3_path):
    command = f"aws s3 sync {local_path} s3://{s3_bucket}/{s3_path}"
    subprocess.run(command, check=True)


def convert_ktx(tiff_path, out, script_path, voxel_size, threads=8):
    command = f"python {script_path} -f {tiff_path} -o {out} -d 2ndmax -t {threads} --ktxonly --voxsize {voxel_size} --verbose"
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Process images in a directory")

    # Add the input argument
    parser.add_argument(
        "-i",
        dest="input",
        type=str,
        help="The directory containing the image files",
    )
    parser.add_argument(
        "-s",
        dest="script",
        type=str,
        help="path to tiff2octree.py script",
    )
    parser.add_argument("-b", dest="bucket", type=str)
    parser.add_argument("-p", dest="path", type=str)
    parser.add_argument("-v", dest="voxel_size", type=str)

    # Parse the command line arguments
    args = parser.parse_args()

    for f in Path(args.input).iterdir():
        if not f.name.startswith("block"):
            continue
        image_path = f / f"{f.name}.tiff"
        ktx_out = f / "octree"
        convert_ktx(image_path, ktx_out, args.script, args.voxel_size)
        upload_directory_to_s3(ktx_out, args.bucket, f"{args.path}/{f.name}/octree")


if __name__ == "__main__":
    main()
