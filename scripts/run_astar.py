import logging
import os
import tempfile

import argschema
import scyjava
import yaml

from neuron_tracing_utils.astar import astar_swcs
from neuron_tracing_utils.fix_swcs import fix_swcs
from neuron_tracing_utils.transform import transform_swcs, WorldToVoxel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)


class MySchema(argschema.ArgSchema):
    config = argschema.fields.InputFile(
        required=True,
        metadata={"description": "path to config file"}
    )


def _load_yaml(file_path):
    with open(file_path, 'r') as yaml_file:
        try:
            yaml_dict = yaml.safe_load(yaml_file)
            return yaml_dict
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")


def main():
    parser = argschema.argschema_parser.ArgSchemaParser(schema_type=MySchema)
    args = parser.args
    logging.info(args)

    config_file = args["config"]
    config = _load_yaml(config_file)
    logging.info(config)

    image_path = config["image"]
    swc_dir = config["swcs"]
    voxel_size = config["voxel_size"]
    out_dir = config["output"]

    scyjava.start_jvm()

    # create a temporary directory to store intermediate .swc outputs
    with tempfile.TemporaryDirectory() as temp_swc_dir:
        transformed_dir = os.path.join(temp_swc_dir, "voxel")
        os.mkdir(transformed_dir)

        transform = WorldToVoxel(None, voxel_size)
        # convert from physical coordinates to pixel coordinates
        transform_swcs(swc_dir, transformed_dir, transform, forward=True, swap_xy=False)

        fixed_dir = os.path.join(temp_swc_dir, "fixed")
        os.mkdir(fixed_dir)
        # Fix any out-of-bounds points
        fix_swcs(transformed_dir, fixed_dir, image_path, mode="clip")

        astar_dir = os.path.join(temp_swc_dir, "astar")
        os.mkdir(astar_dir)
        # Run the A* search
        astar_swcs(fixed_dir, astar_dir, image_path, voxel_size, "relative_difference")
        # Convert back to physical coordinates and copy to output directory
        transform_swcs(astar_dir, out_dir, transform, forward=False, swap_xy=False)


if __name__ == "__main__":
    main()
