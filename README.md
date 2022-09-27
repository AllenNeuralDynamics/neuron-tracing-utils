## Installation
```shell
conda create -n refinery -c conda-forge pyimagej openjdk=8
conda activate refinery
cd refinery
pip install .
```
To use the command-line entry points below, make sure the conda environment is active.
## Typical Workflow
### 1. Transform 

Transform Janelia workstation-created `.swc` files from world to voxel coordinates and back

```shell
transform --input="/path/to/input_swcs" --output="/path/to/output_swcs" --transform="/path/to/transform.txt"  
```

***arguments***:

```--input``` the folder containing `.swc` files to transform

```--output``` the folder to export transformed `.swc` files

```--transform``` path to the `transform.txt` file for the sample used to create the `.swc` files

```--to-world``` convert from voxel coordinates to JWS world coordinates

---

### 2. Fix out-of-bounds points
Prune points that lay outside the image volume

```shell
fix_swcs --input="/path/to/input_swcs" --output="/path/to/output_swcs" --images="/path/to/input_images" --mode=clip
```

***arguments***:

```--input``` the folder containing `.swc` files to prune

```--output``` the folder to export pruned `.swc` files

```--images``` the folder containing the images on which `.swc`s were traced

```--mode``` how to handle out-of-bounds points



---

### 3. Refine
Medial axis refinement to snap nodes to center of fluorescent signal

```shell
refine --input="/path/to/input_swcs" --output="/path/to/output_swcs" --images="/path/to/input_images"
```

***arguments***:

```--input``` the folder containing `.swc` files to refine

```--output``` the folder to export refined `.swc` files

```--images``` the folder containing the images on which `.swc`s were traced

```--mode``` TODO

---

### 4. A-star
A-star search refinement between adjacent nodes to create dense tracings

```shell
astar --input="/path/to/input_swcs" --output="/path/to/output_swcs" --images="/path/to/input_images" --voxel-size="0.3,0.3,1.0"
```

***arguments***:

```--input``` the folder containing `.swc` files to refine

```--output``` the folder to export refined `.swc` files

```--images``` the folder containing the images on which `.swc`s were traced

```--transform``` path to the `transform.txt` file for the sample used to create the SWCs

```--voxel-size``` voxel spacing for the images on which .swcs were traced, as a string of comma-separated floats in XYZ order. E.g., "0.3,0.3,1.0"

Either `--voxel-size` or `--transform` must be specified, but not both.

---

### 5. Fill
Seeded-volume segmentation to generate masks of the tracings

```shell
fill --input="/path/to/input_swcs" --output="/path/to/output_masks" --images="/path/to/input_images" --threshold=0.03 --voxel-size="0.3,0.3,1.0"
```

***arguments***:

```--input``` the folder containing `.swc` files to refine

```--output``` the folder to export refined `.swc` files

```--images``` the folder containing the images on which `.swc`s were traced

```--threshold``` distance threshold for the filling algorithm (must be > 0). Default is `0.03`. A larger value will 
result in thicker annotations, but may tread into background if set too high.

```--transform``` path to the `transform.txt` file for the sample used to create the `.swc`s

```--voxel-size``` voxel spacing for the images on which `.swc`s were traced, as a string of comma-separated floats in X,Y,Z order, e.g., `"0.3,0.3,1.0"`

Either `--voxel-size` or `--transform` must be specified, but not both.

---

### Misc.

Render maximum intensity projections of images along with projected tracings

```shell
render_mips --input="/path/to/input_swcs" --output="/path/to/output_MIPs" --images="/path/to/input_images" --vmin=12000 --vmax=15000
```

***arguments***

```--input``` the folder containing `.swc` files to render

```--output``` the folder to export MIPs

```--images``` the folder containing the images on which `.swc`s were traced

```--vmin``` minimum intensity of the desired display range

```--vmax``` maximum intensity of the desired display range



