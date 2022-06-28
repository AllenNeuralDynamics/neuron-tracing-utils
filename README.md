### Setup
```shell
conda create -n refinery -c conda-forge imglyb numpy scipy tifffile zarr scikit-image
conda activate refinery
```

### Typical Workflow
#### 1. Transform 

Transform Janelia workstation-created `.swc` files from world to voxel coordinates and back

```python transform.py ...```

***arguments***:

```--input``` the folder containing `.swc` files to transform

```--output``` the folder to export transformed `.swc` files

```--transform``` path to the `transform.txt` file for the sample used to create the `.swc` files

```--to-world``` convert from voxel coordinates to JWS world coordinates

#### 2. Prune
Prune points that lay outside the image volume

```python prune.py ...```

***arguments***:

```--input``` the folder containing `.swc` files to prune

```--output``` the folder to export pruned `.swc` files

```--images``` the folder containing the images on which `.swc`s were traced


#### 3. Refine
Medial axis refinement to snap nodes to center of fluorescent signal

```python refine.py ...```

***arguments***:

```--input``` the folder containing `.swc` files to refine

```--output``` the folder to export refined `.swc` files

```--images``` the folder containing the images on which `.swc`s were traced

```--mode``` TODO

#### 4. A-star
A-star search refinement between adjacent nodes to create dense tracings

```python astar.py ...```

***arguments***:

```--input``` the folder containing `.swc` files to refine

```--output``` the folder to export refined `.swc` files

```--images``` the folder containing the images on which `.swc`s were traced

```--transform``` path to the `transform.txt` file for the sample used to create the SWCs

```--voxel-size``` voxel spacing for the images on which .swcs were traced

Either `--voxel-size` or `--transform` must be specified, but not both.

#### 5. Fill
Seeded-volume segmentation to generate masks of the tracings

```python fill.py ...```

***arguments***:

```--input``` the folder containing `.swc` files to refine

```--output``` the folder to export refined `.swc` files

```--images``` the folder containing the images on which `.swc`s were traced

```--threshold``` distance threshold for the filling algorithm (must be > 0)

```--transform``` path to the `transform.txt` file for the sample used to create the `.swc`s

```--voxel-size``` voxel spacing for the images on which `.swc`s were traced

Either `--voxel-size` or `--transform` must be specified, but not both.

#### Misc.

Render maximum intensity projections of images along with projected tracings

```python render_mips.py ...```

***arguments***

```--input``` the folder containing `.swc` files to render

```--output``` the folder to export MIPs

```--images``` the folder containing the images on which `.swc`s were traced

```--vmin``` minimum intensity of the desired display range

```--vmax``` maximum intensity of the desired display range



