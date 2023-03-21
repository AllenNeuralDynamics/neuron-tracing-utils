## Installation
```shell
conda create -n ntu -c conda-forge pyimagej openjdk=8
conda activate ntu
cd neuron-tracing-utils
pip install .
```
To use the command-line entry points below, make sure the conda environment is active.
## Typical Workflow
### 1. Transform 

Transform Janelia workstation-created `.swc` files from world to voxel coordinates and back

```shell
transform --input="/path/to/input_swcs" --output="/path/to/output_swcs" --transform="/path/to/transform.txt"  
```

```
usage: transform [-h] [--input INPUT] [--output OUTPUT] [--transform TRANSFORM] [--to-world] [--log-level LOG_LEVEL]
                 [--swap-xy]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         directory of .swc files to transform
  --output OUTPUT       directory to output transformed .swc files
  --transform TRANSFORM
                        path to the "transform.txt" file
  --to-world            transform voxel to world coordinates
  --log-level LOG_LEVEL
  --swap-xy             swap XY coordinates
```

---

### 2. Fix out-of-bounds points
Prune or clip vertices that lay outside the bounds of the image. If a vertex of degree > 1 is pruned, this will break
connectivity and result in additional .swc outputs.

```shell
fix_swcs --input="/path/to/input_swcs" --output="/path/to/output_swcs" --images="/path/to/input_images" --mode=clip
```

```
usage: fix_swcs [-h] [--input INPUT] [--output OUTPUT] [--images IMAGES] [--mode {clip,prune}] [--log-level LOG_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         directory of .swc files to prune
  --output OUTPUT       directory to output pruned .swc files
  --images IMAGES       directory of images associated with the .swc files
  --mode {clip,prune}   how to handle out-of-bounds points
  --log-level LOG_LEVEL
```

---

### 3. Refine
Medial axis refinement and (optional) radius assignment to tracings

```shell
refine --input="/path/to/input_swcs" --output="/path/to/output_swcs" --images="/path/to/input_images" --radius=2
```

```
usage: refine [-h] [--input INPUT] [--output OUTPUT] [--images IMAGES] [--mode {naive,fit}] [--radius RADIUS]
              [--log-level LOG_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         directory of .swc files to refine
  --output OUTPUT       directory to output refined .swc files
  --images IMAGES       directory of images associated with the .swc files
  --mode {naive,fit}    algorithm type
  --radius RADIUS       search radius for point refinement
  --log-level LOG_LEVEL
```

---

### 4. A-star
A-star search refinement between adjacent pairs of points to create dense tracings

```shell
astar --input="/path/to/input_swcs" --output="/path/to/output_swcs" --images="/path/to/input_images" --voxel-size="0.3,0.3,1.0"
```

```
usage: astar [-h] [--input INPUT] [--output OUTPUT] [--images IMAGES] [--transform TRANSFORM]
             [--voxel-size VOXEL_SIZE] [--log-level LOG_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         directory of .swc files to refine
  --output OUTPUT       directory to output refined .swc files
  --images IMAGES       directory of images associated with the .swc files
  --transform TRANSFORM
                        path to the "transform.txt" file
  --voxel-size VOXEL_SIZE
                        voxel size for images, as a string in XYZ order, e.g., '0.3,0.3,1.0'
  --log-level LOG_LEVEL
```

Either `--voxel-size` or `--transform` must be specified, but not both.

---

### 5. Fill
Seeded-volume segmentation to generate masks (grayscale, labelling, binary) from the tracings

```shell
fill --input="/path/to/input_swcs" --output="/path/to/output_masks" --images="/path/to/input_images" --threshold=0.03 --voxel-size="0.3,0.3,1.0" --cost=reciprocal
```

```
usage: fill [-h] [--input INPUT] [--output OUTPUT] [--images IMAGES] [--threshold THRESHOLD] [--transform TRANSFORM]
            [--voxel-size VOXEL_SIZE] [--cost {reciprocal,one-minus-erf}] [--log-level LOG_LEVEL] [--n5]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         directory of .swc files to fill
  --output OUTPUT       directory to output mask volumes
  --images IMAGES       directory of images associated with the .swc files
  --threshold THRESHOLD
                        distance threshold for fill algorithm
  --transform TRANSFORM
                        path to the "transform.txt" file
  --voxel-size VOXEL_SIZE
                        voxel size of images
  --cost {reciprocal,one-minus-erf}
                        cost function for the Dijkstra search
  --log-level LOG_LEVEL
  --n5                  save masks as n5. Otherwise, save as Tiff.
```

Either `--voxel-size` or `--transform` must be specified, but not both.

---

### Misc.

Render maximum intensity projections of images along with projected tracings

```shell
render_mips --output="/path/to/output_MIPs" --images="/path/to/input_images" --swcs="/path/to/swcs" --masks="/path/to/masks" --vmin=12000 --vmax=15000
```

```
usage: render_mips [-h] [--output OUTPUT] [--images IMAGES] [--swcs SWCS] [--masks MASKS] [--vmin VMIN] [--vmax VMAX]
                   [--log-level LOG_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       directory to output MIPs
  --images IMAGES       directory of images associated with the .swc files
  --swcs SWCS           directory of .swc files to render
  --masks MASKS         directory of masks
  --vmin VMIN           minimum intensity of the desired display range
  --vmax VMAX           maximum intensity of the desired display range
  --log-level LOG_LEVEL
```

---

Resample tracings to have fixed spacing between adjacent pairs of points

```shell
resample --input="/path/to/input_swcs" --output="/path/to/output_swcs" --spacing=5.0
```

```
usage: resample [-h] [--input INPUT] [--output OUTPUT] [--spacing SPACING] [--log-level LOG_LEVEL]

Resample .swc files to have even spacing between consecutive nodes

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         directory of .swc files to resample
  --output OUTPUT       directory to output resampled .swc files
  --spacing SPACING     target spacing between consecutive pairs of points, in spatial units given by the SWC. For
                        example, if your SWCs are represented in micrometers, use micrometers. If they are in pixels,
                        use pixels, etc.
  --log-level LOG_LEVEL
```
