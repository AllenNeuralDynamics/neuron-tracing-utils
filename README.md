### Setup
```shell
conda create -n refinery -c conda-forge imglyb numpy scipy tifffile zarr scikit-image
conda activate refinery
```

### Typical Workflow
1. Transform Janelia workstation created .swc files from world to voxel coordinates

```python -m refinery.transform ...```

2. Prune points that lay outside the image volume

```python -m refinery.prune ...```

3. Medial axis refinement to snap points to center of fluorescent signal

```python -m refinery.refine ...```

4. A-star search refinement between adjacent points to create dense tracings

```python -m refinery.astar ...```

5. Seeded-volume segmentation

```python -m refinery.fill ...```


