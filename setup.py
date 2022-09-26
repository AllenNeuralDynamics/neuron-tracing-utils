#!/bin/env python

from setuptools import setup, find_packages

exec(open('src/refinery/version.py').read())

setup(
    name="refinery",
    version=__version__,
    description="Tools for refinement and generation of dense voxel-wise training data from raw .swc files",
    license="MIT",
    package_dir={'': 'src'},
    packages=find_packages('src'),
    install_requires=[
        "numpy",
        "scipy",
        "zarr",
        "tifffile",
        "scikit-image"
    ],
    python_requires=">=3.7.0",
    entry_points={
        'console_scripts': [
            'transform=refinery.transform:main',
            'align_to_cube=refinery.align_to_cube:main',
            'fix_swcs=refinery.fix_swcs:main',
            'refine=refinery.refine:main',
            'astar=refinery.astar:main',
            'fill=refinery.fill:main',
            'render_mips=refinery.render_mips:main',
            'resample=refinery.resample:main',
        ]},
    classifiers=[],
)
