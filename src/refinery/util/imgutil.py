from refinery.util.java import imglib2, snt


def get_hyperslice(img, ndim=3):
    while img.numDimensions() > ndim:
        img = imglib2.Views.hyperSlice(img, img.numDimensions() - 1, 0)
    return img


def filter(img, sigmas, spacing, filter="tubeness", lazy=False, cell_dim=32, threads=1):
    """
    Filter a RandomAccessibleInterval

    Args:
        img (RandomAccessibleInterval): the image to filter
        sigmas (list): list of physical scales (sigmas) to integrate over
        spacing (list): list of voxel size in each dimension
        filter (str): the type of filter
        lazy (bool): if True, construct a LazyCellImg and cache accessed regions on disk
    """
    Lazy = snt.Lazy
    Tubeness = snt.Tubeness
    Frangi = snt.Frangi
    DoubleType = imglib2.DoubleType
    CellImgFactory = imglib2.CellImgFactory

    if filter == "tubeness":
        op = Tubeness(sigmas, spacing, threads)
    elif filter == "frangi":
        op = Frangi(sigmas, spacing, 256, threads)
    else:
        raise ValueError(f"{filter}")

    if lazy:
        out = Lazy.process(
            img,
            img,
            [cell_dim] * img.numDimensions(),
            DoubleType(),
            op
        )
    else:
        out = CellImgFactory(DoubleType()).create(img.dimensionsAsLongArray())
        op.compute(img, out)

    return out
