from refinery.util.java import imglib2


def get_hyperslice(img, ndim=3):
    while img.numDimensions() > ndim:
        img = imglib2.Views.hyperSlice(img, img.numDimensions() - 1, 0)
    return img
