import jpype.imports

# FIXME this would probably be a lot faster within Java
from refinery.util.java import imglib2


def local_intensities(imglib_iterable):
    import java.lang

    cursor = imglib_iterable.cursor()
    while cursor.hasNext():
        cursor.fwd()
        try:
            yield cursor.get().get()
        except java.lang.ArrayIndexOutOfBoundsException as e:
            # ArrayIndexOutOfBoundsException is expected here.
            # ImgLib cursors don't check whether they are in a valid
            # portion of the image when accessing pixels.
            continue


def get_hyperslice(img, ndim=3):
    while img.numDimensions() > ndim:
        img = imglib2.Views.hyperSlice(img, img.numDimensions()-1, 0)
    return img
