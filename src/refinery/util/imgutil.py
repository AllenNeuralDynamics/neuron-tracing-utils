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
        img = imglib2.Views.hyperSlice(img, img.numDimensions() - 1, 0)
    return img


def interploate(img):
    floatImg = imglib2.Converters.convert(
        imglib2.RandomAccessibleInterval @ img,
        imglib2.RealDoubleConverter(),
        imglib2.DoubleType(),
    )
    interpolant = imglib2.Views.interpolate(
        imglib2.Views.extendZero(floatImg),
        imglib2.NLinearInterpolatorFactory(),
    )

    return interpolant


def invert(img):

    inverted = img.factory().create(img.dimensionsAsLongArray())
    inverted_ra = inverted.randomAccess()

    cursor = img.cursor()
    maximum = 0
    while cursor.hasNext():
        cursor.fwd()
        maximum = max(maximum, cursor.get().get())

    lcursor = img.localizingCursor()
    while lcursor.hasNext():
        lcursor.fwd()
        inverted_ra.setPositionAndGet(lcursor).set(maximum - lcursor.get().get())

    return inverted


def dog(img, sigma1, sigma2, voxel_size, nthreads=1):
    from java.util.concurrent import Executors

    dog_img = imglib2.ArrayImgFactory(imglib2.FloatType()).create(img.dimensionsAsLongArray())
    k = sigma2 / sigma1
    sigmas1 = []
    sigmas2 = []
    for i in range(len(voxel_size)):
        s1 = sigma1 / voxel_size[i]
        s2 = k * s1
        sigmas1.append(s1)
        sigmas2.append(s2)
    imglib2.DifferenceOfGaussian.DoG(
        sigmas1,
        sigmas2,
        img,
        dog_img,
        Executors.newFixedThreadPool(nthreads)
    )
    return dog_img

