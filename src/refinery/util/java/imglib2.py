import scyjava


def _java_setup():
    """
    Lazy initialization function for Java-dependent data structures.
    Do not call this directly; use scyjava.start_jvm() instead.
    """
    global ImageJFunctions
    ImageJFunctions = scyjava.jimport(
        "net.imglib2.img.display.imagej.ImageJFunctions"
    )
    global IJLoader
    IJLoader = scyjava.jimport("net.imglib2.img.io.IJLoader")
    global ImgView
    ImgView = scyjava.jimport("net.imglib2.img.ImgView")
    global HyperSphere
    HyperSphere = scyjava.jimport(
        "net.imglib2.algorithm.region.hypersphere.HyperSphere"
    )
    global Intervals
    Intervals = scyjava.jimport("net.imglib2.util.Intervals")
    global ArrayImgFactory
    ArrayImgFactory = scyjava.jimport("net.imglib2.img.array.ArrayImgFactory")
    global ArrayImgs
    ArrayImgs = scyjava.jimport("net.imglib2.img.array.ArrayImgs")
    global Views
    Views = scyjava.jimport("net.imglib2.view.Views")
    global ComputeMinMax
    ComputeMinMax = scyjava.jimport(
        "net.imglib2.algorithm.stats.ComputeMinMax"
    )
    global BitType
    BitType = scyjava.jimport("net.imglib2.type.logic.BitType")
    global DoubleType
    DoubleType = scyjava.jimport("net.imglib2.type.numeric.real.DoubleType")
    global DiskCachedCellImgFactory
    DiskCachedCellImgFactory = scyjava.jimport(
        "net.imglib2.cache.img.DiskCachedCellImgFactory"
    )
    global UnsignedShortType
    UnsignedShortType = scyjava.jimport(
        "net.imglib2.type.numeric.integer.UnsignedShortType"
    )
    global UnsignedByteType
    UnsignedByteType = scyjava.jimport(
        "net.imglib2.type.numeric.integer.UnsignedByteType"
    )
    global Point
    Point = scyjava.jimport("net.imglib2.Point")
    global LoopBuilder
    LoopBuilder = scyjava.jimport("net.imglib2.loops.LoopBuilder")


scyjava.when_jvm_starts(_java_setup)
