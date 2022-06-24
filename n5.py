import scyjava


def _java_setup():
    global N5FSReader
    N5FSReader = scyjava.jimport("org.janelia.saalfeldlab.n5.N5FSReader")
    global N5FSWriter
    N5FSWriter = scyjava.jimport("org.janelia.saalfeldlab.n5.N5FSWriter")
    global N5Utils
    N5Utils = scyjava.jimport("org.janelia.saalfeldlab.n5.imglib2.N5Utils")
    global GzipCompression
    GzipCompression = scyjava.jimport("org.janelia.saalfeldlab.n5.GzipCompression")


scyjava.when_jvm_starts(_java_setup)
