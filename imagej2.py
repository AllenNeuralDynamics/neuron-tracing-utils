import scyjava


def _java_setup():
    global Images
    Images = scyjava.jimport("net.imagej.util.Images")
    global Dataset
    Dataset = scyjava.jimport("net.imagej.Dataset")
    global DefaultAxisType
    DefaultAxisType = scyjava.jimport("net.imagej.axis.DefaultAxisType")
    global DefaultLinearAxis
    DefaultLinearAxis = scyjava.jimport("net.imagej.axis.DefaultLinearAxis")


scyjava.when_jvm_starts(_java_setup)
