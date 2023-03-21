import scyjava


def _java_setup():
    """
    Lazy initialization function for Java-dependent data structures.
    Do not call this directly; use scyjava.start_jvm() instead.
    """
    global IJ
    IJ = scyjava.jimport("ij.IJ")
    global Calibration
    Calibration = scyjava.jimport("ij.measure.Calibration")
    global ImagePlus
    ImagePlus = scyjava.jimport("ij.ImagePlus")


scyjava.when_jvm_starts(_java_setup)
