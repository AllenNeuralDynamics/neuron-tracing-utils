import scyjava


def _java_setup():
    """
    Lazy initialization function for Java-dependent data structures.
    Do not call this directly; use scyjava.start_jvm() instead.
    """
    global AmazonS3ClientBuilder
    AmazonS3ClientBuilder = scyjava.jimport("com.amazonaws.services.s3.AmazonS3ClientBuilder")
    global ClientConfiguration
    ClientConfiguration = scyjava.jimport("com.amazonaws.ClientConfiguration")
    global Regions
    Regions = scyjava.jimport("com.amazonaws.regions.Regions")


scyjava.when_jvm_starts(_java_setup)
