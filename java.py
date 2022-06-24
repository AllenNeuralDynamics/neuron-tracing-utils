import scyjava


def _java_setup():
    global Executors
    Executors = scyjava.jimport("java.util.concurrent.Executors")
    global Runtime
    Runtime = scyjava.jimport("java.lang.Runtime")


scyjava.when_jvm_starts(_java_setup)
