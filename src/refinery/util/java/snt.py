import scyjava


def _java_setup():
    """
    Lazy initialization function for Java-dependent data structures.
    Do not call this directly; use scyjava.start_jvm() instead.
    """
    global SNTService
    SNTService = scyjava.jimport("sc.fiji.snt.SNTService")
    global SWCPoint
    SWCPoint = scyjava.jimport("sc.fiji.snt.util.SWCPoint")
    global PointInImage
    PointInImage = scyjava.jimport("sc.fiji.snt.util.PointInImage")
    global DirectedWeightedGraph
    DirectedWeightedGraph = scyjava.jimport(
        "sc.fiji.snt.analysis.graph.DirectedWeightedGraph"
    )
    global Graphs
    Graphs = scyjava.jimport("org.jgrapht.Graphs")
    global Tree
    Tree = scyjava.jimport("sc.fiji.snt.Tree")
    global ImgUtils
    ImgUtils = scyjava.jimport("sc.fiji.snt.util.ImgUtils")
    global SNTUtils
    SNTUtils = scyjava.jimport("sc.fiji.snt.SNTUtils")
    global SNT
    SNT = scyjava.jimport("sc.fiji.snt.SNT")
    global OneMinusErf
    OneMinusErf = scyjava.jimport("sc.fiji.snt.tracing.cost.OneMinusErf")
    global Reciprocal
    Reciprocal = scyjava.jimport("sc.fiji.snt.tracing.cost.Reciprocal")
    global Dijkstra
    Dijkstra = scyjava.jimport("sc.fiji.snt.tracing.heuristic.Dijkstra")
    global Euclidean
    Euclidean = scyjava.jimport("sc.fiji.snt.tracing.heuristic.Euclidean")
    global BiSearch
    BiSearch = scyjava.jimport("sc.fiji.snt.tracing.BiSearch")
    global FillerThread
    FillerThread = scyjava.jimport("sc.fiji.snt.tracing.FillerThread")
    global FillConverter
    FillConverter = scyjava.jimport("sc.fiji.snt.FillConverter")
    global TreeAnalyzer
    TreeAnalyzer = scyjava.jimport("sc.fiji.snt.analysis.TreeAnalyzer")
    global PathFitter
    PathFitter = scyjava.jimport("sc.fiji.snt.PathFitter")


scyjava.when_jvm_starts(_java_setup)
