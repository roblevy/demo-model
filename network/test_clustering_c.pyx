cimport demo_model.network.clustering_c as clust
import demo_model.network.clustering_c as cluster
cimport numpy as np
import numpy as np
import pandas as pd

rows = [[0,1,1,0,0,0,0,0],
        [1,0,1,1,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [1,0,1,0,0,0,0,0],
        [0,0,0,1,0,1,1,0],
        [0,0,0,0,1,0,0,1],
        [0,0,0,0,0,0,0,1],
        [0,0,0,0,1,1,0,0],
        ]
names = ["A", "B", "C", "D", "E", "F", "G", "H"]
adj = pd.DataFrame(rows).astype(float)
adj.index = names
adj.columns = names
a = cluster.Network(adj)
c = pd.Series([0,0,0,1,0,1,1,1])

cdef community(c):
    cdef long[:] comm = np.array(c)
    return comm

def count_vals(series):
    series = pd.Series(series)
    return pd.Series(series).groupby(series).count()

cpdef test_weighted_random_int():
    cdef double[:] weights = np.array([0.8, 0.1, 0.1])
    res = []
    for i in range(1000):
        res.append(clust._weighted_random_int(weights))
    print count_vals(res)

cpdef test_rand_float():
    res = []
    for i in range(20):
        res.append(clust.rand_float())
    print count_vals(res)

cdef test_adhesion_ls():
    cdef double res
    clust.set_globals(a, c=community(c))
    print "adhesion node 4 with cluster 0"
    res = clust._adhesion_ls(l=4, s=0)
    print res
    print "adhesion node 4 with cluster 1"
    res = clust._adhesion_ls(l=4, s=1)
    print res

cpdef test_suite():
    test_rand_float()
    test_weighted_random_int()
    test_adhesion_ls()
