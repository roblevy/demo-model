import pstats, cProfile

import pyximport
pyximport.install()

import clustering_c
import clustering
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
a = clustering.Network(adj)

cProfile.runctx("clustering_c.cluster(a)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
