"""
A series of clustering algorithms for use on an adjacency matrix which is
stored in the form of a pandas DataFrame.
"""

import numpy as np
import pandas as pd
import pyximport; pyximport.install()
import clustering_c

class Adjacency():
    def __init__(self, adjacency):
        self.adjacency = adjacency
        self.edges = adjacency[adjacency > 0]
        self.indegrees = self.edges.sum()
        self.outdegrees = self.edges.sum(1)
        self.num_of_edges = self.edges.sum().sum()

    def __repr__(self):
        return str(self.adjacency)
        
def _brute_force_modularity(adjacency):
    a = adjacency.adjacency
    n = len(a)
    names = list(a.index)
    binary = [format(x, '#0%ib' % (n + 2))[2:] for x in range(pow(2, n) / 2)]
    community_lists = [pd.Series(list(x), index=names).astype(np.int64) 
        for x in binary] 
    modularity = pd.Series(
        {str(c.values):
            clustering_c._modularity_given_communities(adjacency, c)
        for c in community_lists})
    return modularity
    
if __name__ == "__main__":
    # A network with two very obvious communities:
    # A, B, C, D and E, F, G, H. A single connection
    # from E to D joins the two communities
    rows = [[0,1,1,0,0,0,0,0],
            [1,0,1,1,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,0],
            [0,0,0,1,1,1,1,0],
            [0,0,0,0,1,0,0,1],
            [0,0,0,0,0,0,0,1],
            [0,0,0,0,1,1,0,0],
            ]
    names = ["A", "B", "C", "D", "E", "F", "G", "H"]
    adj = pd.DataFrame(rows).astype(float)
    adj.index = names
    adj.columns = names
    a = Adjacency(adj)
    test_communities = pd.Series([0,0,0,0,0,0,0,0], index=names)
    results = _brute_force_modularity(a).order()
    