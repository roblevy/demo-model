"""
A series of clustering algorithms for use on an adjacency matrix which is
stored in the form of a pandas DataFrame.
"""

import numpy as np
import pandas as pd
import pyximport; pyximport.install(reload_support=True)
import clustering_c
#reload(clustering_c)

class Network:
    def __init__(self, adjacency):
        self.adjacency = adjacency
        self.edges = adjacency[adjacency > 0].fillna(0)
        self.indegrees = self.edges.sum()
        self.outdegrees = self.edges.sum(1)
        self.edge_count = self.edges.sum().sum()

    def __repr__(self):
        return str(self.adjacency)


def _brute_force(adjacency, hessian):
    """
    Use brute force to test for every combination of two clusters
    using the function specified by `hessian` as the objective function
    """
    a = adjacency.adjacency
    n = len(a)
    names = list(a.index)
    binary = [format(x, '#0%ib' % (n + 2))[2:] for x in range(pow(2, n) / 2)]
    community_lists = [pd.Series(list(x), index=names).astype(np.int64) 
        for x in binary] 
    objective = pd.Series(
        {str(c.values):hessian(adjacency, c.values) for c in community_lists})
    return objective.order(ascending=False)

def reichardt_bornholdt(network, **kwargs):
    """
    Produce a clustering of the network represented by the adjacency
    matrix `adjacency`.
    
    Parameters
    ----------
    network: clustering_c.Network
        A Network object representing an adjacency matrix
        
    Returns
    -------
    A Series representing the community membership
    """
    communities = clustering_c.cluster(network, **kwargs)
    communities = pd.DataFrame(communities, columns=['community'])
    communities['node'] = network.adjacency.index
    return communities
    
if __name__ == "__main__":
    # A network with two very obvious communities:
    # A, B, C, D and E, F, G, H. A single connection
    # from E to D joins the two communities
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
    a = Network(adj)
#    modularity = _brute_force(a, clustering_c.modularity)
#    potts = _brute_force(a, clustering_c.reichardt_bornholdt)
#    clustering_c.test()
    clustering_c.cluster(a)
    