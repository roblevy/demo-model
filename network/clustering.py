"""
A series of clustering algorithms for use on an adjacency matrix which is
stored in the form of a pandas DataFrame.
"""

import pandas as pd

class Adjacency():
    def __init__(self, adjacency):
        self.adjacency = adjacency
        self.edges = adjacency[adjacency > 0]
        self.indegrees = self.edges.sum()
        self.outdegrees = self.edges.sum(1)
        self.num_of_edges = self.edges.sum().sum()

    def __repr__(self):
        return str(self.adjacency)
        

def _modularity_ij(adjacency_ij, indegree_i, outdegree_j, 
                  num_of_edges,
                  community_i, community_j):
    a_ij = adjacency_ij
    k_i = indegree_i
    k_j = outdegree_j
    m = num_of_edges
    delta_ci_cj = 1 * community_i == community_j
    return (a_ij - (k_i * k_j) / m) * delta_ci_cj

def _modularity_given_communities(adjacency, community_labels):
    """
    Given an Adjacency object, and a pd.Series specifying
    community labels for each element of the adjacency matrix,
    return Q, the modularity.
    """
    a = adjacency.adjacency
    k_in = adjacency.indegrees
    k_out = adjacency.outdegrees
    c = community_labels
    m = adjacency.num_of_edges    
    q = [_modularity_ij(
            a.ix[i, j], k_in.ix[i], k_out.ix[j], m, c.ix[i], c.ix[j])
         for i in a.index for j in a.columns]
    return sum(q) / (2 * m)

def _brute_force_modularity(adjacency):
    a = adjacency.adjacency
    n = len(a)
    names = list(a.index)
    binary = [format(x, '#0%ib' % (n + 2))[2:] for x in range(pow(2, n) / 2)]
    community_lists = [pd.Series(list(x), index=names) for x in binary] 
    modularity = pd.Series({str(c.values):_modularity_given_communities(adjacency, c)
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
    adj = pd.DataFrame(rows)
    adj.index = names
    adj.columns = names
    a = Adjacency(adj)
    test_communities = pd.Series([0,0,0,0,0,0,0,0], index=names)
    results = _brute_force_modularity(a).order()
    