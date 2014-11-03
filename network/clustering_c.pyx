"""
A series of clustering algorithms for use on an adjacency matrix which is
stored in the form of a pandas DataFrame.
"""

cimport numpy as np
import numpy as np

class Adjacency():
    def __init__(self, adjacency):
        self.adjacency = adjacency
        self.edges = adjacency[adjacency > 0]
        self.indegrees = self.edges.sum()
        self.outdegrees = self.edges.sum(1)
        self.num_of_edges = self.edges.sum().sum()

    def __repr__(self):
        return str(self.adjacency)
        

def _modularity_ij(double adjacency_ij, int indegree_i, int outdegree_j, 
                  int num_of_edges,
                  int community_i, int community_j):
    """
    c-optimised calculation of the ijth element of modularity
    """
    cdef double a_ij = adjacency_ij
    cdef int kin_i = indegree_i
    cdef int kout_j = outdegree_j
    cdef int m = num_of_edges
    cdef int delta_ci_cj
    if community_i == community_j:
        delta_ci_cj = 1
    else:
        delta_ci_cj = 0
    return (a_ij - (kin_i * kout_j) / m) * delta_ci_cj

def _modularity_given_communities(adjacency, community_labels):
     """
     Given an Adjacency object, and a pd.Series specifying
     community labels for each element of the adjacency matrix,
     return Q, the modularity.
     """
     cdef np.ndarray[double,  ndim=2] a = adjacency.adjacency.values
     cdef Py_ssize_t i, j, n = len(a)
     cdef int k = 0
     cdef np.ndarray[double] k_in = adjacency.indegrees.values
     cdef np.ndarray[double] k_out = adjacency.outdegrees.values
     cdef np.ndarray[double] q = np.empty(n * n)
     cdef np.ndarray[long] c = community_labels.values
     cdef int m = adjacency.num_of_edges
     for i in range(n):
         for j in range(n):
             q[k] = _modularity_ij(a[i,j], k_in[i], k_out[j], m, c[i], c[j])
             k += 1
     return sum(q) / (2 * m)

