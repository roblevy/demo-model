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
        

def _null_model(int indegree_i, int outdegree_j, int num_of_edges):
    """
    Probability that edge ij exists, given in-degree of i and
    out-degree of j
    """
    return indegree_i * outdegree_j / num_of_edges

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
    return (a_ij - _null_model(kin_i, kout_j, m)) * delta_ci_cj

def modularity(adjacency, community_labels):
     """
     Given an Adjacency object, and a pd.Series specifying
     community labels for each element of the adjacency matrix,
     return the Leicht/Newman modularity
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
     return sum(q) / m

def _cohesion_ss(np.ndarray[double, ndim=2] adjacency,
                 np.ndarray[long] community_labels,
                 np.ndarray[double] indegrees,
                 np.ndarray[double] outdegrees,
                 long num_of_edges,
                 long s):
    """
    Cohesion of group s
    
    Defined as :math:`m_{ss} - [m_{ss}_{p_{ij}}]`
    where :math:`m_{rs}` is the number of edges between
    group r and group s
    """
    return _adhesion_rs(adjacency=adjacency,
                        community_labels=community_labels,
                        indegrees=indegrees,
                        outdegrees=outdegrees,
                        num_of_edges=num_of_edges,
                        r=s, s=s)

def _adhesion_rs(np.ndarray[double, ndim=2] adjacency,
                 np.ndarray[long] community_labels,
                 np.ndarray[double] indegrees,
                 np.ndarray[double] outdegrees,
                 long num_of_edges,
                 long r, long s):
    """
    Adhesion of group r with group s
    
    Defined as :math:`m_{rs} - [m_{rs}_{p_{ij}}]`
    where :math:`m_{rs}` is the number of edges between
    group r and group s
    """
    cdef int i, j, n = len(community_labels)
    cdef long edge_count = 0
    cdef double total_indegree = 0
    cdef double total_outdegree = 0
    cdef double expectation
    for i in range(n):
        if community_labels[i] == r:
            # The i node is in group r
            total_outdegree += outdegrees[i]        
            for j in range(n):
                if community_labels[j] == s:
                    # The j node is in group s
                    if adjacency[i, j] > 0:
                        # An i-j edge exists
                        edge_count += 1
        if community_labels[i] == s:
            # The i node is in group s
            total_indegree += indegrees[i]
    expectation = total_indegree * total_outdegree / num_of_edges
    return edge_count - expectation

def reichardt_bornholdt(adjacency, community_labels):
     """
     Given an Adjacency object, and a pd.Series specifying
     community labels for each element of the adjacency matrix,
     return the Reichardt/Bornholdt 
     """
     cdef np.ndarray[double,  ndim=2] a = adjacency.adjacency.values
     cdef Py_ssize_t i, j, n = len(a)
     cdef int k = 0
     cdef np.ndarray[double] k_in = adjacency.indegrees.values
     cdef np.ndarray[double] k_out = adjacency.outdegrees.values
     cdef double h = 0
     cdef np.ndarray[long] c = community_labels.values
     cdef int m = adjacency.num_of_edges
     for s in range(c.max()):
         h += _cohesion_ss(a, c, k_in, k_out, m, s)
     return h