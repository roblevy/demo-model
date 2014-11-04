"""
A series of clustering algorithms for use on an adjacency matrix which is
stored in the form of a pandas DataFrame.
"""

cimport numpy as np
import numpy as np
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport floor

class Network:
    def __init__(self, adjacency):
        self.adjacency = adjacency
        self.edges = adjacency[adjacency > 0]
        self.indegrees = self.edges.sum()
        self.outdegrees = self.edges.sum(1)
        self.edge_count = self.edges.sum().sum()

    def __repr__(self):
        return str(self.adjacency)
        
cdef double rand_float():
    """
    A random number between 0 and 1
    """
    return rand() / <double>RAND_MAX
        
cdef int rand_int(int n):
    """
    Return a random integer between 0 and (n - 1)
    """
    return <int>floor(rand_float() * n)

cdef double _null_model(double indegree_i, double outdegree_j, 
                        int edge_count):
    """
    Probability that edge ij exists, given in-degree of i and
    out-degree of j
    """
    return indegree_i * outdegree_j / edge_count

cdef double _modularity_ij(double adjacency_ij, double indegree_i, 
                           double outdegree_j, 
                           int edge_count,
                           int community_i, int community_j):
    """
    c-optimised calculation of the ijth element of modularity
    """
    cdef:
        double a_ij = adjacency_ij
        double kin_i = indegree_i
        double kout_j = outdegree_j
        int m = edge_count
        int delta_ci_cj
    if community_i == community_j:
        delta_ci_cj = 1
    else:
        delta_ci_cj = 0
    return (a_ij - _null_model(kin_i, kout_j, m)) * delta_ci_cj

cdef double _cohesion_ss(np.ndarray[double, ndim=2] adjacency,
                 np.ndarray[long] community_labels,
                 np.ndarray[double] indegrees,
                 np.ndarray[double] outdegrees,
                 long edge_count,
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
                        r=s, s=s)

cdef double _adhesion_rs(np.ndarray[double, ndim=2] adjacency,
                 np.ndarray[long] community_labels,
                 np.ndarray[double] indegrees,
                 np.ndarray[double] outdegrees,
                 long r, long s):
    """
    Adhesion of group r with group s
    
    Defined as :math:`m_{rs} - [m_{rs}_{p_{ij}}]`
    where :math:`m_{rs}` is the number of edges between
    group r and group s
    """
    cdef:
        int n = len(community_labels)
        double edge_count = 0
        double total_indegree = 0
        double total_outdegree = 0
        double expectation
    for i in range(n):
        if community_labels[i] == r:
            # The i node is in group r
            total_outdegree += outdegrees[i] 
            for j in range(n):
                if community_labels[j] == s:
                    # The j node is in group s
                    if adjacency[i, j] > 0:
                        # An i-j edge exists (from r to s)
                        edge_count += 1
        if community_labels[i] == s:
            # The i node is in group s
            total_indegree += indegrees[i]
    expectation = total_indegree * total_outdegree / edge_count
    return edge_count - expectation

def modularity(adjacency, community_labels):
     """
     Given an Adjacency object, and a pd.Series specifying
     community labels for each element of the adjacency matrix,
     return the Leicht/Newman modularity
     """
     cdef:
         np.ndarray[double,  ndim=2] a = adjacency.adjacency.values
         Py_ssize_t i, j, n = len(a)
         int k = 0
         np.ndarray[double] k_in = adjacency.indegrees.values
         np.ndarray[double] k_out = adjacency.outdegrees.values
         np.ndarray[double] q = np.empty(n * n)
         np.ndarray[long] c = community_labels.values
         int m = adjacency.edge_count
     for i in range(n):
         for j in range(n):
             q[k] = _modularity_ij(a[i,j], k_in[i], k_out[j], m, c[i], c[j])
             k += 1
     return sum(q) / m

def reichardt_bornholdt(network, community_labels):
    """
    Given a Network object, and a pd.Series specifying
    community labels for each element of the adjacency matrix,
    return the Reichardt/Bornholdt hessian.
    
    This boils down to summing the adhesion between all groups.
    """
    cdef:
        # Adjacency matrix, a
        np.ndarray[double,  ndim=2] a = network.adjacency.values
        # in- and out-degreee vectors
        np.ndarray[double] k_in = network.indegrees.values
        np.ndarray[double] k_out = network.outdegrees.values
        # total edge count, m
        int m = network.edge_count
        # community labels, c
        np.ndarray[long] c = community_labels.values
        # the Hessian
        double hessian = 0
        # number of groups, q
        int q = c.max() + 1
        # Calculate the Hessian
    for r in range(q):
        for s in range(r):
            hessian += _adhesion_rs(a, c, k_in, k_out, r, s)
    return hessian
