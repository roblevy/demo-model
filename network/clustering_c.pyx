# encoding: utf-8
# cython: profile=False
# filename: clustering_c.pyx
"""
A series of clustering algorithms for use on an adjacency matrix which is
stored in the form of a pandas DataFrame.
"""

cimport numpy as np
import numpy as np
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport floor, exp
import cython # Need this for decorators

# Global variables
cdef:
    double[:, :] adjacency # Memory view
    long[:] community_labels
    double[:] indegrees
    double[:] outdegrees
    double total_edge_count
    long group_count
    long node_count
    double[:] prob_node_in_group

@cython.cdivision(True) #  Need this to stop Cython checking for div by zero
cdef double rand_float() :
    """
    A random number between 0 and 1
    """
    cdef double x = rand()
    return x / <double>RAND_MAX
        
cdef int rand_int(int n) :
    """
    Return a random integer between 0 and (n - 1)
    """
    return <int>floor(rand_float() * n)

@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef int _weighted_random_int(double[:] weights) :
    """
    A random number between 1 and len(weights) with each
    integer, i, occuring with weight weights[i]
    """
    cdef:
        double previous_prob = 0
        double current_prob = 0
        double weight
        double random = rand_float() # Number bewteen 0 and 1
        long n = weights.shape[0]
    for i in range(n):
        current_prob = previous_prob + weights[i]
        if random > previous_prob and random <= current_prob:
            return i
        previous_prob = current_prob
    return -1

cdef double _null_model(double indegree_i, double outdegree_j, 
                        int total_edges):
    """
    Probability that edge ij exists, given in-degree of i and
    out-degree of j
    """
    # TODO: Fix this
    return indegree_i * outdegree_j / total_edges

cdef double _modularity_ij(double adjacency_ij, double indegree_i, 
                           double outdegree_j, 
                           int edge_count,
                           int community_i, int community_j):
    """
    c-optimised calculation of the ijth element of modularity
    """
    # TODO: Fix this to use globals
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

cdef double _cohesion_ss(long s, double gamma):
    """
    Cohesion of group s
    
    Defined as :math:`m_{ss} - [m_{ss}_{p_{ij}}]`
    where :math:`m_{rs}` is the number of edges between
    group r and group s
    """
    return _adhesion_rs(r=s, s=s, gamma=gamma)

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef double _adhesion_rs(long r, long s, double gamma):
    """
    Adhesion of group r with group s
    
    Defined as :math:`m_{rs} - [m_{rs}_{p_{ij}}]`
    where :math:`m_{rs}` is the number of edges between
    group r and group s
    """
    cdef:
        double adhesion = 0
    for i in range(node_count):
        # Since adhesion is additive can just calculate
        # adhesion_is for all i in r
        if community_labels[i] == r:
            # The i node is in group r
            adhesion += _adhesion_ls(l=i, s=s, gamma=gamma)
    return adhesion

@cython.boundscheck(False)
@cython.cdivision(True) #  Need this to stop Cython checking for div by zero
cdef double _adhesion_ls(long l, long s, double gamma)  except -1.0:
    """
    Adhesion between node l and group s
    """
    cdef:
        double m_ls = 0
        double kin_s = 0
        double kout_s = 0
        double sum_degree_s = 0 # k_i^out / (m - k_i^in)
        double[:] outd = outdegrees
        double[:] ind = indegrees
        double[:, :] a = adjacency
        long[:] c = community_labels
        double kout_l = outd[l]
        double kin_l = ind[l]
        double m = total_edge_count
    for i in range(node_count):
        if c[i] == s:
            kin_s += ind[i]
            kout_s += outd[i]
            #sum_degree_s += outd[i] / (m - ind[i])
            if i != l:
                # Node i is in group s (and is NOT node l!)
                if a[l, i] > 0:
                    # Edge l -> i exists
                    m_ls += a[l, i]
                if a[i, l] > 0:
                    # Edge i -> l exists
                    m_ls += a[i, l]
    #print "s %s c %s" % (s, np.array(c))
    #print "ind %s" % np.array(ind)
    #print "kin_s %s m_ls %s kout_l %s kin_l %s sum_degree_s %s" % (kin_s, m_ls, kout_l, kin_l, sum_degree_s)

    # Original formula all the analysis was based on:
    #return m_ls - (kin_s * kout_l / (m - kin_l)
        #+ kin_l * sum_degree_s) * gamma / m

    # Naive undirected formula directly out of Reichardt and Bornholdt
    # Note that if the adjacency matrix is symmetric, kin_s
    # is the same as kout_s etc:
    #return m_ls - (gamma / (2 * m) * kin_l * kin_s)
    return m_ls - gamma * (kout_l * kin_s + kin_l * kout_s) / (m * m)

@cython.boundscheck(False)
@cython.cdivision(True)
cdef int _l_update_probabilities(long l, double t,
    long l_current_group, double[:] p, double gamma)  except -1:
    """
    An array of probablities for node l to move to each group
    at temperature t
    
    See eqns 23 and 29 in Reichardt and Bornholdt
    """
    cdef:
        double l_current_adhesion = _adhesion_ls(l=l, s=l_current_group,
            gamma=gamma)
        double prob_all_groups = 0
        float delta_H_s # Change in energy moving l to group s
        
    for s in range(group_count):
        if s == l_current_group:
            # Node l is already in group s
            delta_H_s = 0
        else:
            # Node l not already in group s
            delta_H_s = l_current_adhesion - _adhesion_ls(l=l, s=s, 
                gamma=gamma)
        p[s] = exp((-1 / t) * delta_H_s)
        prob_all_groups += p[s]
    # Divide each element of prob_node_in_group by the total probability
    for i in range(group_count):
        p[i] = p[i] / prob_all_groups
    return 1
                    
cdef long[:] _randomise_community_membership(long n, long g):
    cdef long[:] groups = np.empty(n, dtype=long)
    for i in range(n):
        groups[i] = rand_int(g)
    return groups

cdef long[:] _push_value(long[:] in_array, value):
    cdef:
        long n = len(in_array)
        long[:] out_array = np.ndarray(n, dtype=long)
        long i
    out_array[n - 1] = value
    for i in range(n - 1):
        out_array[i] = in_array[i + 1]
    return out_array

cdef long _sum_array(long[:] in_array):
    result = 0
    for i in range(len(in_array)):
        result += in_array[i]
    return result

@cython.boundscheck(False)
cdef int cluster_simulated_annealing(double start_t, double end_t, 
                                     double t_step,
                                     double gamma)  except -1:
    cdef:
        long node_count = adjacency.shape[0]
        long loops_at_current_t = node_count * 50
        long node, new_group
        long node_group
        double t = start_t
        double[:] update_probabilities
        double[:] p = prob_node_in_group
        long[:] c = community_labels

    while t > end_t:
        for i in range(loops_at_current_t):
            # Pick a random node
            node = rand_int(node_count)
            node_group = c[node] 
            _l_update_probabilities(l=node, t=t, l_current_group=node_group, 
                                    p=p, gamma=gamma)
            new_group = _weighted_random_int(p)
            if new_group < 0:
                return -1
            c[node] = new_group
        t *= t_step
    return 1
        
cpdef set_globals(network, c=None, int g=25):
    global adjacency, indegrees, outdegrees, total_edge_count
    global community_labels, node_count, group_count, prob_node_in_group
    # Adjacency matrix, a
    adjacency = network.adjacency.values
    # node count
    node_count = adjacency.shape[0]
    # in- and out-degreee vectors
    indegrees = network.indegrees.values
    outdegrees = network.outdegrees.values
    # total edge count, m
    total_edge_count = network.edge_count
    # group count
    group_count = g
    # community labels, c
    if c is None:
        community_labels = _randomise_community_membership(node_count, group_count)
    else:
        community_labels = c
    # probability vector for a node going into a group
    prob_node_in_group = np.ndarray(group_count)

def modularity(adjacency, c):
     """
     Given an Adjacency object, and a pd.Series specifying
     community labels for each element of the adjacency matrix,
     return the Leicht/Newman modularity
     """
     global community_labels
     set_globals(adjacency, c=c)
     cdef:
         np.ndarray[double,  ndim=2] a = adjacency.adjacency.values
         Py_ssize_t i, j, n = len(a)
         int k = 0
         np.ndarray[double] k_in = adjacency.indegrees.values
         np.ndarray[double] k_out = adjacency.outdegrees.values
         np.ndarray[double] q = np.empty(n * n)
         int m = adjacency.edge_count
     for i in range(n):
         for j in range(n):
             q[k] = _modularity_ij(a[i,j], k_in[i], k_out[j], m, c[i], c[j])
             k += 1
     return sum(q) / m

cpdef cluster(network, group_count=25, start_t=1, end_t=1e-4, t_step=0.9,
            gamma=1):
    cdef double t0 = start_t
    cdef double t_end = end_t
    cdef double tstep = t_step
    cdef double c_gamma = gamma
    cluster_simulated_annealing(start_t=t0, end_t=t_end, t_step=tstep,
                                gamma=c_gamma)
    #print "Final results:"
    #print np.array(community_labels)
    return community_labels
    
def test():
    pass
#    cdef long[:] x = np.array([1,2,3,4])
#    x = _push_value(x, 5)
#    print np.array(x)
#    print _sum_array(x)
