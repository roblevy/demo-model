"""
A series of clustering algorithms for use on an adjacency matrix which is
stored in the form of a pandas DataFrame.
"""

cimport numpy as np
import numpy as np
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport floor, exp
import cython # Need this for decorators

class Network:
    def __init__(self, adjacency):
        self.adjacency = adjacency
        self.edges = adjacency[adjacency > 0]
        self.indegrees = self.edges.sum()
        self.outdegrees = self.edges.sum(1)
        self.edge_count = self.edges.sum().sum()

    def __repr__(self):
        return str(self.adjacency)

# Global variables
cdef:
    double[:, :] adjacency # Memory view
    long[:] community_labels
    double[:] indegrees
    double[:] outdegrees
    double total_edge_count
    long group_count
    long node_count

@cython.cdivision(True) #  Need this to stop Cython checking for div by zero     
cdef double rand_float():
    """
    A random number between 0 and 1
    """
    cdef double x = rand()
    return x / <double>RAND_MAX
        
cdef int rand_int(int n):
    """
    Return a random integer between 0 and (n - 1)
    """
    return <int>floor(rand_float() * n)

@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef int _weighted_random_int(double[:] weights):
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

cdef double _cohesion_ss(long s):
    """
    Cohesion of group s
    
    Defined as :math:`m_{ss} - [m_{ss}_{p_{ij}}]`
    where :math:`m_{rs}` is the number of edges between
    group r and group s
    """
    return _adhesion_rs(r=s, s=s)

@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef double _adhesion_rs(long r, long s):
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
            adhesion += _adhesion_ls(l=i, s=s)
    return adhesion

@cython.boundscheck(False)
@cython.cdivision(True) #  Need this to stop Cython checking for div by zero
cdef double _adhesion_ls(long l, long s):
    """
    Adhesion between node l and group s
    """
    cdef:
        double m_ls = 0
        double kin_s = 0
        double sum_degree_s = 0 # k_i^out / (m - k_i^in)
        double[:] outd = outdegrees
        double[:] ind = indegrees
        double[:, :] a = adjacency
        long[:] c = community_labels
        double kout_l = outd[l]
        double kin_l = ind[l]
        double m = total_edge_count
    for i in range(node_count):
        if i != l:
            if c[i] == s:
                # Node i is in group s (and is NOT node l!)
                kin_s += ind[i]
                if abs(m - ind[i]) > 0:
                    sum_degree_s += outd[i] / (m - ind[i])
                if a[l, i] > 0:
                    # Edge l -> i exists
                    m_ls += a[l, i]
                if a[i, l] > 0:
                    # Edge i -> l exists
                    m_ls += a[i, l]

    return m_ls - (kin_s * kout_l / (m - kin_l) + kin_l * sum_degree_s) / m

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:] _l_update_probabilities(long l, double t):
    """
    An array of probablities for node l to move to each group
    at temperature t
    
    See eqns 23 and 29 in Reichardt and Bornholdt
    """
    cdef:
        long l_current_group = community_labels[l]
        double l_current_adhesion = _adhesion_ls(l=l, s=l_current_group)
        double prob_all_groups = 0
        double[:] prob_node_in_group = np.zeros(group_count)
        float delta_H_s # Change in energy moving l to group s
        
    for s in range(group_count):
        if community_labels[l] == s:
            # Node l is already in group s
            delta_H_s = 0
        else:
            # Node l not already in group s
            delta_H_s = l_current_adhesion - _adhesion_ls(l=l, s=s)
        if t > 0:
            prob_node_in_group[s] = exp((-1 / t) * delta_H_s)
            prob_all_groups += prob_node_in_group[s]
    # Divide each element of prob_node_in_group by the total probability
    for i in range(group_count):
        prob_node_in_group[i] = prob_node_in_group[i] / prob_all_groups
    return prob_node_in_group
                    
cdef long[:] _randomise_community_membership(long node_count, 
                                             long group_count):
    cdef long[:] groups = np.empty(node_count, dtype=long)
    for i in range(node_count):
        groups[i] = rand_int(group_count)
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

cdef int cluster_simulated_annealing(double start_t, double end_t, 
                                     double t_step = 0.99,
                                     long stop_if_no_change_after=3) except -1:
    cdef:
        long node_count = len(adjacency)
        long loops_at_current_t = len(adjacency) * 50
        long node, new_group
        double t = start_t
        double[:] update_probabilities
        long changes_this_iteration
        long[:] changes = np.ones(stop_if_no_change_after, dtype=long)
    
    while t > end_t:
        changes_this_iteration = 0        
        for i in range(loops_at_current_t):
            # Pick a random node
            node = rand_int(node_count)
            update_probabilities = _l_update_probabilities(l=node, t=t)
            new_group = _weighted_random_int(update_probabilities)
            if new_group < 0:
                print "Something's wrong: %s" % np.array(update_probabilities)
                raise ValueError
            if new_group != community_labels[node]:
                # A node has changed groups
                community_labels[node] = new_group
                changes_this_iteration += 1
        _push_value(changes, changes_this_iteration)
        if _sum_array(changes) == 0:
            print "Stopped due to no change"
            return 1
        t *= t_step
    return 1
        
def set_globals(network, c, g=25):
    global adjacency, indegrees, outdegrees, total_edge_count
    global community_labels, group_count
    # Adjacency matrix, a
    adjacency = network.adjacency.values
    # in- and out-degreee vectors
    indegrees = network.indegrees.values
    outdegrees = network.outdegrees.values
    # total edge count, m
    total_edge_count = network.edge_count
    # community labels, c
    community_labels = c
    # Node count
    node_count = len(adjacency)
    # group count
    group_count = g

def modularity(adjacency, community_labels):
     """
     Given an Adjacency object, and a pd.Series specifying
     community labels for each element of the adjacency matrix,
     return the Leicht/Newman modularity
     """
     set_globals(adjacency, community_labels)
     cdef:
         np.ndarray[double,  ndim=2] a = adjacency.adjacency.values
         Py_ssize_t i, j, n = len(a)
         int k = 0
         np.ndarray[double] k_in = adjacency.indegrees.values
         np.ndarray[double] k_out = adjacency.outdegrees.values
         np.ndarray[double] q = np.empty(n * n)
         np.ndarray[long] c = community_labels
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
    set_globals(network, community_labels)
    cdef:
        # the Hessian
        double hessian = 0
    # Calculate the Hessian
    for r in range(group_count):
        for s in range(r):
            hessian += _adhesion_rs(r=r, s=s)
    return hessian

def cluster(network, group_count=25, start_t=1, end_t=1e-4, t_step=0.99):
    cdef long n = len(network.adjacency)
    cdef long[:] c = _randomise_community_membership(n, group_count)
    print "initial groups: %s" % np.array(c)
    set_globals(network, c, group_count)
    cluster_simulated_annealing(start_t=start_t, end_t=end_t, t_step=t_step)
    print "Final results: %s" % np.array(community_labels)
    
def test():
    pass
#    cdef long[:] x = np.array([1,2,3,4])
#    x = _push_value(x, 5)
#    print np.array(x)
#    print _sum_array(x)