"""
A series of clustering algorithms for use on an adjacency matrix which is
stored in the form of a pandas DataFrame.
"""

import numpy as np
import pandas as pd
#import pyximport; pyximport.install(reload_support=True)
import pyximport; pyximport.install()
import igraph
import clustering_c
#reload(clustering_c)
import multiprocessing as mp
from functools import partial

class Network:
    def __init__(self, adjacency, make_symmetric=False):
        if make_symmetric:
            adjacency = (adjacency + adjacency.transpose()) / 2
        self.adjacency = adjacency
        self.edges = adjacency[adjacency > 0].fillna(0)
        self.indegrees = self.edges.sum()
        self.outdegrees = self.edges.sum(1)
        self.edge_count = self.edges.sum().sum()

    def __repr__(self):
        return str(self.adjacency)

def _igraph_from_dataframe(df):
    """
    Convert an adjency matrix stored as a `pandas.DataFrame`
    into an igraph Graph
    """
    if not np.alltrue(df.columns == df.index):
        raise NameError('Column and index labels must be identical')
    vals = df.values
    g = igraph.Graph.Adjacency((vals > 0).tolist())
    g.es['weight'] = vals[vals.nonzero()]
    g.vs['label'] = df.columns.tolist()
    return g    

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

def repeated_clustering(gamma, network, iterations):
    for i in range(iterations):
        try:
            res['iter%i' % i] = cluster(network, gamma).set_index('node')
        except NameError:
            res = cluster(network, gamma).set_index('node')
            res = res.rename(columns={'community':'iter%i' % i})
    res['gamma'] = round(gamma, 3)
    return res.reset_index()
    
def gamma_sweep(network, iterations_per_gamma=1, gamma_range=None):
    if gamma_range is None:
        gamma_range = np.arange(0.1, 0.7, 0.01)
    pool = mp.Pool()
    _repeated_cluster = partial(repeated_clustering, 
                      network=network, iterations=iterations_per_gamma)
    results = pool.map(_repeated_cluster, gamma_range)
    pool.close()
    pool.join()
    return pd.concat(results)

def clustering_single_gamma(network, gamma, iterations):
    return cluster_co_occurence(reichardt_bornholdt, 
                                network=network, gamma=gamma)    

def _cluster_iteration(i, cluster_function, adjacency, **kwargs):
    return {i: cluster_function(adjacency, **kwargs)}
    
def cluster_co_occurence(iterations, cluster_function, adjacency, **kwargs):
    res = {}
    pool = mp.Pool()
    _cluster_fn = partial(_cluster_iteration, 
                          cluster_function=cluster_function,
                          adjacency=adjacency, **kwargs)
    res = pool.map(_cluster_fn, range(iterations))
    co_occurences = [co_occurence(x.values()[0]) for x in res]
    for x in co_occurences:
        try:
            summed += x
        except NameError:
            summed = x
    return co_occurences, summed / iterations
    
def smallest_community(df):
    return df.apply(lambda col: col.groupby(col).count().min())

def _co_occurence(col, membership):
    return (membership == membership.ix[col.name]).sum(axis=1)

def co_occurence(membership):
    """
    Returns a DataFrame with the number of times the row node
    appears in the same cluster as the column node
    """
    m = membership
    n = len(m)
    try:
        m = m.set_index('node')
    except:
        pass    
    matrix = np.zeros([n, n])
    matrix = pd.DataFrame(matrix, columns=m.index, index=m.index)
    return matrix.apply(_co_occurence, membership=m)

def reichardt_bornholdt(network, t_step=0.9, **kwargs):
    """
    Produce a clustering of the network represented by the adjacency
    matrix `adjacency`.
    
    Parameters
    ----------
    network: clustering.Network
        A Network object representing an adjacency matrix
        
    Returns
    -------
    A Series representing the community membership
    """
    clustering_c.set_globals(network, g=25)
    communities = clustering_c.cluster(network, **kwargs)
    communities = pd.DataFrame(np.array(communities), columns=['community'])
    communities['node'] = network.adjacency.index
    return communities

def infomap(df, **kwargs):
    """
    Cluster the adjacency matrix, stored as a `pandas.DataFrame`
    using Rosvall & Bergstrom's infomap method
    """
    g = _igraph_from_dataframe(df)
    member = g.community_infomap(**kwargs).membership
    res = pd.DataFrame()
    res['community'] = member
    res['node'] = g.vs['label']
    return res
    
if __name__ == "__main__":
    # A network with two very obvious communities:
    # A, B, C, D and E, F, G, H. A single connection
    # from E to D joins the two communities
    rows = [[0,1,1,0,0,0,0,0],
            [1,0,1,1,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [1,0,1,0,0,0,0,0],
            [0,0,0,1,0,1,1,0],
            [0,0,0,1,1,0,0,1],#[0,0,0,0,1,0,0,1],
            [0,0,0,0,0,0,0,1],
            [0,0,0,0,1,1,0,0],
            ]
    names = ["A", "B", "C", "D", "E", "F", "G", "H"]
    adj = pd.DataFrame(rows).astype(float)
    adj.index = names
    adj.columns = names
    a = Network(adj, make_symmetric=True)
#    modularity = _brute_force(a, clustering_c.modularity)
#    clustering_c.test()
    clusters = reichardt_bornholdt(a, gamma=10)
    print clusters.groupby('community').aggregate(';'.join)
    

    
    
