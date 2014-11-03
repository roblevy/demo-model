# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:46:25 2014

@author: Rob
"""

import numpy as np
import networkx as nx

def denumpify(v):
    """
    Make `v` a scalar if `v` is a numpy type. Otherwise
    leave `v` unchanged
    """
    if str(type(v)).find('numpy') > 0:
        return np.asscalar(v)
    return v
    
#%%
# Adjacency matrix to GEXF file
def write_gexf(filename, graph=None, adjacency=None, attributes=None):
    """
    Output a matrix of nodal flows to GEXF format
    """
    if graph is None:
        graph = to_graph(adjacency, attributes)
    nx.write_gexf(graph, filename, prettyprint=True)

#%%
def to_graph(adjacency, attributes, normalise=True):
    """
    Create a networkx graph object from an adjacency matrix
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(adjacency.columns)
    if normalise:
        attributes = attributes.div(attributes.max())
    # Add attributes to nodes
    for name, node in graph.nodes_iter(data=True):
        for attr_name in attributes.columns:
            to_add = attributes.ix[name,attr_name]
            node[attr_name] = denumpify(to_add)
    links = _dataframe_to_edges(adjacency, normalise=normalise)
    graph.add_edges_from(links)
    return graph
   
def filter_edges_top_n(adjacency, n, by_outflow=False):
    """
    Filter adjacency matrix to only include top `n` edge weights
    """
    return adjacency[adjacency.rank(ascending=False, 
                                    axis=(1 * by_outflow)) <= n]

def _dataframe_to_edges(adjacency, normalise=True):
    """
    Create a container of edges from an adjacency matrix
    
    graph.add_edges_from() requires a list of 3-tuples, the
    first and second elements of which are source and destination
    node lables, and the third is a dictionary of attributes
    
    Parameters
    ----------
    adjacency: pandas.core.frame.DataFrame
        A dataframe with rows and columns being nodes and
        values being edge weights
    """
    A = adjacency
    # Remove all links <= 0
    links = A.stack()[A.stack() > 0]
    if normalise:
        links = links / links.abs().max()
    return [(k[0], k[1], {'weight':denumpify(v)}) 
        for k, v in links.iteritems()]