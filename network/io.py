# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:46:25 2014

@author: Rob
"""

import numpy as np
import networkx as nx


    
#%%
# Adjacency matrix to GEXF file
def to_gexf(adjacency, attributes, filename):
    """
    Output a matrix of nodal flows to GEXF format
    """
    graph = to_graph(adjacency, attributes)
    nx.write_gexf(graph, filename, prettyprint=True)

#%%
def to_graph(adjacency, attributes):
    """
    Create a networkx graph object from an adjacency matrix
    """
    A = adjacency
    attr = attributes
    links = A.stack()[A.stack() > 0].index.values
    graph = nx.DiGraph()
    graph.add_nodes_from(A.columns)
    # Add attributes to nodes
    for name, node in graph.nodes_iter(data=True):
        for attr_name in attributes.columns:
            to_add = attr.ix[name,attr_name]
            if str(type(to_add)).find('numpy') > 0:
                to_add = np.asscalar(to_add)
            node[attr_name] = to_add
    graph.add_edges_from(links)
    return graph
    
