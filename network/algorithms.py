# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:17:06 2014

@author: rob

A variety of network algorithms to be run on networkx networks but which
are not already implemented by the networkx package
"""

import numpy as np
import networkx as nx
import demo_model.network.io as network_io

#%%
## Testing only
G = nx.DiGraph()
G.add_node("A", {'size':10})
G.add_node("B", {'size':5})
G.add_node("C", {'size':8})
G.add_node("D", {'size':9})
G.add_edge("A", "C", {'weight':2})
G.add_edge("A", "D", {'weight':8})
G.add_edge("C", "A", {'weight':10})
G.add_edge("C", "B", {'weight':4})
G.add_edge("C", "D", {'weight':3})
G.add_edge("D", "B", {'weight':0.9})
G.add_edge("D", "B", {'weight':1})
G.add_edge("D", "C", {'weight':0.5})

#%%
def edmonds(graph, root_key=None):
    """
    Implement Edmonds' arborescence algorithm for calculating the
    minimum spanning tree of a directed graph.
    
    The resulting MST is defined for a particular 'root' node, the
    key of which is specified by `root_key`. If none is specified
    the root is chosen by highest total weighted degree.
    
    Parameters
    ----------
    graph: networkx.classes.digraph.DiGraph
        A weighted-directed graph
        
    root_key: list optional
        The dictionary key related to the node to be used as the root of the
        MST.
    """
    G = nx.DiGraph(graph)
    G1 = nx.DiGraph()
    G1.add_nodes_from(G.nodes(data=True))
    if root_key is None:
        nx.set_node_attributes(G1, 'degree', G.degree(weight='weight'))
        root_key = max_node(G1, 'degree')
    # Add all root edges:
    G1.add_edges_from(G.edges(root_key, data=True))
    for n in [n for n in G1.nodes() if n != root_key]:
        while True:
            top_e = max_edge(G, 'weight', G.edges(n, data=True))
            if top_e is not None:
                G1.add_edges_from([top_e])
                if len(list(nx.simple_cycles(G1))) > 0:
                    # The most recent edge has made a loop. Remove it
                    G1.remove_edges_from([top_e])
                    G.remove_edges_from([top_e])
                else:
                    # Most recent edge has not made a loop
                    break
            else:
                break
    return G1

#%%
# Nodal flow
def nodal_flow(X, sizes, find_sources=True):
    """
    calculate the Nystuen & Dacy nodal flow matrix of flow matrix X,
    sizing each node using `sizes`.

    Take a matrix and return a matrix with only the largest value
    remaining in each column.
    """
    # Remove all negative entries
    X = X[X > 0].fillna(0)
    # remove all rows and columns whose column sums are zero:
    X = X.ix[X.sum(0) > 0, X.sum(0) > 0]
    if not X.empty:
        # Adjust raw-data matrix to proportion of largest total association
        # (p36 of the paper)
        max_col_total = X.sum(0).max()
        Y = X/max_col_total
        # Computation of the power series of the adjacency matrix (p38)
        I = Y * 0 + np.eye(Y.shape[0])
        B = np.linalg.inv(I - Y) - I
        if find_sources:
            B = B.transpose()
        nodal_B = B * 0
        largest_flows = X.idxmax(1)
        for row_name, column_largest_flow in largest_flows.iteritems():
            nodal_B.loc[row_name, largest_flows[row_name]] \
                = (1 * (sizes[row_name] < sizes[column_largest_flow]))
        return nodal_B
    else:
        return X

def max_node(G, attrib_name, nodes=None):
    """
    Get the node in `nodes` (or in all nodes of G)
    with the highest value of attrib_name.
    
    In the case of a tie, a node is returned defined by the
    behaviour of Python's `max()` function.
    
    Parameters
    ----------
    G: nx.Graph (or DiGraph etc.)
        the graph to work on
    attrib_name: string
        Which attribute to max over
    nodes: list optional
        a subset of nodes, such as returned by G.nodes() 
        or G['foo'].neighbours
    
    """
    if nodes is None:
        nodes = G.nodes()
    n = max(nodes, 
            key=lambda(n): G.node[n][attrib_name])
    return n

def max_edge(G, attrib_name, edges=None):
    """
    Get the edge in `edges` (or in all edges of G) 
    with the highest value of attrib_name.
    
    In the case of a tie, an edge is returned defined by the
    behaviour of Python's `max()` function.
    
    Parameters
    ----------
    G: nx.Graph (or DiGraph etc.)
        the graph to work on
    attrib_name: string
        Which attribute to max over
    edges: list optional
        a subset of edges, such as returned by G.edges(data=True) 
        or G.edges('some_node', data=True)
    """
    if edges is None:
        edges = G.edges()
    if len(edges) > 0:
        return  max(edges,
                    key=lambda(e): G.edge[e[0]][e[1]][attrib_name])
    else:
        return None
    
# test_G = edmonds(G)
# print test_G.edges(data=True)
# Expect:
#C-4->B 
#^    ^
#2    1
#|    |
#A-8->D
