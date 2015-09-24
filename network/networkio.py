# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:46:25 2014

@author: Rob
"""
import pandas as pd
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
def to_graph(adjacency, attributes=None, normalise=True, positive_edges=True):
    """
    Create a networkx graph object from an adjacency matrix
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(adjacency.columns)
    if attributes is not None:
        if normalise:
            attributes = attributes.div(attributes.max())
        # Add attributes to nodes
        for name, node in graph.nodes_iter(data=True):
            for attr_name in attributes.columns:
                to_add = attributes.ix[name,attr_name]
                node[attr_name] = denumpify(to_add)
    links = _dataframe_to_edges(adjacency, 
        normalise=normalise, positive_edges=positive_edges)
    graph.add_edges_from(links)
    return graph

def to_pajek(adjacency, filename, min_flow=1e-3):
    """
    create a .net file according to the Pajek specification
    
    See http://www.mapequation.org/apps/MapGenerator.html#fileformats
    for details.
    """
    if filename[-4:] != '.net':
        filename += '.net'
    adj = adjacency
    flows = adj.stack().reset_index()
    flows = flows[flows > min_flow].dropna(how='any')
    nodes = {k:i + 1 for i, k in enumerate(adj.columns)}
    from_col = _get_col_containing(flows, 'from')
    to_col = _get_col_containing(flows, 'to')
    flows['from_number'] = flows[from_col].replace(nodes)
    flows['to_number'] = flows[to_col].replace(nodes)
    # Now build the output file
    out = '*Vertices %i\n' % len(nodes)
    out += '\n'.join(['%i "%s"' % (v, k) for k, v in sorted(nodes.iteritems())])
    out += '\n*arcs %i\n' % len(flows)
    out += flows.to_string(columns=['from_number', 'to_number', 0],
                           header=False, index=False)
    with open(filename, 'w') as text_file:
        text_file.write(out)
    print '%s written' % filename
   
def filter_edges_top_n(adjacency, n, by_outflow=False):
    """
    Filter adjacency matrix to only include top `n` edge weights
    """
    return adjacency[adjacency.rank(ascending=False, 
                                    axis=(1 * by_outflow)) <= n]

def _get_col_containing(df, containing):
    try:
        return [c for c in df.columns if containing in str(c)][0]
    except IndexError:
        return ""

def _dataframe_to_edges(adjacency, normalise=True, positive_edges=True):
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
    if positive_edges:
        # Remove all links <= 0
        links = A.stack()[A.stack() > 0]
    if normalise:
        links = links / links.abs().max()
    return [(k[0], k[1], {'weight':denumpify(v)}) 
        for k, v in links.iteritems()]