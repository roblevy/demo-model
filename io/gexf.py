import itertools
import pandas as pd
import numpy as np
_node_ids = None
_attributes = None
_attr_ids = None
_id = -1
_attribute_id = 0
_sector_aggregation = None

def flows_to_gexf(model,
        cs_to_cs=None, c_to_cs=None, countries=None,
        sectors=None, sector_aggregation=None, filter_edges_less_than=0.1):
    """
    Convert an ENFOLDing model object or a set of cs/cs and c/cs flows to GEXF format
    for display in e.g. Gephi

    cs_to_cs : Series
        country/sector -> country/sector flows, indexed as sector|from_country|to_country
    c_to_cs : Series
        country -> country/sector flows, indexed as country|sector (or sector|country)
    sector_aggregation : dict
        Dictionary keyed on sector names with values of aggregated sector names
    """
    if countries is None:
        countries = model.country_names
    if sectors is None:
        sectors = model.sectors
    if cs_to_cs is None:
        cs_to_cs = _trade_flows(model, countries, sectors, sector_aggregation)
    if c_to_cs is None:
        c_to_cs = _production_flows(model, countries, sectors, sector_aggregation)
    _set_globals(countries, sectors, sector_aggregation)
    nodes = _set_nodes(countries, sectors, sector_aggregation)
    gexf_nodes = _gexf_nodes_with_attvalues(nodes, model)
    edges = _edges(cs_to_cs, c_to_cs, countries, sectors, sector_aggregation)
    filtered_edges = _filter_edges(edges, filter_edges_less_than)
    gexf_edges = _gexf_edges_with_attvalues(filtered_edges, model)
    gexf_attributes = _gexf_attributes(_attributes)
    return _gexf_graph(gexf_nodes, gexf_edges, gexf_attributes)
    
def _filter_edges(edges, less_than):
    """
    Remove edges with an absolute value smaller than `less_than`
    """

    return {k:v for k, v in edges.iteritems() if float(_get_item_attribute(v, 'weight')) >= less_than}

def _gexf_graph(gexf_nodes, gexf_edges, gexf_attributes):
    """
    an entire gexf specification graph
    with nodes, edges and attributes as specified
    """
    out = '\n'.join([_gexf_header(), gexf_attributes, gexf_nodes, gexf_edges, _gexf_footer()])
    return out

def _gexf_header():
    return """<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.org/2001/xmlschema-instance" xsi:schemalocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">
    <meta lastmodifieddate="">
        <creator>rob levy</creator>
        <description>an enfolding model graph</description>
    </meta>
    <graph defaultedgetype="directed">
    """

def _gexf_footer():
    return """    </graph>
</gexf>"""
      
def _gexf_attributes(attributes):
    node_attributes = _filter_dict(attributes, 'attr_class', 'node')
    edge_attributes = _filter_dict(attributes, 'attr_class', 'edge')
    gexf_node_attributes = _gexf_class_attributes(node_attributes, 'node')
    gexf_edge_attributes = _gexf_class_attributes(edge_attributes, 'edge')
    return '\n'.join([gexf_node_attributes, gexf_edge_attributes])

def _gexf_class_attributes(attributes, class_name):
    out = '    <attributes class="{0}">\n{{attrs}}\n    </attributes>'.format(class_name)
    attr_string = '<attribute id="{id}" title="{title}" type="{attr_type}"/>'
    attrs = ['        ' + attr_string.format(id=k, title=v['title'],
        attr_type=v['attr_type'])
        for k, v in attributes.iteritems()]
    return out.format(attrs='\n'.join(attrs))

def _node_size(model, node):
    if node['node_type'] == 'country':
        size = model.countries[node['label']].x.sum()
    elif node['node_type'] == 'cs':
        country, sector = node['label']
        x = _aggregate_sectors(model.countries[country].x, _sector_aggregation)
        size = x[sector]
    else:
        size = -1
    return size
            
def _gexf_nodes_with_attvalues(nodes, model, size_func=_node_size):
    """
    Create a string of newline-separated GEXF nodes with attributes.

    size_func is called with model and a node from `nodes`
    """
    _create_node_attributes()
    node_attr_funcs = dict(size=size_func, type=_node_type)
    node_attvalues = _gexf_item_attvalues(nodes, model, node_attr_funcs)
    nodes = _gexf_item_with_attributes(nodes, node_attvalues)
    return "\n".join(['<nodes>'] + nodes + ['</nodes>'])

def _set_nodes(countries, sectors, sector_aggregation):
    global _id, _node_ids
    nodes = {}
    for c in countries:
        nodes[_id] = dict(label=c, node_type='country')
        _id += 1
    sectors = _aggregated_sectors(sectors, sector_aggregation)
    for cs in itertools.product(countries, sectors):
        nodes[_id] = dict(label=cs, node_type='cs')
        _id += 1
    _node_ids = _reverse_dictionary(nodes, key='label')
    return nodes

def _edges(cs_to_cs, c_to_cs, countries, sectors, sector_aggregation):
    """
    Convert two series, one of country/sector->country/sector flows
    and one of country->country/sector flows into a dict
    of gexf edge strings keyed on edge id
    """
    cs_to_cs = _aggregate_sectors(cs_to_cs.loc[:, countries, countries], sector_aggregation)
    c_to_cs = _aggregate_sectors(c_to_cs.loc[countries, :], sector_aggregation)
    cs_flows = _cs_to_cs(cs_to_cs)
    cs_edges = _gexf_edges(cs_flows, 'from_country_sector', 'to_country_sector')
    c_flows = _c_to_cs(c_to_cs)
    c_edges = _gexf_edges(c_flows, 'country', 'country_sector')
    return _set_edge_ids(cs_edges + c_edges)

def _gexf_edges_with_attvalues(edges, model):
    _create_edge_attributes()
    edge_attr_funcs = dict(is_positive=_edge_is_positive)
    edge_attvalues = _gexf_item_attvalues(edges, model, edge_attr_funcs, 'edge')
    edges = _gexf_item_with_attributes(edges, edge_attvalues, item_class='edge')
    return "\n".join(['<edges>'] + edges + ['</edges>'])

def _set_globals(countries, sectors, sector_aggregation):
    global _node_ids, _id, _sector_aggregation
    _id = 1
    _sector_aggregation = sector_aggregation

def _create_attributes(titles, types, attr_class='node'):
    """
    Create a list of attributes with IDs with titles and types
    """
    global _attribute_id, _attributes, _attr_ids
    if not _attributes:
        _attributes = {}
    if len(titles) != len(types):
        raise ValueError('types and titles must be the same length')
    for i in range(len(titles)):
        attr = dict(title=titles[i], attr_type=types[i], attr_class=attr_class)
        _attributes[_attribute_id] = attr
        _attribute_id += 1
    _attr_ids = _reverse_dictionary(_attributes, key=('attr_class', 'title'))

def _create_node_attributes():
    titles = ['size', 'type']
    types = ['float', 'string']
    _create_attributes(titles, types)

def _create_edge_attributes():
    titles = ['is_positive']
    types = ['boolean']
    _create_attributes(titles, types, attr_class='edge')

def _reverse_dictionary(dictionary, key=None):
    if key is None:
        return {v:k for k, v in dictionary.iteritems()}
    else:
        if isinstance(key, (list, tuple)):
            return {tuple([v[u] for u in key]): k for k, v in dictionary.iteritems()}
        else:
            return {v[key]:k for k, v in dictionary.iteritems()}

def _trade_flows(model, countries, sectors, sector_aggregation):
    """
    A Series of trade flows indexed on sector, from_country, to_country
    """
    flows = model.trade_flows()
    flows = _aggregate_sectors(flows, sector_aggregation)
    sectors = _aggregated_sectors(sectors, sector_aggregation)
    return flows.loc[sectors, countries, countries]
    
def _production_flows(model, countries, sectors, sector_aggregation):
    """
    Imaginary 'flows' from a country to a sector, indicating that
    country's production level
    """
    production = model.total_production()
    production.name = 'x'
    production = _aggregate_sectors(production, sector_aggregation)
    sectors = _aggregated_sectors(sectors, sector_aggregation)
    return production.loc[countries, sectors]

def _aggregate_sectors(series, sector_aggregation, sector_level='sector'):
    """
    Replace sector values and sum
    """
    if sector_aggregation is None:
        return series
    old_levels = series.index.names
    out = series.reset_index(sector_level)
    out[sector_level] = out[sector_level].map(sector_aggregation)
    out = out.dropna()
    append = len(series.index.names) > 1
    out = out.set_index(sector_level, append=append)
    try:
        out = out.reorder_levels(old_levels)
    except TypeError:
        pass
    try:
        return out.groupby(level=old_levels).sum().squeeze()
    except ValueError:
        return out.groupby(out.index).sum().squeeze()

def _aggregated_sectors(sectors, sector_aggregation):
    if sector_aggregation:
        return list(set(sector_aggregation.values()))
    else:
        return sectors

def _cs_to_cs(series, from_name='from_country', to_name='to_country', sector_level='sector'):
    """
    Convert a series to a series of country/sector to country/sector
    flows indexed on the IDs of the from and to CSs
    """
    with_cids = _replace_index(series, [from_name, sector_level],
            _node_ids, keep_level='sector')
    with_cids = _replace_index(with_cids, [to_name, sector_level],
            _node_ids)
    return with_cids.squeeze()

def _c_to_cs(series, country_level='country', sector_level='sector'):
    """
    Convert to a series of country to country/sector flows
    """
    with_csids = _replace_index(series, [country_level, sector_level], _node_ids, keep_level='country')
    with_cids = _replace_index(with_csids, country_level, _node_ids)
    return with_cids.squeeze()

def _replace_dict(v, dictionary, key=None, missing_value=-1):
    try:
        lookup = dictionary[v]
        if key:
            return lookup[key]
        else:
            return lookup
    except KeyError:
        return missing_value

def _filter_dict(dictionary, key, value):
    return {k:v for k, v in dictionary.iteritems() if v[key] == value}

def _replace_index(df_or_series, level_names, replace_dict, key=None, keep_level=None):
    if isinstance(level_names, (list, tuple)) and len(level_names) > 2:
        raise ValueError("level_names must have at most 2 values")
    out = df_or_series.reset_index(level_names)
    if len(level_names) == 2:
        new_name = '_'.join(level_names)
        out[new_name] = [_replace_dict(tuple(x), replace_dict) for x in out[level_names].values]
        out.drop([level for level in level_names if level != keep_level], axis=1, inplace=True)
    else:
        out[level_names] = [_replace_dict(x, replace_dict) for x in out[level_names].values]
        new_name = level_names
    if keep_level is not None:
        new_name = [new_name, keep_level]
    out = out.set_index(new_name, append=True)
    return out

def _gexf_node(node_id, node):
    """
    A <node> command
    """
    node_string = '<node id="{id:d}" label="{lbl}">'
    return node_string.format(id=node_id, lbl=node['label'])

def _gexf_edge(edge_id, edge):
    """
    An <edge> command
    """
    return edge

def _gexf_item_attvalues(items, model, attr_funcs, class_name='node'):
    """
    A list of <attvalue> commands. Values for each attr will be
    retrieved by calling `attr_funcs[attr_title]` with `model`
    and `node`
    """
    att_string = '<attvalue for="{id:d}" value="{val}" />'
    item_attributes = _filter_dict(_attributes, 'attr_class', class_name)
    item_attvalues = {}
    for item_id, n in items.iteritems():
        attvalues = []
        for attr_id, a in item_attributes.iteritems():
            attvalues.append(att_string.format(id=attr_id, val=attr_funcs[a['title']](model, n)))
        item_attvalues[item_id] = attvalues
    return item_attvalues

def _gexf_item_with_attributes(items, attributes, item_class='node'):
    item_string = "{{item}}\n  <attvalues>\n{{attvalues}}\n  </attvalues>\n</{item_class}>"
    item_string = item_string.format(item_class=item_class)
    if item_class == 'node':
        _gexf_fun = _gexf_node
    elif item_class == 'edge':
        _gexf_fun = _gexf_edge
    else:
        raise ValueError("Item class %s not recognised." % item_class)
    return [item_string.format(item=_gexf_fun(item_id, n),
        attvalues='\n'.join(['    ' + a for a in attributes[item_id]]))
        for item_id, n in items.iteritems()]

def _gexf_edges(series, from_name, to_name):
    """
    A list of <edge> commands with sources and targets from the
    index levels from_name and to_name of `series`. The weight
    of the edge is the value of `series`.
    """
    series = series[abs(series) > 0] # Remove zero weight edges
    edge_string = '<edge id="{{id:d}}" source="{src:d}" target="{tgt:d}" weight="{wt:.3f}" neg="{neg}">'
    sources = series.index.get_level_values(from_name)
    targets = series.index.get_level_values(to_name)
    weights = abs(series.values)
    neg = series < 0
    return [edge_string.format(src=x[0], tgt=x[1], wt=x[2], neg=x[3])
            for x in zip(sources, targets, weights, neg)]

def _set_edge_ids(edges):
    return {i:edge.format(id=i) for i, edge in enumerate(edges)}

def _node_type(model, node):
    return node['node_type']

def _edge_is_positive(model, edge):
    return _get_item_attribute(edge, 'neg')

def _get_item_attribute(item, attr_name):
    elements = item.split()
    weight = [x for x in elements if x.startswith(attr_name)][0]
    return weight.split('"')[1]
