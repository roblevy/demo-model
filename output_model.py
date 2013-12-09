# -*- coding: utf-8 -*-
'''
Created on Mon Sep 09 16:18:16 2013

@author: Rob
'''

import json
from math import sin, cos, pi

__FD_COLOUR__ = 'darkgrey'

def jsonify_model(models, filename, sectors=[]):
    offset = [800,800]
    r1 = 200
    r2 = 100
    alpha = 1.2
    data = {'models':[]}
    for model_number, model in enumerate(models):
        if len(sectors) == 0:
            sectors = model.sectors
        model_data = {'nodes':[], 'links':[]}
        country_ids = _set_arbitrary_country_ids(model)
        countries = [c for k, c in sorted(model.countries.items()) if k != 'RoW']
        io_cutoff = 0.0
        trade_cutoff = 0.0
        import_cutoff = 0.0
        # Nodes
        for c in countries:
            # Country nodes
            node = {}
            node['id'] = node['parent'] = country_ids[c.name]
            node['size'] = c.x.sum() # Total output of country c
            node['group'] = None
            node['label'] = c.name
            model_data['nodes'].append(node)
            # Final demand nodes
            node = {}
            node['id'] = model.get_id(c.name, 'FD')
            node['size'] = c.f.sum()
            node['group'] = -1
            node['colour'] = __FD_COLOUR__
            node['parent'] = country_ids[c.name]
            node['label'] = '%s FD' % c.name
            model_data['nodes'].append(node)
            for sector_id, s in enumerate([sector for sector in model.sectors if sector in sectors]):
                # Sector nodes            
                node = {}
                node['size'] = c.x[s] # Total output of sector s in country c
                node['id'] = model.get_id(c.name, s)
                node['group'] = sector_id
                node['parent'] = country_ids[c.name]
                node['label'] = '%s %s' % (c.name, s)
                model_data['nodes'].append(node)
                # Import sector flows to final demand
                link_value = c.f_star[s]
                if link_value > import_cutoff:
                    link = {}
                    link['source'] = _country_id(country_ids, c.name)
                    link['target'] = model.get_id(c.name, 'FD')
                    link['group'] = sector_id
                    link['value'] = link_value
                    model_data['links'].append(link)
                # Domestic sector flows to final demand
                link_value = c.f_dagger[s]
                if link_value > io_cutoff:
                    link = {}
                    link['source'] = model.get_id(c.name, s)
                    link['target'] = model.get_id(c.name, 'FD')
                    link['group'] = sector_id
                    link['value'] = link_value
                    model_data['links'].append(link)
                for sector_id2, s2 in enumerate([sector for sector in model.sectors if sector in sectors]):
                    if s != s2:           
                        # Internal flows (input-output)
                        link_value = c.B_dagger[s2][s]
                        if link_value > io_cutoff:
                            link = {}
                            link['source'] = model.get_id(c.name, s)
                            link['target'] = model.get_id(c.name, s2)
                            link['group'] = sector_id                        
                            link['value'] = link_value
                            model_data['links'].append(link)
                        # Import flows
                        link_value = c.B_star[s2][s]
                        if link_value > import_cutoff:
                            link = {}
                            link['source'] = _country_id(country_ids, c.name)
                            link['target'] = model.get_id(c.name, s2)
                            link['group'] = sector_id
                            link['value'] = link_value
                            model_data['links'].append(link)
                        
        for sector_id, s in enumerate([sector for sector in model.sectors if sector in sectors]):
            for c1 in countries:
                for c2 in countries:
                    if c1.name != c2.name:
                        # External flows (trade)
                        link_value = model.trade_flows(s)[c2.name][c1.name]
                        if link_value > trade_cutoff:                
                            link = {}
                            link['source'] = _country_id(country_ids, c1.name)
                            link['target'] = _country_id(country_ids, c2.name)
                            link['group'] = sector_id
                            link['value'] = link_value
                            model_data['links'].append(link)
                    
        _set_xy_coords(model_data, offset, r1, r2, alpha)
        # Build up the top-level object, which is a collection of models
        data['models'].append(model_data)
    f = open('../JSON Dumps/%s_gdm.json' % filename #time.strftime('%Y-%m-%d %H_%M_%S', time.localtime()) 
             ,'w')
    json.dump(data, f)
            
def _set_arbitrary_country_ids(model):
    node_id = max(model.id_list.keys()) + 1
    country_ids = {}    
    for c in model.countries.iterkeys():
        country_ids[c] = node_id
        node_id += 1
    return country_ids
    
def _set_xy_coords(data, offset, r1, r2, alpha):
    ''' Places the countries equally around the circumference of a circle
    then places the sectors in a smaller circle 'outside' the circle of
    countries, nearest to the country in question '''
    top_level = [node for node in data['nodes'] if _node_is_top_level(node)]
    sub_levels = [node for node in data['nodes'] 
                    if not _node_is_top_level(node) and not _node_is_final_demand(node)]
    sub_levels_fd = [node for node in data['nodes']
                        if _node_is_final_demand(node)]
    n = len(top_level)
    k = 1
    #  Top level first
    for node in top_level:
        xy = _get_xy(node, offset, k, n, r1)
        node['x'] = xy[0]
        node['y'] = xy[1]
        node['k'] = k
        k += 1
    # Now the sub-levels. Don't count final demand as a sector, though
    for node in sub_levels:
        xy = _get_xy(node, offset, k=_get_attr_from_node(data['nodes'],'k',node['parent']), 
                     n=n, r1=r1, is_top_level=False, 
                     l=node['group'], m=_size_of_sub_level(data['nodes'], node['parent']), r2=r2, alpha=alpha)
        node['x'] = xy[0]
        node['y'] = xy[1]
    for node in sub_levels_fd:
        xy = _get_fd_xy(node, offset, k=_get_attr_from_node(data['nodes'],'k',node['parent']),
                     n=n, r1=r1, alpha=alpha)
        node['x'] = xy[0]
        node['y'] = xy[1]

    
def _node_is_top_level(node):
    return node['id'] == node['parent']

def _node_is_final_demand(node):
    return node['label'][len(node['label']) - 2:] == 'FD'


def _size_of_sub_level(nodes, parent_id, exclude_fd=True):
    """ How many sub_nodes belong to the node with id parent_id. Final deman
    nodes don't count if exclude_fd is True"""
    if exclude_fd:        
        sub_level = [node for node in nodes if (node['parent'] == parent_id 
                                            and node['parent'] != node['id'])
                                            and not _node_is_final_demand(node) ]   
    else:
        sub_level = [node for node in nodes if (node['parent'] == parent_id 
                                            and node['parent'] != node['id'])]
    return len(sub_level)
    
def _get_xy(node, offset, k, n, r1, is_top_level=True, l=0, m=0, r2=None, alpha=None):
    """ Get the xy co-ordinates which will place a single node in a circles
    orbiting a circle arrangement. The inner ring is made up of is_top_level=True nodes
    and has radius r1. There are n elements in the inner ring and we're finding the
    xy values for the kth element.
    Associated with each of the n inner elements is an outer ring. This ring has m
    elements and we are finding the xy values of the lth of these. This outer ring
    has a radius r2 and is (1+alpha) * r1 away from the centre of the whole system"""
    if is_top_level:
        x = r1 * sin(2 * pi * k / n)
        y = r1 * cos(2 * pi * k / n)
    else:
        x = (1 + alpha) * r1 * sin(2 * pi * k / n) + r2 * sin(2 * pi * (float(k) / n + float(l) / m))
        y = (1 + alpha) * r1 * cos(2 * pi * k / n) + r2 * cos(2 * pi * (float(k) / n + float(l) / m))
        
    return [x + offset[0], y + offset[1]]

def _get_fd_xy(node, offset, k, n, r1, alpha):
    """ Applies the same formula as _get_xy but
    beta is the fraction of alpha to apply and 
    gamme is an offset around the circumference. Smaller
    numbers mean bigger offset"""    
    beta = 2.3
    gamma = 3.0
    xy = _get_xy(node, offset, k=(gamma*k + 1), n=gamma*n, r1=(1 + beta) * r1, alpha=beta)
    return xy
    
def _get_attr_from_node(nodes, attr, nodeid):
    node = [node for node in nodes if node['id'] == nodeid][0]
    return node[attr]
    
def _country_id(country_ids, country_name):
    return [cid for name, cid in country_ids.iteritems() if name==country_name][0]