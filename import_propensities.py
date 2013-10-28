# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 14:02:49 2013

@author: Rob
"""
import pandas as pd

def calculate_import_propensities(countries, trade_data):
    """ Calculate import propensities by the following algorithm:
    1. Split the trade data into importer and sector
    2. Sum the trade value of each group
    3. Divide every trade value by the grouped sums
    In cases where a country imports zero of a particular product, the propensities
    cannot be calculated in this way. We therefore set all import propensities
    to zero apart from that associated with RoW
    """
    sectors = countries[countries.keys()[0]].D.index.values
    
    P = {}
    g = trade_data.groupby(['to_iso3','sector_group_short_name'])
    div_by_sum = lambda(x): x / sum(x)
    trade_data['p_j'] = g['value'].apply(div_by_sum)

    for sector in sectors:
        sector_data = trade_data[trade_data['sector_group_short_name'] == sector]
        ## Initialise P[sector]
        P[sector] = pd.DataFrame(0.0, index=countries.keys(), columns=countries.keys())
        for row_number in sector_data.index:
            _populate_P(P[sector], sector_data[sector_data.index==row_number])
        _set_countries_which_import_from_themselves(P[sector])

        
    return P

        
def _data_contains_sector(data, sector):
    sectors = pd.unique(data.sector_group_short_name)
    return sector in sectors
    
def _populate_P(P, flow_data_row):
    row = flow_data_row   
    from_iso3 = row['from_iso3'].values[0]
    to_iso3 = row['to_iso3'].values[0]
    p_j = row['p_j'].values[0]
    P[to_iso3][from_iso3] = p_j
    
def _set_countries_which_import_from_themselves(P):
    """ There may be some countries which don't feature at all in the
    trade data (RoW is an obvious example). Others may simply not import
    any of a certain good. These countries are allowed to
    trade entirely with RoW (i.e. their P_ij is 1)"""
    for colname, col in P.iteritems():
        if col.sum() < pow(10,-8):
            P[colname] = 0
            P[colname]['RoW'] = 1
    
    
