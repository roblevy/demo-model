# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 14:02:49 2013

@author: Rob
"""
import pandas as pd
from itertools import product

def calculate_import_propensities(trade_data, import_totals, countries, sectors):
    """ Calculate import propensities by the following algorithm:
    1. Split the trade data into importer and sector
    2. Sum the trade value of each group
    3. Divide every trade value by the grouped sums
    In cases where a country imports zero of a particular product, the propensities
    cannot be calculated in this way. We therefore set all import propensities
    to zero apart from that associated with RoW
    """
    t = trade_data
    # Calculate each import propensity, p
    t = t.set_index(['sector','from_iso3','to_iso3']).sortlevel()
    import_totals = t.groupby(level=['sector', 'to_iso3']).transform(sum)
    p = t / import_totals
    p = p.sum(level=[0,1,2]) # This sums over all duplicate from/to entries
    sectors = sorted(sectors)
    country_names = sorted(countries.keys())
    P = pd.DataFrame(list(product(sectors, country_names, country_names)),
                     columns=['sector', 'from_country', 'to_country'])
    P = P.set_index(['sector', 'from_country', 'to_country'])
    P['p'] = p
    P = P.p.fillna(0).squeeze()
     
    # any sector, to_country which doesn't sum to unity,
    # gets the remainder from RoW
    P = P.unstack(level='to_country')
    for s in P.index.levels[0]:
        P_s = P.ix[s]
        col_sums = P_s.sum(0)
        P.ix[s, 'RoW'][col_sums < 1] = \
            P_s.ix['RoW'][col_sums < 1] + (1 - col_sums)
    return P.stack()
        
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
    
    
