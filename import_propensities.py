# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 14:02:49 2013

@author: Rob
"""
import pandas as pd

def calculate_import_propensities(trade_data, import_totals, countries, sectors):
    """ Calculate import propensities by the following algorithm:
    1. Split the trade data into importer and sector
    2. Sum the trade value of each group
    3. Divide every trade value by the grouped sums
    In cases where a country imports zero of a particular product, the propensities
    cannot be calculated in this way. We therefore set all import propensities
    to zero apart from that associated with RoW
    """
    P = {}
    t = trade_data
    # Calculate each individual import propensity
    t = t.merge(pd.DataFrame(import_totals),
                left_on=['sector','to_iso3'],
                right_index=True)
    t['p_j'] = t.trade_value / t.i
#    # 'Unstack' the table. This creates a matrix with from_iso3 down the left
#    # and to_iso3 along the top
    t = t.set_index(['sector','from_iso3','to_iso3'])
    t = t.sum(level=[0,1,2]) # This sums over all duplicate from/to entries
    p_matrices = t['p_j'].unstack(2)
#    
    # Now create the P matrices    
    for sector in sectors:  
        P_i = pd.DataFrame(0,index=countries,columns=countries)
        try:
            P_i = P_i.add(p_matrices.ix[sector],fill_value=0)
        except:
            pass
        # Any columns which don't sum to unity, get the remainder
        # Coming from the RoW
        col_sums = P_i.sum(0)
        P_i.ix['RoW'][col_sums < 1] = P_i.ix['RoW'][col_sums < 1] + (1 - col_sums)
        P_i.columns.name = 'to_iso3'
        P_i.index.name = 'from_iso3'
        P[sector] = P_i
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
    
    
