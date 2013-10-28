# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 12:14:04 2013

@author: Rob
"""

from gdm_library import diagonalise
from country import Country 
import numpy as np
import pandas as pd

global __IMPORT_PREFIX
__IMPORT_PREFIX = "IM:"

def create_countries_from_io_data(data):
    """Given a list of data, creates all the necessary Country objects"""
    
    data = _preprocess_data(data)    
    countries = {'RoW':_create_RoW_country(pd.unique(data['from']),pd.unique(data['to']))}    
    for iso3, dat in data.groupby("country_iso3", sort=False):
        countries[iso3] = _create_country_from_data(iso3, dat)
    print (countries)
    return countries

def _preprocess_data(data):
    """ Prefix the "from" column with IMPORT_PREFIX for imports
    Add the group number onto the from and to columns
    Get rid of Value Added"""
    
    data = data[data['from'] != 'VA']    
#    data['from'] = (data['from_sector_group_number'] + " " 
#                    + data['from'])
    data['from'][data.is_import] = (__IMPORT_PREFIX + 
                                    data['from'][data.is_import])
#    data['to'] = data['to_sector_group_number'] + " " + data['to']
    data = data.drop("is_import",1)
    data = data.drop("from_sector_group_number", 1)
    data = data.drop("to_sector_group_number", 1)    

    return data

def _create_country_from_data(iso3, data):
    """Takes one country-worth of data and returns 
    a Country object containing all the relevant matrices,
    A, B_dagger, X, x, i, e, fd and the import ratios"""
        
    # We don't need the country_iso3 or year columns any more
    try:
        data = data.drop(["country_iso3","year"],1)
    except:
        pass            

    # start with production matrices X (domestic) and I (imported)
    domestic = data[~(data["from"].apply(_is_import))]
    imports = data[data["from"].apply(_is_import)]
    
    X = _sector_data_to_matrix(domestic)
    M = _sector_data_to_matrix(imports) # M is iMports
    x = X.sum(1) # Row sums
    i = M.sum(1) # Row sums
    # Get rid of the __IMPORT_PREFIX from i's row names:
    i = i.rename(_remove_import_prefix)
    
    # Now T, the actual input-output table
    T_domestic = domestic.pivot("from", "to", "flow_amount")
    T_imports = imports.pivot("from", "to", "flow_amount")
    T = pd.concat([T_domestic, T_imports])
    
    # Now B_dagger, the inter-sector flow matrix
    inter_sector_data = domestic[~(domestic.to.str.contains('EX')) & ~(domestic.to.str.contains('FD'))]    
    B_dagger = _sector_data_to_matrix(inter_sector_data)
    
    # And B_star the import part of the inter-sector flows
    inter_sector_import_data = imports[~(imports.to.str.contains('EX')) & ~(imports.to.str.contains('FD'))]    
    B_star = _sector_data_to_matrix(inter_sector_import_data)
    
    # The complete inter-sector flow matrix (domestic AND imported)
    B = pd.concat([B_dagger, B_star])    
    B_star = B_star.rename(_remove_import_prefix) # Get rid of pesky import prefixes
    
    # And the vector of exports, e
    e = _sector_data_to_matrix(domestic[domestic.to.str.contains('EX')])    
    
    # Now the final demand vector
    f_dagger = _sector_data_to_matrix(domestic[domestic.to.str.contains('FD')])
    f_star = _sector_data_to_matrix(imports[imports.to.str.contains('FD')])
    f = f_dagger + f_star.values # Total demand = demand for domestic + demand for imported
    
    # Now work out the import ratios
    D = _calculate_import_ratios(x, i)
    
    # Now calculate A
    F = f + e # total demand for each product
    xhat = diagonalise(1./x)
    A = (B_dagger + B_star).dot(xhat)
      
    country_data = {'x':x, 'i':i, 'e':e, 'B_dagger':B_dagger,
                    'B_star':B_star,'A':A, 'f':f, 'f_dagger':f_dagger, 'f_star':f_star, 'D':D}    
    
    return Country(iso3, country_data)

def _create_RoW_country(from_sectors, to_sectors):
    RoW_data = []
    for s1 in from_sectors:
        for s2 in to_sectors:
            RoW_row = {}
            RoW_row['country_iso3'] = 'RoW'
            RoW_row['from'] = s1
            RoW_row['to'] = s2
            if s1 == s2 or s2 == 'FD':
                RoW_row['flow_amount'] = 1.0
            else:
                RoW_row['flow_amount'] = 0.0
            RoW_data.append(RoW_row)
    RoW_data = pd.DataFrame(RoW_data)
    return _create_country_from_data('RoW', RoW_data)
 
def _sector_data_to_matrix(data):
    """Takes a DataFrame with from and to (sectors or sector
    groups, and produces a DataFrame from the 
    value_col_name entries)"""
    
    #return np.matrix(data.pivot('from','to','flow_amount'))
    pivoted = data.pivot('from','to','flow_amount')
    if np.any(pivoted.shape) == 1:
        pivoted = pivoted.squeeze()
    return pivoted

def _is_import(sector_name):
    s = str(sector_name)
    return s.startswith(__IMPORT_PREFIX)

def _calculate_import_ratios(domestic_production, total_imports):
    import_fractions = diagonalise(total_imports / (domestic_production + total_imports))
    return import_fractions
  
def _remove_import_prefix(name):
    return name[len(__IMPORT_PREFIX):]
    
