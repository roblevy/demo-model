# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 12:14:04 2013

@author: Rob
"""

import country as c
import import_propensities
import pandas as pd
import numpy as np

reload(import_propensities)
reload(c)

def _get_technical_coefficients(Z,x):
    """
    Calculates A = Zxhat^-1 as per Miller and Blair. The only exception
    is where total production (x) of a product is zero. In this case
    the technical coefficient is set to zero.
    """
    Z.name = 'flow_value'
    A = Z.reset_index(['from_sector'])
    A['x'] = rename_multiindex_level(x,'sector','to_sector')
    A['a'] = A['flow_value'] / A['x']
    A['a'] = A['a'].replace(np.inf,0).fillna(0)
    A = A.set_index('from_sector',append=True).swaplevel(1,2)
    return A['a']
    
def create_countries_from_data(sector_flows, trade_flows):
    """
    Using sector flow data, create:
      - the import vector, i
      - the export vector, e,
      - the matrix of technical coefficients, A
      - the vector of import ratios, D
    for all the countries in the data
    
    Set any negative flows to zero
    """
    sector_flows = sector_flows.set_index(['country','from_sector','to_sector'])
    flows = sector_flows['flow_amount']
    flows[flows < 0] = 0
    imports = flows[sector_flows['is_import']]
    final_demand = flows[sector_flows['is_final_demand'] \
        & sector_flows['from_production_sector']]
    exports = flows[sector_flows['is_export'] \
        & sector_flows['from_production_sector']]
    domestic = flows[~sector_flows['is_import'] \
        & sector_flows['from_production_sector']]

    # Domestic production (sum across all to_sector)
    x = domestic.sum(level=['country','from_sector'])
    x = rename_multiindex_level(x, 'from_sector','sector')
    # Imports (sum across all to_sector)
    m = imports.sum(level=['country','from_sector'])
    m = rename_multiindex_level(m, 'from_sector','sector')
    # Exports. There are some negative export demands. Set these to zero
    e = exports.sum(level=['country','from_sector'])
    e[e < 0] = 0
    e = rename_multiindex_level(e, 'from_sector','sector')
    # Final demands. Sum across all to_sector which will be the 3 final
    # demand sectors only!
    f = final_demand.sum(level=['country','from_sector'])
    f = rename_multiindex_level(f, 'from_sector','sector')
    # Now work out the import ratios
    d = m / ((x - e) + m)
    
    # Get only those flows relevant for input-output. Since we no longer care about
    # domestic versus imported, we simply sum to get the totals
    # First: 70x35 (including imports)    
    Z = flows[sector_flows['from_production_sector'] \
              & sector_flows['to_production_sector']]
    # Now: 35x35
    Z = Z.groupby(level=[0,1,2]).sum()

    # Technical coefficients
    A = _get_technical_coefficients(Z, x)
    
    # Get a list of country names
    countries = sector_flows.index.levels[0].values.tolist()
    # Create a dictionary of country objects
    country_objects = {}
    for country in countries:
        country_objects[country] = c.Country(country, 
            final_demand=f.ix[country],
            # Create a matrix from long-format data
            technical_coefficients=A.ix[country].unstack(1), 
            import_ratios=d.ix[country])
    return country_objects

def create_RoW_country(stray_exports, stray_imports, sectors):
    """ 
    Create a Rest of World country according to special rules.
    
    Imports to RoW are all those
    flows not going to a country in `countries`.
    Import propensities are then calculated as normal. Final Demand
    is set to be identical to imports. Investments are 0.
    
    Since production will later be set to be identical to exports,
    the RoW will simply produce whatever is needed for export, and 
    consume only that which is imported.    
    """
    # Initialise the pd.Series which will store the i and e values
    # This is necessary to ensure ALL sectors have an entry    
    df = pd.DataFrame({'sector':sectors,'trade_value':0}) \
        .set_index('sector').squeeze()
    i = stray_exports.groupby('sector').aggregate(sum)['trade_value']
    # Since COMTRADE numbers are in $s and WIOD numbers are in 
    # $1,000,000s we need to divide first
    RoW_imports = df.add(i,fill_value=0) / 1e6
    return c.Country('RoW',final_demand=RoW_imports,
                     technical_coefficients=0,
                     import_ratios=0)
  

def rename_multiindex_level(X, old, new):
    """
    This is a workaround to this bug in Pandas:
    https://github.com/pydata/pandas/pull/3175
    """
    is_series = isinstance(X, pd.Series)
    X = pd.DataFrame(X)
    index_names = X.index.names
    X = X.reset_index()
    X = X.rename(columns={old:new})
    index_names = [name.replace(old,new) for name in index_names]
    X = X.set_index(index_names)
    if is_series:
        X = X.ix[:,0]
    return X
    