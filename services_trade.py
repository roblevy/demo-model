# -*- coding: utf-8 -*-
"""
Created on Tue Jul 09 15:01:21 2013

@author: Rob
"""
import numpy as np
import pandas as pd

__INIT_VALUE__ = 1.0
__EXPORT_CODE__ = 'Export'
__IMPORT_CODE__ = 'Import'
__WITH_RoW__ = True
__MAX_ITERATIONS__ = 30

def estimate_services_trade_flows(data):
    """ Makes sure that global imports match global exports,
    then performs an iterative balancing procedure to estimate
    a flow matrix from 'data' which is a pandas DataFrame with
    total imports and exports per country/sector."""
    # Try to drop any 'year' column
    try:
        data = data.drop('year',1)
    except:
        pass
    # Scale the data
    sectors = data['sector_group_short_name'].unique()
    scaled_data = _balance_imports_and_exports(data, sectors)
    # Get lists of countries (possibly include RoW)
    countries = scaled_data['country_iso3'].unique()
    # Get the flow_data    
    flow_data = _services_flow_data(scaled_data, countries, sectors, __MAX_ITERATIONS__)
    return flow_data

def _add_rest_of_world(data, sectors):
    for sector in sectors:
        for import_export in [__EXPORT_CODE__, __IMPORT_CODE__]:
            new_row = pd.Series({'country_iso3':'RoW',
                                 'sector_group_short_name':sector,
                                 'trade_flow':import_export,
                                 'value':float(pow(10,4))})
            data = data.append(new_row, ignore_index=True) # Required for appending a Series to a DataFrame                     
    data = data.groupby('sector_group_short_name').apply(_set_RoW_sector_imports)
    return data

def _set_RoW_sector_imports(data):
    """ data is a single sectors-worth of global import/export
    data."""
    total_export = data[data.trade_flow==__EXPORT_CODE__].value.sum()
    total_import = data[data.trade_flow==__IMPORT_CODE__].value.sum()
    required_import = total_export - total_import
    data.value[((data.country_iso3=='RoW') & (data.trade_flow==__IMPORT_CODE__))] += required_import
    return data
    
def _balance_imports_and_exports(data, sectors):
    if __WITH_RoW__:
        return _add_rest_of_world(data, sectors)
    else:
        return _scale_exports_to_sum_to_imports(data, sectors)

def _scale_exports_to_sum_to_imports(data, sectors):
    """ For each sector in turn, scale the exports such that
    totals global exports matches total global imports."""
    data = data.sort(['country_iso3','sector_group_short_name','trade_flow'])
    totals = data.groupby(['sector_group_short_name','trade_flow']).sum()
    for sector in sectors:
        _scale_sector_exports(data, totals, sector)
    # Test the balancing procedure
    if ~(np.allclose(data[data.trade_flow=="Import"]['value'].sum(), 
                     data[data.trade_flow=="Export"]['value'].sum())):
        print "ERROR: Balancing imports to exports has failed."
    return data
 
def _scale_sector_exports(data, totals, sector):
    """ data is a pandas DataFrame containing global import and export
    totals for a single sector. This function multiplies the exports
    such that total exports matches total imports"""
    sector_totals = totals.ix[sector]            
    i_total = sector_totals.ix[__IMPORT_CODE__].values[0]
    e_total = sector_totals.ix[__EXPORT_CODE__].values[0]
    scale_factor = float(i_total) / e_total
    e = data[(data.sector_group_short_name == sector) & (data.trade_flow == __EXPORT_CODE__)]
    e['value'] = e['value'] * scale_factor
    data.update(e)
    
def _initial_trade_flow_matrices(data, sectors, countries):
    """ Initialises the trade flow matrices, one per sector in
    'sectors', that will later contain the estimated trade flows. 
    Has rows and columns named after the countries in 'countries'"""
    Y = {}
    num_countries = len(countries)
    for sector in sectors:
        Y_s = np.ones([num_countries,num_countries]) * __INIT_VALUE__
        Y_s = Y_s * (1 - (np.eye(num_countries))) # * (1 - 10**-8))) # Set diagonals to zero
        df_Y = pd.DataFrame(Y_s, index=countries, columns=countries)
        df_Y.index.name = "from_iso3"
        df_Y.columns.name = "to_iso3"
        df_Y.name = sector
        if __WITH_RoW__:
            # Set the RoW to be able to import from itself
            df_Y.ix['RoW']['RoW'] = __INIT_VALUE__ * pow(10,8)
        Y[sector] = df_Y
    return Y

def _services_flow_data(data, countries, sectors, num_iterations):
    """ Data is a pandas DataFrame containing import and export totals
    for each country in 'countries' and each sector in 'sectors'. This
    function runs an iterative process to produce an actual flow
    matrix from country to country per sector. """
    eps = 0.00010
    alpha = 0.0
    Y = _initial_trade_flow_matrices(data, sectors, countries)
    print "Estimating services flow matrix..."
    for sector in Y.keys():
        Y_s = Y[sector]
        i = data[(data.sector_group_short_name==sector) & (data.trade_flow==__IMPORT_CODE__)]
        i = i.set_index('country_iso3').value
        e = data[(data.sector_group_short_name==sector) & (data.trade_flow==__EXPORT_CODE__)]
        e = e.set_index('country_iso3').value
        for iteration in range(num_iterations):
            Y_s_prev = Y_s.copy()
            # Columns (imports)
            Y_s = _proportional_fitting(Y_s, countries, i, alpha, False)
            if _has_converged(Y_s_prev, Y_s, eps):
                break
            Y_s_prev = Y_s.copy()
            # Rows (exports)
            Y_s = _proportional_fitting(Y_s, countries, e, alpha, True)
            if _has_converged(Y_s_prev, Y_s, eps):
                break
        print "Sector %s" % sector
        if num_iterations - iteration < 2:
            print "Failed to converge!"
        else:
            print "stopped after %i iterations. Converged?" % iteration
            #print np.allclose(Y_s.sum(0), i) & np.allclose(Y_s.sum(1), e)
        Y[sector] = Y_s
    return _long_format_services_flow_data(Y)

def _proportional_fitting(flow_matrix, countries, required_totals, alpha, do_rows):
    """ Runs half an iteration (either rows or columns) of the 
    iterative proportional fitting procedure (IPFP) attempting 
    to match the elements of 'flow_matrix' such that the rows or 
    columns sum to 'totals'."""
    old_matrix = flow_matrix.copy()
    direction = 1 * do_rows
    totals = flow_matrix.sum(direction) # Row/column total
    scale_factors = required_totals / totals
    flow_matrix = flow_matrix.mul(scale_factors, 1 - direction)

    return (alpha * old_matrix) + (1 - alpha) * flow_matrix
    
def _has_converged(old, new, epsilon):
    """ Finds the absolute difference between each element of the 
    matrix 'old' and the corresponding element of matrix 'new'.
    Sums the differences and tests whether the total difference
    is smaller than 'epsilon'"""
    diffs = (old - new).abs()
    total_diff = diffs.sum().sum()
    return total_diff < epsilon

def _long_format_services_flow_data(data):
    """ 'data' is a dictionary, keyed by sector, of flow matrices,
    country-to-country, giving flow amounts such that import and
    export totals are balanced. This function reformats the
    dictionary into a 'long format' pandas.DataFrame"""
    long_format = pd.DataFrame()    
    for sector in data.keys():
        Y_s = data[sector]
        long_Y = Y_s.stack() # Creates long-format data
        long_Y.name = 'value'
        long_Y = long_Y.reset_index() # Converts pandas MultiIndex into normal DataFrame
        long_Y['sector_group_short_name'] = sector # Add a sector column
        long_format = pd.concat([long_format, long_Y])
    # Get rid of the rows where from_iso3 = to_iso3
    long_format = long_format[long_format.from_iso3 != long_format.to_iso3]
    # Reorder the columns and it's ready to go!
    return pd.DataFrame(long_format, 
                        columns=['from_iso3','to_iso3','sector_group_short_name','value'])
        