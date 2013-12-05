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
__MAX_ITERATIONS__ = 500


def estimate_services_trade_flows(data):
    """ Make sure that global imports match global exports,
    then perform an iterative balancing procedure to estimate
    a flow matrix from 'data' which is a pandas DataFrame with
    total imports and exports per country/sector."""
    data = data[pd.notnull(data.country_iso3)] #  Remove unknown areas (which ServicesTrade is full of)
    sectors = data['sector'].unique()
    countries = data['country_iso3'].unique()
    # Get the flow_data    
    flow_data = _services_flow_data(data, countries, sectors, __MAX_ITERATIONS__)
    return flow_data

def _add_rest_of_world(data, sectors):
    for sector in sectors:
        for import_export in [__EXPORT_CODE__, __IMPORT_CODE__]:
            new_row = pd.Series({'country_iso3':'RoW',
                                 'sector':sector,
                                 'trade_flow':import_export,
                                 'trade_value':float(pow(10,4))})
            data = data.append(new_row, ignore_index=True) # Required for appending a Series to a DataFrame                     
    data = data.groupby('sector').apply(_set_RoW_sector_imports)
    return data

def _set_RoW_sector_imports(data):
    """ data is a single sectors-worth of global import/export
    data."""
    total_export = data[data.trade_flow==__EXPORT_CODE__].trade_value.sum()
    total_import = data[data.trade_flow==__IMPORT_CODE__].trade_value.sum()
    required_import = total_export - total_import
    data.trade_value[((data.country_iso3=='RoW') & (data.trade_flow==__IMPORT_CODE__))] += required_import
    return data
    
     
def _initial_trade_flow_matrices(data, sectors, countries):
    """ 
    Initialises the trade flow matrices, one per sector in
    'sectors', that will later contain the estimated trade flows.
    """
    Y = {}
    for sector in sectors:
        import_deficit = _import_deficit(data.ix[sector])
        Y[sector] = _initial_trade_flow_matrix(sector, countries, import_deficit < 0)
    return Y

def _initial_trade_flow_matrix(sector, countries, im_exceed_ex):
    """n
    For three countries, A,B,C and an __INIT_VALUE__ of 1 produces
    one of two types of matrix. First, if imports exceed exports:
        A B C
      A 0 1 1
      B 1 0 1
      C 1 1 0
    RoW 1 1 1
    
    Or, if exports exceed imports:
        A B C RoW
      A 0 1 1 1
      B 1 0 1 1
      C 1 1 0 1
    
    """
    num_countries = len(countries)
    Y_s = np.ones([num_countries,num_countries]) * __INIT_VALUE__
    Y_s = Y_s * (1 - (np.eye(num_countries))) # Set diagonals to zero
    RoW = np.ones(len(countries)) * __INIT_VALUE__
    if im_exceed_ex:
        Y_s = np.vstack([Y_s,RoW]) # Add a row on the bottom for RoW
        row_names = np.append(countries,'RoW')
        col_names = countries
    else:
        Y_s = np.column_stack([Y_s,RoW])
        row_names = countries
        col_names = np.append(countries,'RoW')
    df_Y = pd.DataFrame(Y_s, index=row_names, columns=col_names)
    df_Y.index.name = "from_iso3"
    df_Y.columns.name = "to_iso3"
    df_Y.name = sector
    return df_Y

def _targets(data, sector):
    d = data.ix[sector]
    row_targets = d.ix[__EXPORT_CODE__]['trade_value']
    col_targets = d.ix[__IMPORT_CODE__]['trade_value']
    deficit = _import_deficit(d) # Work out RoW's target
    RoW = pd.Series({'RoW':np.abs(deficit)})
    if deficit < 0:
        # Columns are bigger than rows. Add an RoW row total
        row_targets = row_targets.append(RoW)
    else:
        col_targets = col_targets.append(RoW)
    return row_targets, col_targets

def _import_deficit(d):
    export_total = d.ix[__EXPORT_CODE__]['trade_value'].sum()
    import_total = d.ix[__IMPORT_CODE__]['trade_value'].sum()
    return export_total - import_total

def _services_flow_data(data, countries, sectors, num_iterations):
    """ Data is a pandas DataFrame containing import and export totals
    for each country in 'countries' and each sector in 'sectors'. This
    function runs an iterative process to produce an actual flow
    matrix from country to country per sector. """
    data = data.set_index(['sector','trade_flow','country_iso3'])    
    eps = 1
    alpha = 0.0
    Y = _initial_trade_flow_matrices(data, sectors, countries)
    
    print "Estimating services flow matrix..."
    for sector, Y_s in Y.iteritems():
        row_targets, col_targets = _targets(data, sector)
        for iteration in range(num_iterations):
            Y_s_prev = Y_s.copy()
            # Columns (imports)
            Y_s = _proportional_fitting(Y_s, col_targets, alpha, False)
            if _has_converged(Y_s_prev, Y_s, eps):
                break
            Y_s_prev = Y_s.copy()
            # Rows (exports)
            Y_s = _proportional_fitting(Y_s, row_targets, alpha, True)
            if _has_converged(Y_s_prev, Y_s, eps):
                break
        print "Sector %s" % sector
        if num_iterations <= iteration:
            print "Failed to converge!"
            break
        else:
            print "stopped after %i iterations. Converged: %s" % (
                iteration,
                np.allclose((Y_s.sum(0) - col_targets).dropna(), 0, atol=1)
                & np.allclose((Y_s.sum(1).dropna() - row_targets).dropna(), 0, atol=1)
                                                                  )
        Y[sector] = Y_s
    return _long_format_services_flow_data(Y)

def _proportional_fitting(flow_matrix, required_totals, alpha, do_rows):
    """ Runs half an iteration (either rows or columns) of the 
    iterative proportional fitting procedure (IPFP) attempting 
    to match the elements of 'flow_matrix' such that the rows or 
    columns sum to 'totals'."""
    if alpha > 0:
        old_matrix = flow_matrix.copy()
    else:
        old_matrix = 0
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
        long_Y = pd.DataFrame(Y_s.stack()) # Creates long-format data
        long_Y.columns = ["trade_value"]
        long_Y.index.names = ('from_iso3','to_iso3')
        long_Y = long_Y.reset_index()
        long_Y['sector'] = sector # Add a sector column
        long_format = pd.concat([long_format, long_Y])
        long_format.reset_index()
    # Get rid of the rows where from_iso3 = to_iso3
    long_format = long_format[long_format.from_iso3 != long_format.to_iso3]
    # Get rid of zero flows
    long_format = long_format[long_format['trade_value'] > 0]
    # Reorder the columns and it's ready to go!
    return pd.DataFrame(long_format, 
                        columns=['from_iso3','to_iso3','sector','trade_value'])
        