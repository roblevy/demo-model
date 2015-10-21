import os
import itertools
import pandas as pd
import numpy as np
from demo_model.global_demo_model import GlobalDemoModel as gdm
from demo_model.tools import sectors

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(SCRIPT_DIR, '..', 'models/')
data_path = os.path.join(SCRIPT_DIR, '..', 'resources/')
services_path = data_path

def augmented_flows(model, add_zeros=False):
    """
    Get all the COMTRADE and ServiceTrade flows for a particular year and
    augment with various observables from the model
    for that year.

    Note that d_ij is measured in 1,000s of kilometres
    """
    year = model.year
    #%%
    # Create df
    commodities = pd.read_csv(data_path + 'fn_trade_flows_%i.csv' % year)
    services = pd.read_csv(services_path
            + 'services_flows_sectors_%i.csv' % year)
    df = pd.concat([commodities, services])
    df = df[~df.sector.isnull()] # Drop flows with no sector
    df = df[['from_iso3', 'to_iso3', 'sector', 'trade_value']]
    df = df.rename(columns={'from_iso3':'from_country',
        'to_iso3':'to_country',
        'trade_value':'y_ijs'})
    # Only interested in flows within modelled countries
    df = df[df.to_country.isin(model.country_names)]
    df = df[df.from_country.isin(model.country_names)]

    # Sum over duplicates, since some sectors come from COMTRADE and
    # ServiceTrade (see, e.g. "H3-490600" which is a Business Services
    # commoditiy!
    df = df.groupby(['from_country', 'to_country', 'sector']).sum().reset_index()

    # Add zeros
    if add_zeros:
        key_columns = ['from_country', 'to_country', 'sector']
        all_values = _all_possible_values(df, key_columns).set_index(key_columns).sortlevel()
        all_values['y_ijs'] = 0
        combined = (all_values + df.set_index(key_columns)).reset_index().fillna(0)
        df = combined.copy()

    # Not interested in self-flows
    df = df[df.from_country != df.to_country]

    # y
    #df = df[df.y_ijs > 0]
    df.y_ijs /= 1e6

    # m_js
    df['m_js'] = df.groupby(['to_country', 'sector'])['y_ijs'].transform(sum)

    # p_ijs
    df['p_ijs'] = df.y_ijs / df.m_js

    # x_is
    x_is = pd.DataFrame(model.total_production(), columns=[['x_is']])
    df = pd.merge(df, x_is, how='inner',
        left_on=['from_country', 'sector'], right_index=True)

    # x_i
    x_i = x_is.groupby(level='country').sum().rename(columns={'x_is':'x_i'})
    df = pd.merge(df, x_i, how='inner',
                left_on=['from_country'], right_index=True)

    # x_j
    x_j = pd.DataFrame(model.total_production(), columns=[['x_j']])
    x_j = x_j.groupby(level='country').sum()
    df = pd.merge(df, x_j, how='inner',
	        left_on=['to_country'], right_index=True)

    # f_is
    f_is = pd.DataFrame(model.final_demand()).rename(columns={'final_demand':'f_is'})
    df = pd.merge(df, f_is, how='inner',
        left_on=['from_country', 'sector'], right_index=True)

    # e_i
    df['e_i'] = df.groupby(['from_country'])['y_ijs'].transform(sum)

    # technical coefficients
    for s in pd.unique(df.sector):
        df = _add_tech_coeffs(model, df, s)

    # v_is
    # NOTE: This removes rows relating to country/sectors with no technical coefficients!
    v_is = pd.DataFrame(model.value_added_per_unit(), columns=[['v_is']])
    print "To be excluded:"
    print [cs for cs in v_is[(v_is == 0) | (v_is == 1)].dropna().index
        if sectors.sector_is_commodity(cs[1])]
    v_is = v_is[(v_is > 0) & (v_is < 1)].dropna()
    df = pd.merge(df, v_is, how='inner',
        left_on=['from_country', 'sector'], right_index=True)

    d = minimum_distances(year)
    df = pd.merge(df, d, how='left',
        left_on=['from_country', 'to_country'], right_index=True)

    return df.dropna()

def minimum_distances(year):
    """
    Minimum distance, d, in 1000s of kilometres
    """
    d = pd.read_csv(data_path + 'country_distances_%i.csv' % year)
    d = d.set_index(['country1', 'country2'])
    d.names = ['d']
    return d / 1000

def _add_tech_coeffs(model, df, sector):
    tech_coeffs = model.technical_coefficients() \
            .reset_index().set_index(['from_sector', 'country', 'to_sector'])
    sector_name = sectors.format_sector_name(sector)
    to_sector = tech_coeffs.ix[sector].rename(columns={0:sector_name})
    return pd.merge(df, to_sector, how='inner',
            left_on=['from_country', 'sector'], right_index=True)

def _all_possible_values(df, column_names):
    """
    All possible combinations of all (unique) values
    from `df[column_names]`

    Will also work if column_names are index levels
    """
    try:
        index_levels = {k: df[k].unique() for k in column_names}
    except KeyError:
        # Maybe these are index levels, not column names
        index_levels = {k: df.reset_index()[k].unique() for k in column_names}
    all_values = [p for p in itertools.product(*index_levels.values())]
    column_names = [k for k in index_levels] # This ensures the order of the columns is maintained
    return pd.DataFrame(all_values, columns=column_names)
