import pandas as pd
import numpy as np
import itertools
from functools import partial
from demo_model.tools import flows, dataframe

estimate_import_propensities = None

def _augmented_css(exportness, augmented_flows, countries, sectors):
    """
    Create a DataFrame with every country/country/sector
    augmented with information from augmented_flows and exportness
    """
    # Make a DataFrame with every country/country/sector combination
    ccs_iter = itertools.product(countries, countries, sectors)
    every_ccs = pd.DataFrame([x for x in ccs_iter], columns=['from_country', 'to_country', 'sector'])
    # Augment with all the extra info, like f_is, v_is etc.
    join_cols = ['from_country', 'to_country', 'sector']
    augmented_ccs = every_ccs.join(augmented_flows.set_index(join_cols), on=join_cols)
    augmented_ccs = augmented_ccs.join(exportness, on='from_country').set_index(join_cols)
    return augmented_ccs

def _y_hat(df, coefficients):
    """
    Estimate flow from every from_country, to_country and sector in `df` using
    log(y_ijs) ~ factor(from_country) + mindist + log(x_j) + log(f_is) + log(v_is)
    """
    log = np.log
    c = coefficients
    log_y_ijs = df.exportness + \
            c['mindist'] * df.mindist + \
            c['x_j'] * log(df.x_j) + \
            c['f_is'] * log(df.f_is) + c['v_is'] * log(df.v_is)
    y_ijs = np.exp(log_y_ijs)
    return y_ijs.fillna(0)

def _import_propensities_from_flows(flows_df):
    """
    Calculate y_ijs / sum_i(y_ijs) for every flow in flows_df
    """
    sum_y = flows_df.groupby(level=['to_country', 'sector']).sum()
    ip = dataframe.broadcast(flows_df, sum_y, binary_operator='div')
    # Re-include any sectors dropped due to divide by zero:
    ip = ip.reindex(flows_df.index).fillna(0)
    # Set from RoW propensity to 1 anywhere the total over to_country and sector is zero
    sums = ip.groupby(level=['to_country', 'sector']).sum()
    zero_sums = sums[sums == 0]
    ip.loc[[('RoW', ) + idx for idx in zero_sums.index.values]] = 1
    return ip.reorder_levels(['sector', 'from_country', 'to_country']).sortlevel()

def _estimate_import_propensities(model, coefficients, exportness, augmented_flows):
    """
    Use the values in exportness and the metadata in augmented_flows
    along with the coefficients to estimate y_hat.
    Calculate the import propensities based
    on the y_hat.
    """
    df = _augmented_css(exportness, augmented_flows, countries=model.countries, sectors=model.sectors)
    y_hat = _y_hat(df, coefficients)
    return _import_propensities_from_flows(y_hat)

def estimate_ip_fun(model, coefficients, augmented_flows):
    """
    Return a function which will return a set of import propensities for
    a given set of exportness values, without the need to pass `model`, `coefficients`
    and `augmented_flows` every time.

    Uses functools.partial and _estimate_import_propensities
    """
    kwargs = dict(model=model, coefficients=coefficients, augmented_flows=augmented_flows)
    # Here's the function which does all the hard work:
    return partial(_estimate_import_propensities, **kwargs)

def _get_coefficient(df, term):
    try:
        return df[df.term == 'log(%s)' % term].estimate.values[0]
    except IndexError:
        # log(term) not found in df. Maybe it's not logged?
        return df[df.term == term].estimate.values[0]

def _exportness(regression_df):
    """
    A pd.Series of the country factors from regression_df
    """
    factors = regression_df[regression_df.term.str.startswith('factor')][['term', 'estimate']]
    factors['country'] = factors.term.str[-3:]
    # factor(from_country)
    exportness = factors.set_index('country').estimate
    exportness.name = 'exportness'
    return exportness

def _coefficients(regression_df):
    """
    Extract a dictionary of coefficients from regression_df
    for use in estimating y_ijs in
    log(y_ijs) ~ factor(from_country) + mindist + log(x_i) + log(x_j) + log(f_is) + log(v_is)
    """
    coeff_names = ('mindist', 'x_j', 'f_is', 'v_is')
    coefficients = {x: _get_coefficient(regression_df, x) for x in coeff_names}
    return coefficients

def regression_results(filename):
    """
    Return intercepts, country factors and coefficients
    from the regression results stored in `filename`
    """
    regression_df = pd.read_csv(filename)
    intercept = regression_df.loc[regression_df.term == '(Intercept)', 'estimate'][0]
    exportness = _exportness(regression_df)
    coefficients = _coefficients(regression_df)
    return intercept, exportness, coefficients

def initialise_model_with_exportness(model, regression_filename):
    """
    Use the regression outputs in `regression_filename` to set the import propensities
    of `model`. Create the estimate_import_propensities function.

    Return an array of exportness figures
    """
    global estimate_import_propensities
    augmented_flows = flows.augmented_flows(model)
    intercept, original_exportness, coefficients = regression_results(regression_filename)
    estimate_import_propensities = estimate_ip_fun(model, coefficients, augmented_flows)
    model._import_propensities = estimate_import_propensities(exportness=original_exportness)
    model.recalculate_world()
    model.exportness = original_exportness
    return model


