"""
Date created: 28 July 2015

@author: rob

Some useful DataFrame/Series functions which don't
seem to have made their way into pandas
"""
import numpy as np
import pandas as pd

def sum_over(df, value_cols, over_column):
    """
    Sum the columns `value_cols` over `over_column`.

    This removes `over_column` and keeps all other columns.
    """
    idx_cols = [x for x in df.columns if x not in value_cols]
    sum_cols = [x for x in idx_cols if x not in over_column]
    return df.set_index(idx_cols).sum(level=sum_cols).reset_index()

def broadcast(x, y, binary_operator='mul', **kwargs):
    """
    Apply a binary operation to x and y, broadcasting across
    the levels of x which are not in y.

    kwargs are passed to the binary_operator
    """
    missing_names = [lvl for lvl in x.index.names if lvl not in y.index.names]
    df = x.unstack(missing_names)
    try:
        y = y.reorder_levels(df.index.names).sortlevel()
    except KeyError:
        raise KeyError("Couldn't match levels of y with levels of x")
    df_fun = getattr(df, binary_operator)
    return df_fun(y, axis=0, **kwargs).stack(missing_names).reorder_levels(x.index.names).sortlevel()
    
def slice(df, other):
    """
    Slice df using the index of other
    """
    not_in = [k for k in df.index.names if k not in other.index.names]
    remaining_idx = df.reset_index().set_index(not_in).index.unique()
    slice_idx = [tuple(x) + y for x in remaining_idx for y in other.index.values]
    return df.loc[slice_idx]

def _col_round(col, places):
    try:
        return np.round(col, places[col.name])
    except KeyError:
        return col

def df_round(df, places):
    """
    Round df to the number(s) specified in `places`.

    `places` is a dict-like with keys being columns and values being number
    of decimal places to round to
    """
    return pd.concat([_col_round(col, places) for k, col in df.iteritems()], axis=1)

def is_in_index(df, to_match, level=None):
    """
    A boolean mask indicating if `to_match` appears in the index of `df`.

    If `level` is None, look in all levels, otherwise, each element of `level`
    will be passed to `df.index.get_level_values()`
    """
    if level is None:
        level = df.index.names
    if isinstance(to_match, basestring):
        to_match = [to_match]
    mask = [df.index.get_level_values(x).isin(to_match) for x in level]
    return np.bitwise_or(*mask)

def filter_pandas(x, filter_on, filter_by, exclude=False):
    """
    Filter a pd.Generic x by `filter_by` on the 
    MultiIndex level or column `filter_on`.
    
    If `exclude` include everything except `filter_by`. Uses
    `pd.Index.get_level_values()` in the background
    """
    if filter_by is None:
        return x
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        index = _filter_mask(x, filter_on, filter_by, exclude)
        return x[index]
    else:
        raise TypeError("Not a pandas object")

def _filter_mask(x, filter_on, filter_by, exclude):
    if isinstance(filter_on, (list, tuple)):
        masks = [_filter_mask(x, field, filter_by, exclude) for field in filter_on]
        return np.logical_and(*masks)

    if isinstance(filter_by, basestring):
        filter_by = [filter_by]
    try:
        index = x.index.get_level_values(filter_on).isin(filter_by)
    except KeyError:
        index = x[filter_on].isin(filter_by)
    if exclude:
        index = ~index
    return index

def set_index_values(df_or_series, new_values, level):
    """
    Replace the MultiIndex level `level` with `new_values`

    `new_values` must be the same length as `df_or_series`
    """
    # TODO Improve how this works
    levels = df_or_series.index.names
    retval = df_or_series.reset_index(level)
    retval[level] = new_values
    retval = retval.set_index(level, append=True).reorder_levels(levels).sortlevel().squeeze()
    return retval

