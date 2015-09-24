import pandas as pd

def value_added(country):
    """
    Calculate the value added of `country`
    """
    try:
        intermediate_usage = country.Z().sum() # Column sum
        value_added = country.x - intermediate_usage
    except AttributeError:
        return 0
    return value_added

def country_metrics(country):
    """
    Return a (dictionary of) metrics used to 
    evaluate changes to `country`
    """
    # Total Value Added (tva)
    tva = value_added(country).sum()
    # Balance of Trade (bot)
    try:
        bot = country.e.sum() - country.m.sum()
    except AttributeError:
        bot = 0
    return {'tva':tva, 'bot':bot}

def all_metrics(model, include_RoW=False):
    """
    Return a DataFrame of metrics for each country in the model
    """
    if include_RoW:
        countries = [c.name for k, c in model.countries.iteritems()]
    else:
        countries = model.country_names
    metrics = {c: country_metrics(model.countries[c]) for c in countries}
    return pd.DataFrame(metrics).transpose()
