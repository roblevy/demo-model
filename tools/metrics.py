def country_metrics(country):
    """
    Return a (dictionary of) metrics used to 
    evaluate changes to `country`
    """
    # Total Value Added (tva)
    intermediate_usage = country.Z().sum() # Column sum
    value_added = country.x - intermediate_usage
    tva = value_added.sum()
    # Balance of Trade (bot)
    bot = country.e.sum() - country.m.sum()
    return {'tva':tva, 'bot':bot}
