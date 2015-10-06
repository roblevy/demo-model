"""

A set of tools for the Global Demo Model relating to sectors

"""

import pandas as pd
from demo_model.tools import dataframe

__sectors__  = pd.DataFrame([
    {'name':'Food', "is_services":False, "is_manufacturing":False, "is_commodity":True},
    {'name':'Leather', "is_services":False, "is_manufacturing":False, "is_commodity":True},
    {'name':'Fuel', "is_services":False, "is_manufacturing":False, "is_commodity":True},
    {'name':'Plastics', "is_services":False, "is_manufacturing":True, "is_commodity":True},
    {'name':'Metals', "is_services":False, "is_manufacturing":True, "is_commodity":True},
    {'name':'Machinery', "is_services":False, "is_manufacturing":True, "is_commodity":True},
    {'name':'Vehicles', "is_services":False, "is_manufacturing":True, "is_commodity":True},
    {'name':'Utilities', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Wholesale Trade', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Hospitality', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Air Transport', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Communications', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Business Services', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Education', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Private Households', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Agriculture', "is_services":False, "is_manufacturing":False, "is_commodity":True},
    {'name':'Wood', "is_services":False, "is_manufacturing":True, "is_commodity":True},
    {'name':'Construction', "is_services":False, "is_manufacturing":False, "is_commodity":False},
    {'name':'Real Estate', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Public Services', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Health', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Other Services', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Minerals', "is_services":False, "is_manufacturing":False, "is_commodity":True},
    {'name':'Manufacturing', "is_services":False, "is_manufacturing":True, "is_commodity":True},
    {'name':'Vehicle Trade', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Water Transport', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Transport Services', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Paper', "is_services":False, "is_manufacturing":False, "is_commodity":True},
    {'name':'Electricals', "is_services":False, "is_manufacturing":False, "is_commodity":True},
    {'name':'Textiles', "is_services":False, "is_manufacturing":True, "is_commodity":True},
    {'name':'Retail Trade', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Inland Transport', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Financial Services', "is_services":True, "is_manufacturing":False, "is_commodity":False},
    {'name':'Mining', "is_services":False, "is_manufacturing":False, "is_commodity":True},
    {'name':'Chemicals', "is_services":False, "is_manufacturing":True, "is_commodity":True}
])

__super_sectors__ = {
"Inland Transport":"transport", "Water Transport":"transport", "Air Transport":"transport",
"Transport Services":"transport",
"Agriculture":"primary", "Food":"primary", "Leather":"primary", "Paper":"primary", "Textiles":"primary",
"Chemicals":"secondary", "Electricals":"secondary", "Plastics":"secondary",
"Manufacturing":"secondary", "Machinery":"secondary", "Vehicles":"secondary",
"Wood":"raw", "Mining":"raw", "Metals":"raw", "Fuel":"raw", "Minerals":"raw",
"Vehicle Trade":"trade", "Wholesale Trade":"trade", "Retail Trade":"trade", "Real Estate":"trade",
"Business Services":"services", "Financial Services":"services",
"Hospitality":"services", "Private Households":"services",
"Education":"public", "Health":"public", "Other Services":"public", "Communications":"public",
"Construction":"public", "Public Services":"public", "Utilities":"public"
}

def manufacturing_sectors():
    return _services_with_condition('is_manufacturing')

def commodity_sectors():
    return _services_with_condition('is_commodity')

def services_sectors():
    return _services_with_condition('is_services')

def _services_with_condition(condition_name):
    s = __sectors__
    return s[s[condition_name]].name.tolist()

def sector_is_services(sector_name):
    """
    Does `sector_name` relate to a services sector?
    
    Returns False if sector_name is not one of the list
    of sectors in `__sector_names__`
    """
    return _sector_is(sector_name, 'is_services')

def sector_is_manufacturing(sector_name):
    """
    Does `sector_name` relate to a manufacturing sector?
    
    Returns False if sector_name is not one of the list
    of sectors in `__sector_names__`
    """
    return _sector_is(sector_name, 'is_manufacturing')
    
def sector_is_commodity(sector_name):
    """
    Does `sector_name` relate to a commodity sector?
    
    Returns False if sector_name is not one of the list
    of sectors in `__sector_names__`
    """
    return _sector_is(sector_name, 'is_commodity')
    
def _sector_is(sector_name, attr):
    s = __sectors__
    s = s[s[attr]].name.tolist()
    try:
      return sector_name.isin(s)
    except AttributeError:
      return sector_name in s

def format_sector_name(sector_name):
    """
    A database-friendly version of `sector_name`

    Puts the name into all lower case and replaces
    whitespace with `_`
    """
    return sector_name.replace(' ', '_').lower()

def aggregate_to_supersectors(series, level_name='sector'):
    """
    Sum values of `series` across the `level_name` index level
    according to super sector
    """
    idx_vals = series.index.get_level_values(level_name)
    new_idx = [__super_sectors__[s] for s in idx_vals]
    temp = dataframe.set_index_values(series, new_idx, level_name)
    return temp.groupby(level=temp.index.names).sum()
    
