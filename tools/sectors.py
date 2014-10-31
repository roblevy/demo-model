"""

A set of tools for the Global Demo Model relating to sectors

"""

import pandas as pd

__sectors__  = pd.DataFrame([
    {'name':'Food', "is_services":False, "is_manufacturing":False},
    {'name':'Leather', "is_services":False, "is_manufacturing":False},
    {'name':'Fuel', "is_services":False, "is_manufacturing":False},
    {'name':'Plastics', "is_services":False, "is_manufacturing":True},
    {'name':'Metals', "is_services":False, "is_manufacturing":True},
    {'name':'Machinery', "is_services":False, "is_manufacturing":True},
    {'name':'Vehicles', "is_services":False, "is_manufacturing":True},
    {'name':'Utilities', "is_services":True, "is_manufacturing":False},
    {'name':'Wholesale Trade', "is_services":True, "is_manufacturing":False},
    {'name':'Hospitality', "is_services":True, "is_manufacturing":False},
    {'name':'Air Transport', "is_services":True, "is_manufacturing":False},
    {'name':'Communications', "is_services":True, "is_manufacturing":False},
    {'name':'Business Services', "is_services":True, "is_manufacturing":False},
    {'name':'Education', "is_services":True, "is_manufacturing":False},
    {'name':'Private Households', "is_services":True, "is_manufacturing":False},
    {'name':'Agriculture', "is_services":False, "is_manufacturing":False},
    {'name':'Wood', "is_services":False, "is_manufacturing":True},
    {'name':'Construction', "is_services":False, "is_manufacturing":False},
    {'name':'Real Estate', "is_services":True, "is_manufacturing":False},
    {'name':'Public Services', "is_services":True, "is_manufacturing":False},
    {'name':'Health', "is_services":True, "is_manufacturing":False},
    {'name':'Other Services', "is_services":True, "is_manufacturing":False},
    {'name':'Minerals', "is_services":False, "is_manufacturing":False},
    {'name':'Manufacturing', "is_services":False, "is_manufacturing":True},
    {'name':'Vehicle Trade', "is_services":True, "is_manufacturing":False},
    {'name':'Water Transport', "is_services":True, "is_manufacturing":False},
    {'name':'Transport Services', "is_services":True, "is_manufacturing":False},
    {'name':'Paper', "is_services":False, "is_manufacturing":False},
    {'name':'Electricals', "is_services":False, "is_manufacturing":False},
    {'name':'Textiles', "is_services":False, "is_manufacturing":True},
    {'name':'Retail Trade', "is_services":True, "is_manufacturing":False},
    {'name':'Inland Transport', "is_services":True, "is_manufacturing":False},
    {'name':'Financial Services', "is_services":True, "is_manufacturing":False},
    {'name':'Mining', "is_services":False, "is_manufacturing":False},
    {'name':'Chemicals', "is_services":False, "is_manufacturing":True}
])

def sector_is_services(sector_name):
    """
    Does `sector_name` relate to a services sector?
    
    Returns False if sector_name is not one of the list
    of sectors in `__sector_names__`
    """
    s = __sectors__
    return sector_name in s[s['is_services']].name.values

def sector_is_manufacturing(sector_name):
    """
    Does `sector_name` relate to a manufacturing sector?
    
    Returns False if sector_name is not one of the list
    of sectors in `__sector_names__`
    """
    s = __sectors__
    return sector_name in s[s['is_manufacturing']].name.values
    