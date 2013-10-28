# -*- coding: utf-8 -*-
"""
Created on Tue Jul 09 14:51:12 2013

@author: Rob
"""

from dummy_data import *
from services_trade import *



data = get_dummy_services_trade_data()

[scaled_data, trade_flows] = estimate_services_trade_flows(data)

total_balanced_imports = trade_flows.groupby(['to','sector']).sum()
total_balanced_exports = trade_flows.groupby(['from','sector']).sum()
flow_matrix = trade_flows.pivot(index='from', columns='to', values='flow_value')
print scaled_data
print trade_flows
#print "Total balanced imports: " 
#print total_balanced_imports
#print "Total balanced exports:"
#print total_balanced_exports
