# -*- coding: utf-8 -*-
"""
Created on Tue Jul 09 14:51:12 2013

@author: Rob
"""

import services_trade
import pandas as pd
reload(services_trade)

data = pd.read_csv('../Data/40 Countries/2010/vw_services_totals.csv')

balanced_services = services_trade.estimate_services_trade_flows(data)

#total_balanced_imports = trade_flows.groupby(['to','sector']).sum()
#total_balanced_exports = trade_flows.groupby(['from','sector']).sum()
#flow_matrix = trade_flows.pivot(index='from', columns='to', values='flow_value')
#print scaled_data
#print trade_flows
#print "Total balanced imports: " 
#print total_balanced_imports
#print "Total balanced exports:"
#print total_balanced_exports
