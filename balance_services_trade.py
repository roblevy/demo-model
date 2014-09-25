# -*- coding: utf-8 -*-
"""
Created on Tue Jul 09 14:51:12 2013

@author: Rob
"""

import services_trade
import pandas as pd
reload(services_trade)

def perform_balancing(year):
    data = pd.read_csv('../Data/200 Countries/vw_services_totals_%s.csv'
        % (year))
    services_flows = services_trade.estimate_services_trade_flows(data)    
    services_flows.to_csv('../Data/200 Countries/balanced_services_%s.csv'
        % (year),index=False)    

