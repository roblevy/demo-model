# -*- coding: utf-8 -*-
"""
Created on Tue Jul 09 14:51:12 2013

@author: Rob
"""

import services_trade
import pandas as pd
reload(services_trade)

year = input("Which year? ")

data = pd.read_csv('../Data/200 Countries/%s/vw_services_totals_%s.csv' % (year,year))

services_flows = services_trade.estimate_services_trade_flows(data)

services_flows.to_csv('../Data/200 Countries/%s/balanced_services_%s.csv' % (year,year),index=False)