# -*- coding: utf-8 -*-
"""
Created on Tue Jul 09 14:51:12 2013

@author: Rob
"""

import services_trade
import pandas as pd
reload(services_trade)

data = pd.read_csv('../Data/200 Countries/2010/vw_services_totals.csv')

balanced_services = services_trade.estimate_services_trade_flows(data)

balanced_services.to_csv('../Data/200 Countries/2010/balanced_services_2010.csv',index=False)