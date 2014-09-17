# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:53:41 2013

@author: Rob
"""
import pandas as pd
import country_setup

reload(country_setup)

sector_flows = pd.read_csv('../Data/40 Countries/2008/sector_flows.csv',true_values='t',false_values='f')
trade_flows = pd.read_csv('../Data/200 Countries/2009/fn_trade_flows 2009.csv',true_values='t',false_values='f')
c = country_setup.create_countries_from_data(sector_flows,trade_flows)
row = c['RoW']
