"""
Created on Mon Apr 07 14:16:10 2014

@author: Rob
"""
import pandas as pd
import global_demo_model
reload(global_demo_model)
sector_flows = pd.read_csv('../Data/sector_flows/vw_sector_flows_2011.csv',true_values='t',false_values='f')
trade_flows = pd.read_csv('../Data/200 Countries/fn_trade_flows_2011.csv',true_values='t',false_values='f')
services_flows = pd.read_csv('../Data/200 Countries/balanced_services_2011.csv')
model = global_demo_model. \
    GlobalDemoModel.from_data(sector_flows,
                              trade_flows,
                              services_flows, 
                              silent=False, tolerance=1, year=2010)
