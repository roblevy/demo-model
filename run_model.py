# -*- coding: utf-8 -*-
"""
Created on Mon Apr 07 14:16:10 2014

@author: Rob
"""
import pandas as pd
import global_demo_model
reload(global_demo_model)
sector_flows = pd.read_csv('../Data/40 Countries/2010/vw_sector_flows.csv',true_values='t',false_values='f')
trade_flows = pd.read_csv('../Data/200 Countries/2010/fn_trade_flows_2010.csv',true_values='t',false_values='f')
services_flows = pd.read_csv('../Data/200 Countries/2010/balanced_services_2010.csv')
model = global_demo_model. \
    GlobalDemoModel.from_data(sector_flows,
                              trade_flows,
                              services_flows, 
                              silent=False, tolerance=1)
model.to_file('../Models/model2010.gdm')