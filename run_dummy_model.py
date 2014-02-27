# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:05:01 2014

@author: rob
"""

import pandas as pd
import numpy as np
import global_demo_model
import output_model

reload(global_demo_model)
reload(output_model)

# Get fantasy data

io_data = pd.read_csv('Dummy Data/dummy_io_flows.csv',
                      true_values='t',false_values='f')
goods_flows = pd.read_csv('Dummy Data/dummy_trade_flows.csv',
                          true_values='t',false_values='f')
                                
#%%
# Create model
model = global_demo_model.GlobalDemoModel.from_data(io_data, goods_flows)
