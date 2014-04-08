# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:05:01 2014

@author: rob
"""

import pandas as pd
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
model = global_demo_model.GlobalDemoModel.from_data(io_data, 
                                                    goods_flows,
                                                    tolerance=1e-5)


flows = model.flows(None, None)
flows = flows.unstack(level=['to_country', 'to_sector'])

#%% Do some testing

c = model.countries['B']
s = 'R'
production_flows = flows.ix[c.name, s]
tot_flows = production_flows.sum()
m = c.m[s]
e = c.e[s]
z_dag = c.Z_dagger().ix[s]
z_star = c.Z_star().ix[s]
f_domestic = c.f[s] * (1 - c.d[s])
f_foreign = (model.final_demand() * 
    model.import_ratios()).sum(level='sector')[s]
d = c.d[s]

print 'e + z_dag.sum() + (1 - d) * f: %s' % (e + z_dag.sum() + (1 - d) * f_domestic)
print 'Total Prod %s' % c.x[s]