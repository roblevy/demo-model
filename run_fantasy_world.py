# -*- coding: utf-8 -*-
"""
TPOE was here (but Rob was here first)
"""

import pandas as pd
import numpy as np
import global_demo_model
import output_model

# Get fantasy data

io_data = pd.read_csv('Fantasy World Data/fantasy_world_data.csv',
                      true_values='t',false_values='f')
goods_flows = pd.read_csv('Fantasy World Data/fantasy_trade_flows.csv',
                          true_values='t',false_values='f')
                                
#%%
# Create models
model_peace = global_demo_model.GlobalDemoModel.from_data(io_data, goods_flows)
model_war = global_demo_model.GlobalDemoModel.from_data(io_data, goods_flows)
#%%
# Run model
model_peace.recalculate_world()
model_peace.to_pickle('../Models/fantasy_model.gdm')

#%%
# Create war
for sector in model_war.import_propensities():
    for country1 in model_war.countries:
        for country2 in model_war.countries:
            if set([country1,country2]) == set(['A','B']):

                model_war.import_propensities() \
                    [sector][country1][country2] = 0
        
        model_war.import_propensities()[sector][:][country1] \
            /= np.sum(model_war.import_propensities()[sector][:][country1])
        
# Run model
model_war.recalculate_world()

models = [model_peace,model_war]
output_model.jsonify_model(models, 'fantasy')

d = model_peace.countries['D']
b = model_peace.countries['B']
c = model_peace.countries['C']

print "d f_dagger:"
print d.f_dagger()

print "d f_star:"
print d.f_star()

print "b f_dagger:"
print b.f_dagger()

print "b f_star:"
print b.f_star()
