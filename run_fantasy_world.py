# -*- coding: utf-8 -*-
"""
TPOE was here (but Rob was here first)
"""

from os import chdir, path
chdir(path.dirname(__file__))

import pandas as pd
import numpy as np
import global_demo_model
import output_model

reload(global_demo_model)
reload(output_model)

# Get fantasy data

fname = ''.join([path.dirname(__file__),'/Fantasy World Data/%s.csv'])

io_data = pd.read_csv(fname % 'fantasy_world_data',
                      true_values='t',false_values='f')
goods_flows = pd.read_csv(fname % 'fantasy_trade_flows',
                          true_values='t',false_values='f')
                                
# Create models
model_peace = global_demo_model.GlobalDemoModel(io_data, goods_flows, None)
model_war = global_demo_model.GlobalDemoModel(io_data, goods_flows, None)

# Run model
model_peace.recalculate_world()

# Create war
for sector in model_war.P:
    for country1 in model_war.countries:
        for country2 in model_war.countries:
            if set([country1,country2]) == set(['A','B']):

                model_war.P[sector][country1][country2] = 0
        
        model_war.P[sector][:][country1] /= np.sum(model_war.P[sector][:][country1])
        
# Run model
model_war.recalculate_world()

models = [model_peace,model_war]
output_model.jsonify_model(models, 'fantasy')

d = model_peace.c['D']
b = model_peace.c['B']
c = model_peace.c['C']

print "d f_dagger:"
print d.f_dagger

print "d f_star:"
print d.f_star

print "b f_dagger:"
print b.f_dagger

print "b f_star:"
print b.f_star
