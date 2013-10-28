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

temp = ''.join([path.dirname(__file__),'/../Fantasy World Data/%s.csv'])

io_data = pd.read_csv(temp % 'fantasy_world_data')
goods_flows = pd.read_csv(temp % 'fantasy_trade_flows')
services_trade_data = pd.read_csv(temp % 'fantasy_services_trade_totals')
                                
# Create models
model_peace = global_demo_model.GlobalDemoModel(io_data, goods_flows)
model_war = global_demo_model.GlobalDemoModel(io_data, goods_flows)


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

usa = model_peace.countries['D']
fra = model_peace.countries['B']
gbr = model_peace.countries['C']

print "USA f_dagger:"
print usa.f_dagger

print "USA f_star:"
print usa.f_star

print "FRA f_dagger:"
print fra.f_dagger

print "FRA f_star:"
print fra.f_star
