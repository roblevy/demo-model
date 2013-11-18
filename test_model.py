# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:15:29 2013

@author: Rob
"""

from os import chdir, path
chdir(path.dirname(__file__))

import numpy as np
import pandas as pd
import global_demo_model
import output_model

reload(global_demo_model)
reload(output_model)


# Get data (either dummy or 'real')
sector_flows = pd.read_csv('../Data/40 Countries/2008/sector_flows.csv',true_values='t',false_values='f')
trade_flows = pd.read_csv('../Data/200 Countries/2009/fn_trade_flows 2009.csv',true_values='t',false_values='f')
services_flows = pd.read_csv('../Data/200 Countries/2010/balanced_services_2010.csv')
                                 
# Create model
countries = ['USA','GBR','IND']
sector_flows = sector_flows[sector_flows['country_iso3'].isin(countries)]
model = global_demo_model.GlobalDemoModel(sector_flows,trade_flows,services_flows)

# Test country object    
deu = model.c['DEU']
i = deu.i
x = deu.x
f = deu.f
e = deu.e
deu.recalculate_economy(f, e) # nothing should happen
print "Test no change when existing f and e are used:"
print np.allclose(i, deu.i) & np.allclose(x, deu.x)
f["Agriculture"] = f["Agriculture"] + 1000
deu.recalculate_economy(f, e)
print "Test that when f and e are changed, both x and i change"
print ~(np.allclose(i,deu.i)) & ~(np.allclose(x,deu.x))
print "Test that when f and e are changed, x + i = Ax + f + e"
print np.allclose(deu.x + deu.i, np.dot(deu.A, deu.x) + deu.f + deu.e)

# Run the model
model.recalculate_world()
print "Test that imports = exports after recalculating world:"
print np.allclose(model.E.sum(1), model.M.sum(1))

print "Test that increasing one country/sector's final demand increases all sector's output for whole world."
gbrx = model.c['GBR'].x
frax = model.c['FRA'].x
model.c['DEU'].f['Air Transport'] = model.c['DEU'].f['Air Transport'] * 2
model.recalculate_world()
print np.alltrue(model.c['GBR'].x > gbrx) and np.alltrue(model.c['FRA'].x > frax)

print "Test that decreasing one country/sector's final demand decreases all sector's output for whole world."
deux = model.c['DEU'].x
usax = model.c['USA'].x
usai = model.c['USA'].i
model.c['DEU'].f['Agriculture'] = model.c['DEU'].f['Agriculture'] / 2
model.recalculate_world()
print np.alltrue(model.c['DEU'].x < deux) and np.alltrue(model.c['USA'].x < usax)
print np.alltrue(model.c['USA'].i < usai)