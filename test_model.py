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
#countries = ['USA','GBR','IND']
countries = pd.unique(sector_flows.country_iso3).tolist()
sector_flows = sector_flows[sector_flows['country_iso3'].isin(countries)]
model = global_demo_model.GlobalDemoModel(sector_flows,trade_flows,services_flows)

model.trade_flows('Agriculture')

# Test country object    
gbr = model.c['GBR']
i = gbr.i.copy()
x = gbr.x.copy()
f = gbr.f.copy()
e = gbr.e.copy()
gbr.recalculate_economy(f, e) # nothing should happen
rtol = 1e-8 # Since numbers are measured in billions, we need a very fine tolerance here!
print "Test no change when existing f and e are used:"
print np.allclose(i, gbr.i,rtol=rtol) & np.allclose(x, gbr.x,rtol=rtol)
f = f.copy()
f["Agriculture"] = f["Agriculture"] + 100000
gbr.recalculate_economy(f, e)
print "Test that when f and e are changed, both x and i change"
print ~(np.allclose(i,gbr.i,rtol=rtol)) & ~(np.allclose(x,gbr.x,rtol=rtol))
print "Test that when f and e are changed, x + i = Ax + f + e"
print np.allclose(gbr.x + gbr.i, np.dot(gbr.A, gbr.x) + gbr.f + gbr.e,rtol=rtol)

# Run the model
model.recalculate_world()
print "Test that imports = exports after recalculating world:"
print np.allclose(model.E.sum(1), model.M.sum(1),rtol=rtol)

print "Test that increasing one country/sector's final demand increases all sector's output for whole world."
gbrx = model.c['GBR'].x
frax = model.c['IND'].x
model.c['GBR'].f['Air Transport'] = model.c['GBR'].f['Air Transport'] * 2
model.recalculate_world()
print np.alltrue(model.c['GBR'].x > gbrx) and np.alltrue(model.c['IND'].x > frax)

print "Test that decreasing one country/sector's final demand decreases all sector's output for whole world."
gbrx = model.c['GBR'].x
usax = model.c['USA'].x
usai = model.c['USA'].i
model.c['GBR'].f['Agriculture'] = model.c['GBR'].f['Agriculture'] / 2
model.recalculate_world()
print np.alltrue(model.c['GBR'].x < gbrx) and np.alltrue(model.c['USA'].x < usax)
print np.alltrue(model.c['USA'].i < usai)