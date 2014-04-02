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
#countries = pd.unique(sector_flows.country_iso3).tolist()
#sector_flows = sector_flows[sector_flows['country_iso3'].isin(countries)]
model = global_demo_model.GlobalDemoModel.from_data(sector_flows,
                                                    trade_flows,
                                                    services_flows)

#%%
# Test country object
gbr = model.countries['GBR']
m = gbr.m.copy()
x = gbr.x.copy()
f = gbr.f.copy()
e = gbr.e.copy()
gbr.recalculate_economy(final_demand=f,
                        exports=e) # nothing should happen
print "Test no change when existing f and e are used:"
print np.allclose(m, gbr.m,rtol=model.tolerance) & \
    np.allclose(x, gbr.x,rtol=1e-8)
f = f.copy()
f["Agriculture"] = f["Agriculture"] + 100000
gbr.recalculate_economy(final_demand=f,
                        exports=e)
print "Test that when f and e are changed, both x and m change"
print ~(np.allclose(m,gbr.m,rtol=1e-8)) & \
    ~(np.allclose(x,gbr.x,rtol=1e-8))
print "Test that when f and e are changed, x + m = Ax + f + e"
print np.allclose(gbr.x + gbr.m, np.dot(gbr.A, gbr.x) + \
    gbr.f + gbr.e,rtol=1e-8)

# Run the model
model.set_tolerance(1e-2)
model.recalculate_world()
print "Test that imports = exports after recalculating world:"
print np.allclose(model.exports.sum(1), model.imports.sum(1),rtol=model.tolerance)

print "Test that increasing one country/sector's final demand " \
    "increases all sector's output for whole world."
gbrx = gbr.x
indx = model.countries['IND'].x
f_air_transport = gbr.f['Air Transport'] * 2
model.set_final_demand(gbr,'Air Transport', f_air_transport)
print np.alltrue(gbr.x > gbrx) and \
    np.alltrue(model.countries['IND'].x > indx)

print "Test that decreasing one country/sector's final demand " \
    "decreases all sector's output for whole world."
gbrx = gbr.x
usax = model.countries['USA'].x
usai = model.countries['USA'].m
model.set_final_demand('GBR','Agriculture',
                       gbr.f['Agriculture'] / 2)
print np.alltrue(gbr.x < gbrx) and np.alltrue(model.countries['USA'].x < usax)
print np.alltrue(model.countries['USA'].m < usai)

model.to_file('model.gdm')