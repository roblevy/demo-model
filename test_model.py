# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:15:29 2013

@author: Rob
"""

from os import chdir, path
chdir(path.dirname(__file__))

import numpy as np
import dummy_data
import get_enfold_data
import global_demo_model
import output_model

reload(dummy_data)
reload(get_enfold_data)
reload(global_demo_model)
reload(output_model)


# Get data (either dummy or 'real')
io_data = get_enfold_data.get_io_flow_data_from_enfold_db()
goods_flows = dummy_data.load_dummy_trade_data()
services_trade_data = dummy_data.get_dummy_services_trade_data()
                                 
# Create model
model = global_demo_model.GlobalDemoModel(io_data, goods_flows, services_trade_data)

# Test country object    
deu = model.countries['DEU']
i = deu.i
x = deu.x
f = deu.f
e = deu.e
deu.recalculate_economy(f, e) # nothing should happen
print "Test no change when existing f and e are used:"
print np.allclose(i, deu.i) & np.allclose(x, deu.x)
f["AG"] = f["AG"] + 1000
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
gbrx = model.countries['GBR'].x
frax = model.countries['FRA'].x
model.countries['DEU'].f['SS'] = model.countries['DEU'].f['SS'] * 2
model.recalculate_world()
print np.alltrue(model.countries['GBR'].x > gbrx) and np.alltrue(model.countries['FRA'].x > frax)

print "Test that decreasing one country/sector's final demand decreases all sector's output for whole world."
deux = model.countries['DEU'].x
usax = model.countries['USA'].x
usai = model.countries['USA'].i
model.countries['DEU'].f['AG'] = model.countries['DEU'].f['AG'] / 2
model.recalculate_world()
print np.alltrue(model.countries['DEU'].x < deux) and np.alltrue(model.countries['USA'].x < usax)
print np.alltrue(model.countries['USA'].i < usai)