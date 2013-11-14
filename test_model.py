# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:15:29 2013

@author: Rob
"""

from os import chdir, path
chdir(path.dirname(__file__))

import numpy as np
import pandas as pd
import dummy_data
import get_enfold_data
import global_demo_model
import output_model

reload(dummy_data)
reload(get_enfold_data)
reload(global_demo_model)
reload(output_model)


# Get data (either dummy or 'real')
sector_flows = pd.read_csv('../Data/40 Countries/2008/sector_flows.csv',true_values='t',false_values='f')
trade_flows = pd.read_csv('../Data/200 Countries/2009/fn_trade_flows 2009.csv',true_values='t',false_values='f')
services_flows = pd.read_csv('../Data/200 Countries/2010/balanced_services_2010.csv')
                                 
# Create model
model = global_demo_model.GlobalDemoModel(sector_flows,trade_flows,services_flows)

## Test country object    
#deu = model.countries['DEU']
#i = deu.i
#x = deu.x
#f = deu.f
#e = deu.e
#deu.recalculate_economy(f, e) # nothing should happen
#print "Test no change when existing f and e are used:"
#print np.allclose(i, deu.i) & np.allclose(x, deu.x)
#f["AG"] = f["AG"] + 1000
#deu.recalculate_economy(f, e)
#print "Test that when f and e are changed, both x and i change"
#print ~(np.allclose(i,deu.i)) & ~(np.allclose(x,deu.x))
#print "Test that when f and e are changed, x + i = Ax + f + e"
#print np.allclose(deu.x + deu.i, np.dot(deu.A, deu.x) + deu.f + deu.e)
#
## Run the model
#model.recalculate_world()
#print "Test that imports = exports after recalculating world:"
#print np.allclose(model.E.sum(1), model.M.sum(1))
#
#print "Test that increasing one country/sector's final demand increases all sector's output for whole world."
#gbrx = model.countries['GBR'].x
#frax = model.countries['FRA'].x
#model.countries['DEU'].f['Air Transport'] = model.countries['Air Transport'].f['SS'] * 2
#model.recalculate_world()
#print np.alltrue(model.countries['GBR'].x > gbrx) and np.alltrue(model.countries['FRA'].x > frax)
#
#print "Test that decreasing one country/sector's final demand decreases all sector's output for whole world."
#deux = model.countries['DEU'].x
#usax = model.countries['USA'].x
#usai = model.countries['USA'].i
#model.countries['DEU'].f['Agriculture'] = model.countries['DEU'].f['Agriculture'] / 2
#model.recalculate_world()
#print np.alltrue(model.countries['DEU'].x < deux) and np.alltrue(model.countries['USA'].x < usax)
#print np.alltrue(model.countries['USA'].i < usai)