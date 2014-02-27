# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:12:44 2014

@author: rob
"""

import global_demo_model
reload(global_demo_model)

#%%
# Get data
#model = global_demo_model.GlobalDemoModel.from_pickle('Dummy Data/dummy-model.gdm')
model = global_demo_model.GlobalDemoModel.from_pickle('model.gdm')
model.set_tolerance(.001)
model.recalculate_world()