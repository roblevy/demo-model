# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:12:44 2014

@author: rob
"""

import global_demo_model
import numpy as np

#%%
# Get data
model = global_demo_model.GlobalDemoModel.from_pickle('../Models/model2010.gdm')    
model.set_tolerance(1)

print [model.recalculate_world() for x in range(10)]