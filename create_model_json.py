# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 13:58:00 2014

@author: Rob
"""
import run_model
import pprint

model = run_model.model

countries = sorted(["USA", "GBR", "DEU", "FRA", "CHN", "IND"])
sectors = sorted(["Agriculture", "Mining", "Food", "Wood", "Vehicles"])
flows = model.trade_flows(countries, sectors)

flow_json = model.flows_to_json(flows)

with open("model.json", "w") as f:
    f.write(pprint.pformat(flow_json))