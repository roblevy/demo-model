# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 13:58:00 2014

@author: Rob
"""
import run_model
import pprint

model = run_model.model

#countries = ["USA", "GBR", "DEU", "FRA", "CHN", "IND"]
#sectors = ["Agriculture", "Mining", "Food", "Wood", "Vehicles"]
#countries = sorted(["USA", "GBR"])
#sectors = sorted(["Agriculture", "Food"])
flows = model.trade_flows()

flow_json = model.flows_to_json(flows)

with open("model.json", "w") as f:
    f.write("json=")
    f.write(pprint.pformat(flow_json))