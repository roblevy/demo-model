# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:56:17 2013

@author: Rob
"""
import pandas as pd
import import_propensities
reload(import_propensities)

from_iso3 = ['GBR','FRA','USA','FRA']
to_iso3  = ['USA','USA','GBR','GBR']
trade_value = [10,30,5,5]
trade_data = pd.DataFrame({'from_iso3':from_iso3 * 2,'to_iso3':to_iso3 * 2,
                           'sector':['Ag'] * 4 + ['Rm'] * 4,'trade_value':trade_value * 2})

"""
We're expecting two P_i matrices, for Ag and Rm, each looking like this:

    FRA  GBR  RoW  USA
FRA   0  0.5    0 0.75
GBR   0    0    0 0.25
RoW   1    0    1    0
USA   0  0.5    0    0

Note that the columns sum to unity
"""

P = import_propensities.calculate_import_propensities(trade_data, ['GBR','FRA','USA','RoW'])