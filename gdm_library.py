# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 13:28:42 2013

@author: Rob
"""

import pandas as pd
import numpy as np

def diagonalise(x):
    return pd.DataFrame(np.diag(x.squeeze()), index=x.index, columns=x.index)

def limit_data(data, limit_field, limit_value, not_equals=False):
    
    if not_equals:
        data = data[data[limit_field] != limit_value]
    else:
        data = data[data[limit_field] == limit_value]

    return data         
