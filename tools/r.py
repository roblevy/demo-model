"""
Some functions for interacting with DataFrame objects as
returned by rpy2 functions
"""
import numpy as np
import pandas as pd

def extract_R_df(df):
    return pd.DataFrame({name:np.asarray(df.rx(name))[0] for name in df.names})

def extract_R_number(robj, attr_name):
    return np.array(robj.rx(attr_name))[0][0]
