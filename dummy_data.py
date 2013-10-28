# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:54:58 2013

@author: rob
"""

import pandas as pd
import pandas.io.sql as psql
import psycopg2 as pg
import numpy as np

def load_dummy_data():
    print "Importing dummy data from csv file"
    return pd.read_csv('flow_amounts.csv')
    
def add_dummy_import_propensities(countries):
    """ Pass in the countries list you get from 
    create_countries_from_data """
    country_names = [c.name for c in countries]
    for c in countries:
        np.random.seed(abs(hash(c.name)))
        rand = np.random.uniform(size=len(countries)-1)
        rand = rand / sum(rand)
        other_countries = [country for country in country_names if country != c.name]
        for i, other_country in enumerate(other_countries):
            c.P[other_country] = rand[i-1]

def load_dummy_trade_data():
    print "Importing dummy trade data from csv file"
    return pd.read_csv('..\\Dummy Data\\dummy_trade_flows.csv')
    
def get_dummy_services_trade_data():
    data = pd.read_csv("../Dummy Data/dummy_services_trade_totals2.csv")
    #data = pd.read_csv("..\\Dummy Data\\dummy IPFP data.csv")
    #data = data.sort(['country_iso3','sector','trade_flow'])
    return data

