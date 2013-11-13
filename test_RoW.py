# -*- coding: utf-8 -*-

import pandas as pd
import country_setup

reload(country_setup)

commodity_flows = pd.read_csv('../Data/200 Countries/2009/fn_trade_flows 2009.csv')
services_flows = pd.read_csv('../Data/200 Countries/2010/balanced_services_2010.csv')

trade_flows = commodity_flows.append(services_flows, ignore_index=True)
trade_flows = trade_flows[['from_iso3','to_iso3','sector','trade_value']]

country_setup._create_RoW_country(trade_flows, ['GBR','USA','DEU'], ['Agriculture','Business Services'])

