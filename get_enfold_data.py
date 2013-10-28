# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:18:43 2013

@author: Rob
"""

import pandas.io.sql as psql
import psycopg2 as pg

global __default_sql

__default_sql = """
    SELECT country_iso3, year, from_sector_group_number, 
    from_sector_group as "from", 
    from_sector_is_import as is_import, 
    to_sector_group_number, to_sector_group as to, 
    aggregate_flow_amount as flow_amount
    FROM input_output.vw_aggregate_sector_flows 
    WHERE country_iso3 IN ('DEU', 'GBR', 'FRA', 'USA') 
    AND year = 2008 ORDER BY country_iso3, from_sector_is_import,
    from_sector_group_number, to_sector_group_number;
    """

def get_io_flow_data_from_enfold_db(sql = __default_sql):
    print "Getting data from 128.40.150.11"
    conn = pg.connect("""host='128.40.150.11' 
                      dbname='enfolding' 
                      user='enfoldreader' 
                      password='enfoldreader'""")
    df = psql.read_frame(sql, conn)
    conn.close()
    return df
    
