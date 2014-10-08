# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 16:34:24 2014

@author: rob
"""

import pandas as pd
import psycopg2 as pg
from os.path import abspath

def get_connection(user='enfoldreader', password='enfoldreader'):
    return pg.connect(host='128.40.150.11',
        database='enfolding',
        user=user,
        password=password)

def df_from_sql(sql):
    """
    Create a dataframe from the given SQL statement.
    
    The connection used is whatever is returned from
    `get_connection`
    """
    conn = get_connection()
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df

def insert_df_to_database(df, tablename, dbcols=None):
    """
    Write the DataFrame `df` to the Enfolding database.
    
    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The dataframe to write
    tablename: str
        The name (with schema prepended if necessary) of the table to write to
    dbcols: dict
        A dictionary whose keys are columns of the DataFrame `db` to write, and 
        whose values are the names of the columns in the database to write
        each DataFrame column to.
    """
    df = pd.DataFrame(df).reset_index()
    if dbcols is None:
        dbcols = {k:k for k in df} # iterate column names
    pd.io.sql.to_sql(df.rename(dbcols), tablename, 
                     get_connection(), index=False)

def update_df_to_database(df, tablename, 
                          indexcols, dbcols):
    """
    Update `tablename` in the Enfolding database with the DataFrame `df`.
    
    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The dataframe to write
    tablename: str
        The name (with schema prepended if necessary) of the table to write to
    indexcols: dict
        A dictionary whose keys are the name(s) of the (Multi)Index levels
        and whose values are the equivalent columns in the database
    dbcols: dict
        A dictionary whose keys are columns of the DataFrame `db` to write, and 
        whose values are the names of the columns in the database to write
        each DataFrame column to.
    """
    queries = build_update(df=df, tablename=tablename,
                           indexcols=indexcols, dbcols=dbcols)
    pwd = raw_input('Password for user enfolding:')
    con = get_connection(user='enfolding', password=pwd)
    for sql in queries:
        cur = con.cursor()
        cur.execute(sql)
        con.commit()
        cur.close()
        print sql
    con.close()
    
def build_update(df, tablename, indexcols, dbcols):
    df = pd.DataFrame(df).reset_index()
    queries = []
    for k, row in df.iterrows():
        sql = 'UPDATE %s ' % tablename
        sql += build_set(row, dbcols)
        sql += build_where(row, indexcols) + ";"
        queries.append(sql)
    return queries

def build_set(row, dbcols):
    set_ = " SET "
    for dbcol in dbcols:
        value = row[dbcol]
        append = type_sensitive_equals(value)
        set_ += append % (dbcols[dbcol], row[dbcol]) + ", "
    return set_[:-2] # Remove the last comma
    
def build_where(row, indexcols):
    """
    Make a SQL WHERE statement from the DataFrame row `row`.
    """
    where = " WHERE "
    for indexcol in indexcols:
        v = row[indexcol]
        if pd.isnull(v):
            where += "1 = 2 AND " # One of the keys is null!
        else:
            append = type_sensitive_equals(v)
            where += append % (indexcols[indexcol], row[indexcol]) + " AND "
    return where[:-5] # Remove the last " AND "
    
def type_sensitive_equals(v):
    try:
        int_v = int(v)
        # v is coercible as a number
        if int(v) == float(v):
            return '"%s" = %i'
        else:
            return '"%s" = %f'
    except ValueError:
        # v is not a number:
        return """ "%s" = '%s'"""
