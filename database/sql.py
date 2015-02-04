# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 16:34:24 2014

@author: rob
"""
import numpy as np
import pandas as pd
import psycopg2 as pg
from os.path import abspath

def get_connection(user='enfoldreader', password='enfoldreader'):
    return pg.connect(host='128.40.150.11',
        database='enfolding',
        user=user,
        password=password)

def get_user_connection(user='enfolding'):
    pwd = raw_input('Password for user %s:' % user)
    return get_connection(user=user, password=pwd)

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
    if dbcols is None:
        dbcols = {k:k for k in df} # iterate column names
    sql = "INSERT INTO %s(%s) VALUES\n" % (tablename, ','.join(dbcols.keys()))
    sql += ','.join(['(%s)' % build_values(r, dbcols) for k, r in df.iterrows()])
    con = get_user_connection('enfolding')
    cur = con.cursor()
    cur.execute(sql)
    con.commit()
    cur.close()
    con.close()

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
    con = get_user_connection('enfolding')
    cur = con.cursor()
    for sql in queries:
        cur.execute(sql)
        con.commit()
    cur.close()
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

def build_values(row, dbcols):
    """
    Turn a row of a DataFrame into a comma separated list of values,
    correctly quotes for Postgres
    """
    select = ""
    for dbcol in dbcols:
        value = row[dbcol]
        select += type_sensitive_quoting(value) % value + ", "
    return select[:-2] # Remove the last comma

def build_set(row, dbcols):
    set_ = " SET "
    for dbcol in dbcols:
        value = row[dbcol]
        append = type_sensitive_equals(value)
        set_ += append % (dbcols[dbcol], value) + ", "
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
    return '"%s" = ' + type_sensitive_quoting(v)

def type_sensitive_quoting(v):
    try:
        int_v = int(v)
        # v is coercible as a number
        if int(v) == float(v):
            return '%i'
        else:
            return '%f'
    except ValueError:
        if np.isnan(v):
            # v is null
            return 'NULL /*%s*/'
        else:
            # v is not a number and is not null:
            return "'%s'"
