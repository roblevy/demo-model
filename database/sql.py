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
    sql += _build_values(df, dbcols)
    with get_user_connection('enfolding') as con:
        with con.cursor() as cur:
            cur.execute(sql)
            con.commit()

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
    sql = build_update(df=df, tablename=tablename,
                           indexcols=indexcols, dbcols=dbcols)
    with get_user_connection('enfolding') as con:
        with con.cursor() as cur:
            cur.execute(sql)
            con.commit()
    
def build_update(df, tablename, indexcols, dbcols):
    all_cols = indexcols.copy()
    all_cols.update(dbcols)
    df = pd.DataFrame(df).reset_index()
    sql = 'update %s as to_update' % tablename
    set_sql = 'set %s' % ', '.join(['"%s" = other."%s"' % (v, k) for k, v in dbcols.iteritems()])
    values_sql = _build_values(df, all_cols)
    as_sql = _build_as('other', all_cols)
    from_sql = "from (values %s) %s" % (values_sql, as_sql)
    where_sql = _build_update_where('to_update', 'other', indexcols)
    return '\n'.join([sql, set_sql, from_sql, where_sql])

def _value_row(row, columns):
    """
    Turn a DataFrame row into a string of the form (col1, col2...)
    with the values appropriately quoted
    """
    return '(%s)' % ', '.join([type_sensitive_quoting(row[k]) % row[k] for k in columns])

def _build_values(df, columns):
    """
    A comma separated list of `_value_row`s, one for each row of
    `df`
    """
    return ',\n'.join(df.apply(_value_row, columns=columns, axis=1))

def _build_as(tbl_name, columns):
    as_string = ",".join('"%s"' % k for k in columns)
    return "AS %s(%s)" % (tbl_name, as_string)

def _build_update_where(update_tbl, values_tbl, columns):
    """
    return a where statement for each entry in columns
    """
    clauses = ['%s."%s" = %s."%s"' % (values_tbl, k, update_tbl, v) for k, v in columns.iteritems()]
    where = ' AND '.join(clauses)
    return "WHERE %s;" % where

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
        try:
            if np.isnan(v):
                # v is null
                return 'NULL /*%s*/'
        except TypeError:
            # v is not a number and is not null:
            # $$ is an escape-hell free way to write strings
            # in Postgres
            return "$string$%s$string$"
