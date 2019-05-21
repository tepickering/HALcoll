#!/usr/bin/env python
# coding: utf-8

# In[211]:


import pandas as pd
import pymysql as pymysql

debug = True

start_date = '2019-01-01'
end_date = '2019-02-01'
resample_interval = '5T'

host = "mariadb.mmto.arizona.edu"
database = "measurements"
table = 'bump'
usr = "webuser"
pw = "Clear7Skies!"

conn = pymysql.connect(host=host, port=3306, user=usr, passwd=pw, db=database)


# In[212]:


#  This gets the individual e-series thermocouples parameter names
def get_names(connection):
    df = None
    try:
        # Read data
        with connection.cursor() as cursor:
            sql = f"""SELECT ds_name FROM aaa_parameters WHERE ds_name LIKE 'cell_e_series_tc%'"""
            df = pd.read_sql(sql, conn)
    except Exception as e:
        print(f"Error: {e}")
    return df


# In[213]:


def get_data(connection, param, database, start_date, end_date):
    df = None
    try:
        # Read data
        with connection.cursor() as cursor:
            sql = f"""SELECT from_unixtime(timestamp/1000) as ts, value as {param} FROM {param} 
                  WHERE timestamp >= UNIX_TIMESTAMP('{start_date}') * 1000
                  AND timestamp < UNIX_TIMESTAMP('{end_date}') * 1000;"""
            # Using a median filter to remove spikes.  Modify as you wish.
            df = pd.read_sql(sql, conn, index_col='ts').resample(resample_interval).median()
    except Exception as e:
        print(f"Error: {e}")
    return df


# In[214]:


df_names = get_names(conn)
# df_names


# In[215]:


df_all = None
for (idx, name) in df_names.itertuples():
    if debug:
        print(f"Getting {name} values...")
    df = get_data(conn, name, database, start_date, end_date)
    if df_all is None:
        df_all = df
    else:
        if df is not None:
            df_all = df_all.join(df, how='inner')


# In[216]:


df_all.head()

