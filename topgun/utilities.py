#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HongXiongMao utility functions

"""

#import hongxiongmao.config as config        # API Keys

# Dependencies
import numpy as np
import pandas as pd
import datetime as dt


# %% Rebase Indices

def rtn2px(df, rebase=100, freq=12, method='percent'):
    """ Convert returns to an index stream
    
    INPUTS:
        rebase - index start price
        freq - smoothing for example monthly YoY CPI
        method - number|percentage(default)
    
    """
    
    df = df.to_frame() if isinstance(df, pd.Series) else df    # ensure df
    
    if method == 'number':
        df = 1 + (df / (100 * freq))    # divide no return by 100
    else:
        df = 1 + (df / freq)            # default assumes percentage input
    
    # Find index of first non-nan in each column
    idx = df.notna().idxmax()
    for i, c in enumerate(df):
        
        # add rebase number to last nan (if available) or 1st number
        ix = df.index.get_loc(idx[0]) 
        ix = ix - 1 if ix != 0 else 0
        df.iloc[ix, i] = rebase
    
    return df.cumprod()


def rebase_index(df, rebase=100, method='log'):
    """ Align one (or more) index series and rebase
    
    methods:
        log is continuously compounded returns [ln(t1 / t0)]
        pct_change (default) 
    
    """
    
    df = df.to_frame() if isinstance(df, pd.Series) else df    # ensure df
    df.dropna(inplace=True)
    
    if method == 'log':
        df = 1 + np.log(df / df.shift(1))
    else:
        df = 1 + df.pct_change(1)
    
    df.iloc[0,:] = rebase
    return df.cumprod()

# %% DATAFRAME MERGER

# Dataframe Merger
def df_merger(a, b, blend='left'):
    """
    Merge two Pandas timeseries dataframes with inconsistent data for e.g.
    different time periods or different asset classes and return single
    dataframe with unique columns and all periods from both input dfs.
    
    Very useful for updating timeseries data where we want to keep the
    original data, but add new data. 
    
    blend can be 'left'(default), 'right' or 'mean'
        left - preserves data from a & appends from b; quick update
        right - preserves data from b & appends from a; for point in time
        mean - takes mean of (non NaN) data where there is a difference
    """
    # Concat along rows to keep all indices and columns from a & b
    # Groupby column names and apply a homemade sorting function
    # Groupby will keep duplicated columns with the same column name
    # Remove these duplicated columns
    c = pd.concat([a, b], axis=1, sort=True)
    df = c.groupby(c.columns, axis=1, sort=False).apply(
            lambda x: _df_merger_helper_func(x, blend))
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df

def _df_merger_helper_func(x, blend='mean'):
    """ Function is part of df_merger in Pandas, i.e. updating a timeseries 
    We pass a df x, as a result of a pd.DataFrame.groupby()
    For example x[:,0] is SPX from 1999-2017 & x[:,1] is 2016-2019
    Here we output a df with a single column that blends the dataseries
    """
    if x.shape[1] == 1:
        # 1st test shape, width==1 => series therefore return itself
        return x
    elif (x.shape[1] == 2) and (blend in ['left', 'right']):        
        # We chose which is the primary (called left) & which is the updater
        # Subset the "left" column of data, then find the index of NaNs
        # Replace nans on left with values from the "right" (could be NaN)
        # Always same length because concat func creates NaNs on missing data
        v = [0,1] if blend == 'left' else [1,0]
        l = x.iloc[:,v[0]]
        i = np.isnan(l)
        l[i] = x[i].iloc[:,v[1]]
        return pd.DataFrame(l, columns=[x.columns.values[0]])
    else:
        # Default otherwise is to return the mean
        # For mean we take the mean of x (which outputs a series)
        # Then convert back to dataframe & return the original column header
        return pd.DataFrame(x.mean(axis=1), columns=[x.columns.values[0]])

# %% DATE FUNCTIONS
        
def relative_date(r='12m', end_date='today', date_format='%Y-%m-%d',
                  as_string=False, unixtimestamp=False):
    """
    Relative Date function
    
    Calculates a datetime from a given end date and a relative reference.
    
    INPUT:
        r - relative date reference as '-12d' accepts d, w, m or y
        end_date - 'today' (default), date string, datetime object
        date_format - input format of string & output if requested
        as_string - True | False (default) 
                    decides if output is converted to string from datetime
        unixtimestamp - converts datetime to an INTEGER unixtimestamp
    """
    
    # Create Datetime object end_date based on supplied end_date
    # If not string or 'today' assume already in datetime format
    if end_date == 'today':
        end_date = dt.datetime.today()        
    elif isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, date_format)
        
    # Breakdown Relative Reference into type (i.e. d, w, m, y) & number
    r = r[1::] if r[0] == '-' else r
    dtype, dnum = str(r[-1]).lower(), float(r[0:-1])
    
    # Manipulate based on relative Days, Weeks, Months or Years
    if   dtype == 'd': start_date = end_date - dt.timedelta(days=dnum)
    elif dtype == 'w': start_date = end_date - dt.timedelta(weeks=dnum)
    elif dtype == 'm': start_date = end_date - dt.timedelta(weeks=dnum*4)
    elif dtype == 'y': start_date = end_date - dt.timedelta(weeks=dnum*52.143)
        
    # Output as Strings if desirable
    if as_string is True:
        start_date = dt.datetime.strftime(start_date, date_format)
        end_date = dt.datetime.strftime(end_date, date_format)
    elif unixtimestamp is True:
        start_date = int(dt.datetime.timestamp(start_date))
        end_date = int(dt.datetime.timestamp(end_date))
    
    return start_date, end_date

# Daily-to-Weekly
def daily2weekly(ts, day=4, date_str_format='%Y-%m-%d'):
    """
    daily2weekly
    
    Converts timeseries Dataframe from daily to weekly data.
    Defaulting to Friday(day=4) but others days possible (Monday=0)
    """
    dt = pd.to_datetime(ts.index, format=date_str_format).to_series().dt.dayofweek
    idx = dt.values == day    # Reindex using desired day of week
    new = ts.iloc[idx] if isinstance(ts, pd.Series) else ts.iloc[idx,:]
    return new