# =========================================================================== #
#                                 DESCRIPTION                                 #
# =========================================================================== #
'''Functions for providing descriptive statistics on variables and dataframes.'''

# %%
# --------------------------------------------------------------------------- #
#                                 LIBRARIES                                   #
# --------------------------------------------------------------------------- #
import os
import sys
import inspect

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import kurtosis, skew

# ---------------------------------------------------------------------------- #
#                                    DESCRIBE                                  #
# ---------------------------------------------------------------------------- #


def describe_qual_df(df):
    s = pd.DataFrame()
    df = df.select_dtypes(include='object')
    cols = df.columns
    for col in cols:
        d = pd.DataFrame(df[col].describe())
        d = d.T
        d['missing'] = df[col].isna().sum()
        s = s.append(d)
    return s


def describe_quant_df(df, sig=0.05, verbose=True):

    if verbose:
        k = ['count', 'missing', 'min', '25%', 'mean', '50%',
             '75%', 'max', 'sd', 'skew', 'kurtosis', 'normality_p', 'normality']
    else:
        k = ['count', 'min', '25%', 'mean', '50%', '75%', 'max']

    s = pd.DataFrame()
    df = df.select_dtypes(include=['float64', 'int64', 'int'])
    cols = df.columns
    for col in cols:
        d = pd.DataFrame(df[col].describe())
        d = d.T
        d['sd'] = np.std(df[col].notnull())
        d['skew'] = skew(df[col].notnull())
        d['kurtosis'] = kurtosis(df[col].notnull())
        d['missing'] = df[col].isna().sum()
        _, d['normality_p'] = stats.shapiro(df[col].notnull())
        d['normality'] = np.where(
            d['normality_p'] < sig, "Reject H0", "Fail to Reject H0")        
        s = s.append(d[k])
    return s


def describe_qual_x(x):
    d = pd.DataFrame(x.value_counts())
    d = d.T
    d['missing'] = x.isna().sum()
    if (len(d.columns) > 10):
        d = d.T
    return d


def describe_quant_x(x, sig=0.05, verbose=True):
    d = pd.DataFrame(x.describe())
    d = d.T
    d['sd'] = np.std(x.notnull())
    d['skew'] = skew(x.notnull())
    d['kurtosis'] = kurtosis(x.notnull())
    d['missing'] = x.isna().sum()
    _, d['normality_p'] = stats.shapiro(x.notnull())
    d['normality'] = np.where(d['normality_p'] < sig,
                              "Reject H0", "Fail to Reject H0")
    if verbose:
        k = ['count', 'missing', 'min', '25%', 'mean', '50%',
            '75%', 'max', 'sd', 'skew', 'kurtosis', 'normality_p', 'normality']
    else:
        k = ['count', 'min', '25%', 'mean', '50%', '75%', 'max']    

    return d[k]

# %%


def group_describe(df, x, y, z=None):
    '''
    Splits a dataframe along values of the categorical variable (y) and returns
    descriptive statistics by level

    Args:
        df (pd.DataFrame): Dataframe containing data
        x (str): The name of the categorical variable
        y (str): The name of the numeric variable
        z (str): The name of an optional categorical variable
    '''
    df2 = pd.DataFrame()
    if z:
        gb = df[[x, y, z]].groupby([x, z])
        groups = [gb.get_group(x) for x in gb.groups]
        for g in groups:
            d = describe_quant_x(g[y])
            d[x] = g[x].unique()
            d[z] = g[z].unique()
            df2 = df2.append(d)
    else:
        cat = x
        num = y
        if df[x].dtype == np.float64 or df[x].dtype == np.int64:
            num = x
            cat = y
        gb = df[[num, cat]].groupby([cat])
        groups = [gb.get_group(x) for x in gb.groups]
        for g in groups:
            d = describe_quant_x(g[num])
            d[y] = g[cat].unique()
            df2 = df2.append(d)
    cols = df2.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    cols = cols[-1:] + cols[:-1]
    return(df2[cols])
