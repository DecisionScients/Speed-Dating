# ---------------------------------------------------------------------------- #
#                                 ANALYSIS                                     #
# ---------------------------------------------------------------------------- #
"""Analysis Module.

Module contains classes used for exploratory data analysis. The Describe class
extends the pandas.describe method with measurements of the shape of the 
distribution, e.g. skew, kurtosis and normality test.

"""
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import kurtosis, skew

class Describe():
    """ Class contains methods that provide summary statistics for DataFrames"""

    def __init__(self, df):
        self.df = df
    
    def quant(self, cols=None, sig=0.05):
        
        k = ['count', 'missing', 'min', '25%', 'mean', '50%',
            '75%', 'max', 'sd', 'skew', 'kurtosis', 'normality_p', 'normality']        
        desc = pd.DataFrame()
        df = self.df.select_dtypes(include=['float64', 'int64', 'int'])
        if cols is None:
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
            desc = desc.append(d[k])
        return desc

    def qual(self, cols=None):
        desc = pd.DataFrame()
        df = self.df.select_dtypes(include='object')
        if cols is None:
            cols = df.columns
        for col in cols:
            d = pd.DataFrame(df[col].describe())
            d = d.T
            d['missing'] = df[col].isna().sum()
            desc = desc.append(d)
        return desc    

# ---------------------------------------------------------------------------- #
#                                 ANALYSIS                                     #
# ---------------------------------------------------------------------------- #


def analysis(df, x, y, hue=None):

    k = independence.Kruskal()
    a = independence.Anova()

    if ((df[x].dtype == np.dtype('int64') or df[x].dtype == np.dtype('float64')) and
            (df[y].dtype == np.dtype('int64') or df[y].dtype == np.dtype('float64'))):
        desc = pd.DataFrame()
        dx = description.describe_quant_x(df[x])
        dy = description.describe_quant_x(df[y])
        desc = pd.concat([dx, dy], axis=0)
        ind = independence.correlation(df, x, y)
        plots = visual.quantitative(df=df, x=x, y=y)
    elif ((df[x].dtype == np.dtype('object')) and (df[y].dtype == np.dtype('object'))):
        ind = independence.association(df, x, y)
        desc = dict()
        desc['pct'] = pd.crosstab(df[x], df[y], rownames=[x],
                           colnames=[y], margins=True, normalize='index')        
        desc['count']= pd.crosstab(df[x], df[y], rownames=[x],
                           colnames=[y], margins=True, normalize=False)
        
        desc['count']['pct'] = round(desc['count']['All'] * 100 / desc['count']['All'][-1],2)
        plots = visual.categorical(df, x, y)
    elif ((df[x].dtype == np.dtype('int64') or df[x].dtype == np.dtype('float64')) and
            (df[y].dtype == np.dtype('object'))):
        ind1 = k.kruskal_test(df=df, x=y, y=x)
        ind2 = a.aov_test(df=df, x=y, y=x)
        ind = pd.concat([ind1, ind2], axis=0, sort=False)
        desc = description.group_describe(df, x, y)
        plots = visual.mixed_plot(df, x, y)
    else:
        ind1 = k.kruskal_test(df=df, x=x, y=y)
        ind2 = a.aov_test(df=df, x=x, y=y)
        ind = pd.concat([ind1, ind2], axis=0, sort=False)
        desc = description.group_describe(df, y, x)
        plots = visual.mixed_plot(df, y, x)
    return ind, desc, plots
    



    