# =========================================================================== #
#                                 ANALYSIS                                    #
# =========================================================================== #
'''Analysis and inference functions'''

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
import textwrap

import univariate
import visual
import description
import independence


# %%
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
