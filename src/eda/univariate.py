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

import description
import visual


# ---------------------------------------------------------------------------- #
#                             UNIVARIATE ANALYSIS                              #
# ---------------------------------------------------------------------------- #


def analysis(df):
    cols = df.columns
    result = []
    for col in cols:
        a = {}
        if (df[col].dtype == np.dtype('int64') or
                df[col].dtype == np.dtype('float64')):
            a['desc'] = description.describe_quant_x(df[col])
            a['plot'] = visual.quant_plot(df[col][df[col].notnull()])
            result.append(a)

        else:
            a['desc'] = description.describe_qual_x(df[col])
            a['plot'] = visual.countplot(x=col, df=df)
            result.append(a)
    return result
