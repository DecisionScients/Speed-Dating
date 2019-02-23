# =========================================================================== #
#                             INDEPENDENCE MODULE                             #
# =========================================================================== #
'''Modules for analyzing indendence between variables.'''
# %%
# --------------------------------------------------------------------------- #
#                                 LIBRARIES                                   #
# --------------------------------------------------------------------------- #
import collections
from collections import OrderedDict
import itertools
from itertools import combinations
from itertools import product
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import scikit_posthocs as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm



# %%
# ---------------------------------------------------------------------------- #
#                                CORRELATION                                   #
# ---------------------------------------------------------------------------- #
class Correlation:
    '''Class that computes the pairwise correlation between numeric variables
    and renders a heatmap
    '''
    def __init__(self):
        self._corr = None
        pass
    
    def test(self, df, method='pearson'):
        self._corr = df.corr(method)
        return(self._corr)        

    def pairwise(self, df, x, y, method='pearson', threshold=None):
        r_tests = pd.DataFrame()
        for xs, ys in zip(x,y):
            r = df[xs].corr(df[ys])
            df_r = pd.DataFrame({'x':xs, 'y':ys}, index=[0])
            df_r['r'] = r
            df_r['r_abs'] = np.absolute(r)
            df_r['strength'] = np.where(df_r.r_abs<0.2, 'Very Weak',
                                np.where(df_r.r_abs<0.4, 'Weak',
                                np.where(df_r.r_abs<0.6, "Moderate",
                                np.where(df_r.r_abs<0.8, "Strong", "Very Strong"))))
            df_r['direction'] = np.where(df_r.r <0, "Negative", "Positive")
            r_tests = pd.concat([r_tests, df_r], axis=0)

        r_tests = r_tests.sort_values(by='r_abs', ascending=False)
        if threshold:
            r_tests = r_tests[r_tests.r_abs > threshold]
        return(r_tests)
    
    def corrtable(self, threshold=None):
        r_tests = pd.DataFrame()
        cols = self._corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i != j:
                    df_r = pd.DataFrame({'x': cols[i], 'y':cols[j], 'r': self._corr.iloc[i][j],
                                        'r_abs': np.absolute(self._corr.iloc[i][j])}, index=[0])
                    df_r['strength'] = np.where(df_r.r_abs<0.2, 'Very Weak',
                                        np.where(df_r.r_abs<0.4, 'Weak',
                                        np.where(df_r.r_abs<0.6, "Moderate",
                                        np.where(df_r.r_abs<0.8, "Strong", "Very Strong"))))
                    df_r['direction'] = np.where(df_r.r <0, "Negative", "Positive")
                    r_tests = pd.concat([r_tests, df_r], axis=0)
        r_tests = r_tests.sort_values(by='r_abs', ascending=False)
        if threshold:
            r_tests = r_tests[r_tests.r_abs > threshold]
        return(r_tests)

    def corrplot(self):
        sns.heatmap(self._corr, xticklabels=self._corr.columns,
                                yticklabels=self._corr.columns)

# ---------------------------------------------------------------------------- #
#                               INDEPENDENCE                                   #
# ---------------------------------------------------------------------------- #


class Independence:
    "Class that performs a test of independence"

    def __init__(self):
        self._sig = 0.05
        self._x2 = 0
        self._p = 0
        self._df = 0
        self._obs = []
        self._exp = []

    def summary(self):
        print("\n*", "=" * 78, "*")
        print('{:^80}'.format("Pearson's Chi-squared Test of Independence"))
        print('{:^80}'.format('Data'))
        print('{:^80}'.format("x = " + self._xvar + " y = " + self._yvar + "\n"))
        print('{:^80}'.format('Observed Frequencies'))
        visual.print_df(self._obs)
        print("\n", '{:^80}'.format('Expected Frequencies'))
        visual.print_df(self._exp)
        results = ("Pearson's chi-squared statistic = " + str(round(self._x2, 3)) + ", Df = " +
                   str(self._df) + ", p-value = " + '{0:1.2e}'.format(round(self._p, 3)))
        print("\n", '{:^80}'.format(results))
        print("\n*", "=" * 78, "*")

    def post_hoc(self, rowwise=True, verbose=False):

        dfs = []
        if rowwise:
            rows = range(0, len(self._obs))
            for pair in list(combinations(rows, 2)):
                ct = self._obs.iloc[[pair[0], pair[1]], ]
                levels = ct.index.values
                x2, p, dof, exp = stats.chi2_contingency(ct)
                df = pd.DataFrame({'level_1': levels[0],
                                   'level_2': levels[1],
                                   'x2': x2,
                                   'N': ct.values.sum(),
                                   'p_value': p}, index=[0])
                dfs.append(df)
            self._post_hoc_tests = pd.concat(dfs)
        else:
            cols = range(0, len(self._obs.columns.values))
            for pair in list(combinations(cols, 2)):
                ct = self._obs.iloc[:, [pair[0], pair[1]]]
                levels = ct.columns.values
                x2, p, dof, exp = stats.chi2_contingency(ct)
                df = pd.DataFrame({'level_1': levels[0],
                                   'level_2': levels[1],
                                   'x2': x2,
                                   'N': ct.values.sum(),
                                   'p_value': p}, index=[0])
                dfs.append(df)
            self._post_hoc_tests = pd.concat(dfs)
        if (verbose):
            visual.print_df(self._post_hoc_tests)

        return(self._post_hoc_tests)

    def test(self, x, y, sig=0.05):
        self._x = x
        self._y = y
        self._xvar = x.name
        self._yvar = y.name
        self._n = x.shape[0]
        self._sig = sig

        ct = pd.crosstab(x, y)
        x2, p, dof, exp = stats.chi2_contingency(ct)

        self._x2 = x2
        self._p = p
        self._df = dof
        self._obs = ct
        self._exp = pd.DataFrame(exp).set_index(ct.index)
        self._exp.columns = ct.columns

        if p < sig:
            self._result = 'significant'
            self._hypothesis = 'reject'
        else:
            self._result = 'not significant'
            self._hypothesis = 'fail to reject'

        return x2, p, dof, exp

    def report(self, verbose=False):
        "Returns or prints results in APA format"
        tup = ("A Chi-square test of independence was conducted to "
               "examine the relation between " + self._xvar + " and " + self._yvar + ". "
               "The relation between the variables was " + self._result + ", "
               "X2(" + str(self._df) + ", N = ", str(self._n) + ") = " +
               str(round(self._x2, 2)) + ", p = " + '{0:1.2e}'.format(round(self._p, 3)))

        self._report = ''.join(tup)

        wrapper = textwrap.TextWrapper(width=80)
        lines = wrapper.wrap(text=self._report)
        if verbose:
            for line in lines:
                print(line)
        return(self._report)

# ---------------------------------------------------------------------------- #
#                                   ANOVA                                      #
# ---------------------------------------------------------------------------- #

#%%
class Anova:
    '''
    Computes Anova tests     
    '''
    def __init__(self):
        pass

    def aov_test(self, df, x, y, type=2, test='F', sig=0.05):
        df2 = pd.DataFrame({'x': df[x], 'y': df[y]})
        df2 = df2.dropna() 
        model = smf.ols('y~x', data=df2).fit()
        aov = sm.stats.anova_lm(model, typ=type, test=test)
        tbl = pd.DataFrame({
            'Test': 'Anova', 
            'Dependent': y, 'Independent': x, 'Statistic': 'F Statistic',
            'Statistic Value': aov['F'][0],  'p-Value': aov['PR(>F)'][0]
        }, index=[0])
        tbl['H0'] = np.where(tbl['p-Value']<sig, 'Reject', 'Fail to Reject')
        return(tbl)

    def aov_table(self, df, x=None, y=None, type=2, test='F', threshold=0):
        tests = pd.DataFrame()
        if x and y:
            df2 = pd.DataFrame({'x': df[x], 'y': df[y]})
            df2 = df2.dropna() 
            model = smf.ols('y~x', data=df2).fit()
            aov = sm.stats.anova_lm(model, typ=type, test=test)
            tbl = pd.DataFrame({
                'Dependent': y, 'Independent': x, 'Sum Sq Model': aov['sum_sq'][0],
                'Sum Sq Residuals': aov['sum_sq'][1], 'df Model': aov['df'][0],
                'df Residuals': aov['df'][1],  'F': aov['F'][0],
                'PR(>F)': aov['PR(>F)'][0]
            }, index=[0])
            tbl['Eta Squared'] = tbl['Sum Sq Model'] / (tbl['Sum Sq Model'] + tbl['Sum Sq Residuals'])
            tests = tests.append(tbl)
        elif x:        
            dfy = df.select_dtypes(include='object')
            ys = dfy.columns
            for y in ys:
                df2 = pd.DataFrame({'x': df[x], 'y': df[y]})
                df2 = df2.dropna()
                model = smf.ols('x~y', data=df2).fit()
                aov = sm.stats.anova_lm(model, typ=type, test=test)
                tbl = pd.DataFrame({
                    'Dependent': y, 'Independent': x, 'Sum Sq Model': aov['sum_sq'][0],
                    'Sum Sq Residuals': aov['sum_sq'][1], 'df Model': aov['df'][0],
                    'df Residuals': aov['df'][1],  'F': aov['F'][0],
                    'PR(>F)': aov['PR(>F)'][0]
                }, index=[0])
                tbl['Eta Squared'] = tbl['Sum Sq Model'] / (tbl['Sum Sq Model'] + tbl['Sum Sq Residuals'])
                tests = tests.append(tbl)            
        elif y:
            dfx = df.select_dtypes(include=[np.number])
            xs = dfx.columns
            for x in xs:
                df2 = pd.DataFrame({'x': df[x], 'y': df[y]})
                df2 = df2.dropna()
                model = smf.ols('x~y', data=df2).fit()
                aov = sm.stats.anova_lm(model, typ=type, test=test)
                tbl = pd.DataFrame({
                    'Dependent': y, 'Independent': x, 'Sum Sq Model': aov['sum_sq'][0],
                    'Sum Sq Residuals': aov['sum_sq'][1], 'df Model': aov['df'][0],
                    'df Residuals': aov['df'][1],  'F': aov['F'][0],
                    'PR(>F)': aov['PR(>F)'][0]
                }, index=[0])
                tbl['Eta Squared'] = tbl['Sum Sq Model'] / (tbl['Sum Sq Model'] + tbl['Sum Sq Residuals'])
                tests = tests.append(tbl)            
        else:
            dfx = df.select_dtypes(include=[np.number])
            dfy = df.select_dtypes(include='object')
            xs = dfx.columns
            ys = dfy.columns
            for pair in list(itertools.product(xs,ys)):
                df2 = df[[pair[0], pair[1]]].dropna()
                df2 = pd.DataFrame({'x': df2[pair[0]], 'y': df2[pair[1]]})
                model = smf.ols('x~y', data=df2).fit()
                aov = sm.stats.anova_lm(model, typ=type, test=test)
                tbl = pd.DataFrame({
                    'Dependent': y, 'Independent': x, 'Sum Sq Model': aov['sum_sq'][0],
                    'Sum Sq Residuals': aov['sum_sq'][1], 'df Model': aov['df'][0],
                    'df Residuals': aov['df'][1],  'F': aov['F'][0],
                    'PR(>F)': aov['PR(>F)'][0]
                }, index=[0])
                tbl['Eta Squared'] = tbl['Sum Sq Model'] / (tbl['Sum Sq Model'] + tbl['Sum Sq Residuals'])
                tests = tests.append(tbl)            
        tests = tests.loc[tests['Eta Squared'] > threshold]
        tests = tests.sort_values(by='Eta Squared', ascending=False)
        return(tests)


# ---------------------------------------------------------------------------- #
#                                   KRUSKAL                                    #
# ---------------------------------------------------------------------------- #

#%%
class Kruskal:
    '''
    Class provides non-parametric methods for testing independence    
    '''    
    def __init__(self):
        pass

    def kruskal_test(self, df, x, y, sig=0.05):
        '''Computes the Kruskal-Wallis H-test tests
        Args:
            df (pd.DataFrame): Dataframe containing data
            x (str): The name of the categorical independent variable
            y (str): The name of the numerical dependent variable
        Returns:
            DataFrame containing statistic and p-value
        '''
        df = df[[x,y]].dropna()
        groups = {}
        for grp in df[x].unique():
            groups[grp] = df[y][df[x]==grp].values
        args = groups.values()        

        k = stats.kruskal(*args)
        columns = ['Test', 'Dependent', 'Independent', 'Statistic', 'Statistic Value', 'p-Value']
        data = [['Kruskal', y, x, 'H-Statistic', k[0], k[1]]]
        r = pd.DataFrame(data, columns = columns)        
        r['H0'] = np.where(r['p-Value']<sig, 'Reject', 'Fail to Reject')
        return(r)

    def kruskal_table(self, df, x=None, y=None, sig=0.05, sort=False):
        tests = pd.DataFrame()
        if x and y:
            test = self.kruskal_test(df, x, y)
            tests = tests.append(test)
        elif x:        
            dfy = df.select_dtypes(include=[np.number])
            ys = dfy.columns.tolist()
            for y in ys:
                df2 = df[[x,y]].dropna()
                test = self.kruskal_test(df2, x, y)
                tests = tests.append(test)
        elif y:
            dfx = df.select_dtypes(include='object')
            xs = dfx.columns.tolist()
            for x in xs:
                df2 = df[[x,y]].dropna()
                test = self.kruskal_test(df2, x, y)
                tests = tests.append(test)           
        else:
            dfx = df.select_dtypes(include='object')
            dfy = df.select_dtypes(include=[np.number])
            xs = dfx.columns.tolist()
            ys = dfy.columns.tolist()
            for pair in list(itertools.product(xs,ys)):
                df2 = df[[pair[0], pair[1]]].dropna()
                test = self.kruskal_test(df2, pair[0], pair[1])
                tests = tests.append(test)   
        if sort:
            tests = tests.sort_values(by=['Independent','Statistic Value'], ascending=False)              
        return(tests)        


    def posthoc(self, df, x, y):
        df = df[[x,y]].dropna()
        p = sp.posthoc_conover(df, val_col=y, group_col=x, p_adjust = 'fdr_bh')
        return(p)

    def sign_plot(self, df, x, y):
        p = self.posthoc(df, x, y)
        heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False,
                        'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        sp.sign_plot(p, **heatmap_args)       


# %%

# ---------------------------------------------------------------------------- #
#                                CORRELATION                                   #
# ---------------------------------------------------------------------------- #


def correlation(df, x, y):
    '''
    Computes the correlation between two quantitative variables x and y.

    Args:
        df (pd.DataFrame): Dataframe containing numeric variables
        x (str): The column name for the x variable
        y (str): The column name for the y variable

    Returns:
        Data frame containing the results of the correlation tests
    '''
    df = df.dropna()
    r = stats.pearsonr(df[x], df[y])
    test = pd.DataFrame({'x': x, 'y': y, "Correlation": r[0], "p-value": r[1]},
                        index=[0])
    test['AbsCorr'] = test['Correlation'].abs()
    test['Strength'] = np.where(test["AbsCorr"] < .1, 'Extremely Weak Correlation',
                                np.where(test["AbsCorr"] < .30, 'Small Correlation',
                                         np.where(test["AbsCorr"] < .5, 'Moderate Correlation',
                                                  'Strong Correlation')))
    return(test)

# ---------------------------------------------------------------------------- #
#                                CORR_TABLE                                    #
# ---------------------------------------------------------------------------- #


def corr_table(df, x=None, y=None, target=None, threshold=0, sig=None):
    '''For a dataframe containing numeric variables, this function
    computes pairwise pearson's R tests of correlation correlation.
    
    Args:
        df (pd.DataFrame): Data frame containing numeric variables
        x(str): Name of independent variable column (optional)
        y(str): Name of dependent variable column (optional)
        target(str):
        threshold (float): Threshold above which correlations should be
                           reported.
    Returns:
        Data frame containing the results of the pairwise tests of correlation.
    '''
    tests = []
    if x is not None:
        for pair in list(itertools.product(x, y)):
            df2 = df[[pair[0], pair[1]]].dropna()
            x = df2[pair[0]]
            y = df2[pair[1]]
            r = stats.pearsonr(x, y)
            tests.append(OrderedDict(
                {'x': pair[0], 'y': pair[1], "Correlation": r[0], "p-value": r[1]}))
        tests = pd.DataFrame(tests, index=[0])
        tests['AbsCorr'] = tests['Correlation'].abs()
        tests['Strength'] = np.where(tests["AbsCorr"] < .1, 'Extremely Weak Correlation',
                                     np.where(tests["AbsCorr"] < .30, 'Small Correlation',
                                              np.where(tests["AbsCorr"] < .5, 'Moderate Correlation',
                                                       'Strong Correlation')))        
    else:
        df2 = df.select_dtypes(include=['int', 'float64'])
        terms = df2.columns
        if target:
            if target not in df2.columns:
                df2 = df2.join(df[target])
            for term in terms:
                df2 = df2.dropna()
                x = df2[term]
                y = df2[target]
                r = stats.pearsonr(x, y)
                tests.append(OrderedDict(
                    {'x': term, 'y': target, "Correlation": r[0], "p-value": r[1]}))
            tests = pd.DataFrame(tests)
            tests['AbsCorr'] = tests['Correlation'].abs()
            tests['Strength'] = np.where(tests["AbsCorr"] < .1, 'Extremely Weak Correlation',
                                         np.where(tests["AbsCorr"] < .30, 'Small Correlation',
                                                  np.where(tests["AbsCorr"] < .5, 'Moderate Correlation',
                                                           'Strong Correlation')))
        else:
            for pair in list(combinations(terms, 2)):
                df2 = df[[pair[0], pair[1]]].dropna()
                x = df2[pair[0]]
                y = df2[pair[1]]
                r = stats.pearsonr(x, y)
                tests.append(OrderedDict(
                    {'x': pair[0], 'y': pair[1], "Correlation": r[0], "p-value": r[1]}))
            tests = pd.DataFrame(tests)
            tests['AbsCorr'] = tests['Correlation'].abs()
            tests['Strength'] = np.where(tests["AbsCorr"] < .1, 'Extremely Weak Correlation',
                                         np.where(tests["AbsCorr"] < .30, 'Small Correlation',
                                                  np.where(tests["AbsCorr"] < .5, 'Moderate Correlation',
                                                           'Strong Correlation')))
    top = tests.loc[tests['AbsCorr'] > threshold]
    if sig is not None:
        top = tests.loc[tests['p-value']<sig]
    top = top.sort_values(by='AbsCorr', ascending=False)
    return top

# ---------------------------------------------------------------------------- #
#                            CRAMER'S V (Corrected)                            #
# ---------------------------------------------------------------------------- #


def cramers(contingency_table, correction=False):
    """ calculate Cramers V statistic for categorical-categorical association.
        If correction is True, it uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328

        Args:
            contingency_table (pd.DataFrame): Contingency table containing
                                              counts for the two variables
                                              being analyzed
            correction (bool): If True, use Bergsma's correction
        Returns:
            float: Corrected Cramer's V measure of Association                                    
    """
    chi2, p = stats.chi2_contingency(contingency_table)[0:2]
    n = contingency_table.sum().sum()
    phi = np.sqrt(chi2/n)    
    r, c = contingency_table.shape

    if correction:
        phi2corr = max(0, phi**2 - ((c-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        ccorr = c - ((c-1)**2)/(n-1)
        V = np.sqrt(phi2corr / min((ccorr-1), (rcorr-1)))
    else:
        V = np.sqrt(phi**2/min(r,c))

    return p, V

# %%
# ---------------------------------------------------------------------------- #
#                                ASSOCIATION                                   #
# ---------------------------------------------------------------------------- #


def association(df, x, y, z=None, sig=0.05):
    '''
    Computes the association between two or three categorical variables.

    Args:
        df (pd.DataFrame): Dataframe containing categorical variables
        x (str): The column name for the x variable
        y (str): The column name for the y variable
        z (str): Optional column containing the z variable

    Returns:
        Data frame containing the results of the correlation tests
    '''
    if z:
        df = df[[x,y,z]].dropna()
        ct = pd.crosstab(df[z], [df[x], df[y]], rownames=[z], colnames=[x, y])
        p, cv = cramers(ct)
        test = pd.DataFrame({'x': x, 'y': y, 'z':z, 'p-Value':p, "Cramer's V": cv},
                        index=[0])
    else:
        df = df[[x,y]].dropna()
        ct = pd.crosstab(df[x], df[y])
        p, cv = cramers(ct)
        test = pd.DataFrame({'x': x, 'y': y, 'p-Value':p, "Cramer's V": cv},
                            index=[0])
    test['Strength'] = np.where(test["Cramer's V"] < .16, 'Very Weak Association',
                                np.where(test["Cramer's V"] < .20, 'Weak Association',
                                         np.where(test["Cramer's V"] < .25, 'Moderate Association',
                                         np.where(test["Cramer's V"] < .30, 'Moderately Strong Association',
                                         np.where(test["Cramer's V"] < .35, 'Strong Association',
                                         np.where(test["Cramer's V"] < .40, 'Very Strong Association',
                                         np.where(test["Cramer's V"] < .50, 'Extremely Strong Association',
                                                  'Redundant')))))))
    test['Result'] = np.where(test['p-Value']<sig, 'Significant', 'Not Significant')
    return(test)


# ---------------------------------------------------------------------------- #
#                                ASSOCTABLE                                    #
# ---------------------------------------------------------------------------- #


def assoc_table(df, x=None, y=None, threshold=0):
    '''For a dataframe containing categorical variables, this function 
    computes a series of association tests for each pair of categorical
    variables. It returns the adjusted Cramer's V measure of 
    association between the pairs of categorical variables.  Note, this 
    is NOT  a hypothesis test. 

    Args:
        df (pd.DataFrame): Data frame containing categorical variables
        x (str): Optional column to be used as the independent variable for 
                      all tests
        y (str): Optional column to be used as the dependent variable for 
                      all tests
        threshold (float): The minimum Cramer's V threshold to report.

    Returns:
        Data frame containing the results of the pairwise association measures.
    '''
    df2 = df.select_dtypes(include='object')
    terms = list(df2.columns)
    tests = []
   
    if x and y:
        ct = pd.crosstab(df[x], df[y])
        cv = cramers(ct)
        tests.append(OrderedDict(
            {'x': x, 'y': y, "Cramer's V": cv}))
    elif x:
        if x in terms:
            terms.remove(x)
        if x not in df2.columns:
            df2 = df2.join(df[x])
            df2 = df2.dropna()
        for term in terms:
            ct = pd.crosstab(df2[x], df2[term])
            cv = cramers(ct)
            tests.append(OrderedDict(
                {'x': x, 'y': term, "Cramer's V": cv}))

    elif y:
        if y in terms:
            terms.remove(y)
        if y not in df2.columns:
            df2 = df2.join(df[y])
            df2 = df2.dropna()
        for term in terms:
            ct = pd.crosstab(df2[term], df2[y])
            cv = cramers(ct)
            tests.append(OrderedDict(
                {'x': term, 'y': y, "Cramer's V": cv}))
    else:
        for pair in list(combinations(terms, 2)):
            df2 = df2.dropna()
            ct = pd.crosstab(df2[pair[0]], df2[pair[1]])
            cv = cramers(ct)
            tests.append(OrderedDict(
                {'x': pair[0], 'y': pair[1], "Cramer's V": cv}))

    tests = pd.DataFrame(tests)
    tests['Strength'] = np.where(tests["Cramer's V"] < .1, 'Very Weak Association',
                                 np.where(tests["Cramer's V"] < .2, 'Weak Association',
                                            np.where(tests["Cramer's V"] < .3, 'Moderate Association',
                                                    'Strong Association')))   
    top = tests.loc[tests["Cramer's V"] > threshold]
    top = top.sort_values(by="Cramer's V", ascending=False)
    return top
#%%
# ---------------------------------------------------------------------------- #
#                             FACTORIAL ANOVA                                  #
# ---------------------------------------------------------------------------- #
def factorial_anova(df, x, y, z):
    '''
    Performs a factorial anova test
    Args:
        df(DataFrame): Dataframe containing x, y, z, columns
        x(str): Name of categorical variable
        y(str): Name of quantitative response variable
        z(str): Name of controlling categorical variable
    Returns:
        Dataframe containing results of Anova with interactions        
    '''
    def eta_squared(aov):
        aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
        return aov
    def omega_squared(aov):
        mse = aov['sum_sq'][-1]/aov['df'][-1]
        aov['omega_sq'] = 'NaN'
        aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
        return aov
    df2 = pd.DataFrame({'x': df[x], 'y': df[y], 'z':df[z]})
    f = 'y ~ C(x) + C(z) + C(x):C(z)'
    model = ols(f, df2).fit()
    aov_table = anova_lm(model, typ=2)
    interaction = x + ':' + z
    aov_table['Terms']= [x,z,interaction, "Residuals"]
    aov_table = aov_table.set_index('Terms')
    aov_table = eta_squared(aov_table)
    aov_table = omega_squared(aov_table)
    aov_table = aov_table.replace(np.nan, '', regex=True)
    return(aov_table)

# ---------------------------------------------------------------------------- #
#                               GROUPED ANOVA                                  #
# ---------------------------------------------------------------------------- #
def grouped_anova(df,x,y,z, sig=0.05):
    '''
    Performs grouped anova test for a dataframe df, with controlling variable
    x, quantitative variable y, and target variable (categorical) z.
    Args:
        df(DataFrame): Dataframe containing x,y,z columns
        x(str): Name of column containing controlling variable
        y(str): Name of column containing quantitative variable
        z(str): Name of column containing target variable
    Returns:
        Dataframe containing F-statistic, p-value, and result of hypothesis
            test by group (controlling variable x).
    '''
    df2 = df[[x,y,z]]
    df2 = df2.dropna()
    aov = pd.DataFrame()
    for control in df2.groupby(x):    
        samples = [value[1] for value in control[1].groupby(z)[y]]   
        f, p = stats.f_oneway(*samples)
        d = {'F':f, 'p-value':p}
        a = pd.DataFrame(d, index=[control[0]])
        a['H0'] = np.where(a['p-value']>sig, 'Fail to Reject', 'Reject')
        aov = pd.concat([aov,a],axis=0) 
    return(aov)

