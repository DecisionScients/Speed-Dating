# =========================================================================== #
#                             MULTIVARIATE ANALYSIS                           #
# =========================================================================== #
'''Multivariate Analysis and inference functions'''

# %%
# --------------------------------------------------------------------------- #
#                                 LIBRARIES                                   #
# --------------------------------------------------------------------------- #
import os
import sys
import inspect

import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------- #
#                         Principal Component Analysis                         #
# ---------------------------------------------------------------------------- #


class PrincipalComponents:
    "Class that performs principal components related analysis"

    def __init__(self, n_components, svd_solver='auto', random_state=None):
        self._pca = PCA(n_components=n_components, svd_solver=svd_solver,
                        random_state=random_state)
        self._colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    def fit(self, X):
        self.X = X
        self._features = X.columns
        return(self._pca.fit(X))

    def fit_transform(self, X):
        self.X = X
        self._features = X.columns
        return(self._pca.fit_transform(X))

    def inverse_transform(self, X):
        self.X = X
        self._features = X.columns
        return(self._pca.inverse_transform(X))

    def log_score(self, X):
        self.X = X
        self._features = X.columns
        return(self._pca.score(X))

    def log_score_samples(self, X):
        self.X = X
        self._features = X.columns
        return(self._pca.score_samples(X))

    def transform(self, X):
        self.X = X
        self._features = X.columns
        return(self._pca.transform(X))

    def variance_explained(self):
        n = self._pca.n_components_
        components = [x for x in range(1, n+1)]
        ve = self._pca.explained_variance_
        pve = self._pca.explained_variance_ratio_
        cpve = np.cumsum(pve)
        result = pd.DataFrame({'Component': components, 'Eigenvalue': ve,
                               '% Variance Explained': pve*100,
                               'Cumulative % Variance Explained': cpve*100})
        return(result)

    def loadings(self,n=0):
        if n:
            n = min(n, self._pca.n_components_)
        else:
            n = self._pca.n_components_
        idx = ['Principal Component {0}'.format(s) for s in range(1, n+1)]
        return(pd.DataFrame(self._pca.components_[0:n], columns=self._features, index=idx).T)

    def score_samples(self):
        return(pd.DataFrame(self.X, columns = self._features).dot(self.loadings()))
        
    def screeplot(self):

        # Aesthetics
        barcolor = 'steelblue'
        linecolor1 = 'darkslategray'    
        linecolor2 = 'teal'
        sns.set(style="white", font_scale=1, rc={"lines.linewidth": 1.5})

        ve = self.variance_explained()
        ve_long = pd.melt(ve, id_vars=['Component'],
                          value_vars=['Eigenvalue',
                                      '% Variance Explained',
                                      'Cumulative % Variance Explained'],
                          var_name='Variance Explained',
                          value_name='Percent')

        fig, ax = plt.subplots(2, 2, figsize=(16, 8), sharex=True)

        # Eigenvalues Plot
        sns.barplot(
            x='Component', y='Eigenvalue', data=ve, ax=ax[0, 0], color=barcolor)
        sns.lineplot(
            x='Component', y='Eigenvalue', data=ve, ax=ax[0, 0], color=linecolor1)
        ax[0, 0].set_xlabel('Principal Component')
        ax[0, 0].set_ylabel('Eigenvalues')
        ax[0, 0].set_title('Eigenvalues Plot')

        # Percent Variance Explained
        sns.barplot(
            x='Component', y='% Variance Explained', data=ve, ax=ax[0, 1], color=barcolor)
        sns.lineplot(
            x='Component', y='% Variance Explained', data=ve, ax=ax[0, 1], color=linecolor1)
        ax[0, 1].set_xlabel('Principal Component')
        ax[0, 1].set_ylabel('Percent Variance Explained')
        ax[0, 1].set_title('Percent Variance Explained')

        # Cumulative Percent Variance Explained
        sns.barplot(
            x='Component', y='Cumulative % Variance Explained', data=ve, ax=ax[1, 0], color=barcolor)
        sns.lineplot(
            x='Component', y='Cumulative % Variance Explained', data=ve, ax=ax[1, 0], color=linecolor1)
        ax[1, 0].set_xlabel('Principal Component')
        ax[1, 0].set_ylabel('Cumulative Percent Variance Explained')
        ax[1, 0].set_title('Cumulative Percent Variance Explained')

        # Percent and Cumulative Variance Explained
        sns.barplot(
            x='Component', y='% Variance Explained', data=ve, ax=ax[1, 1], color=barcolor)
        line1 = sns.lineplot(
            x='Component', y='% Variance Explained', data=ve, ax=ax[1, 1], color=linecolor1)
        ax[1, 1].set_xlabel('Principal Component')
        ax[1, 1].set_ylabel('Percent Variance Explained')
        ax[1, 1].set_title('Percent Variance Explained')
        ax2 = ax[1, 1].twinx()
        line2 = sns.lineplot(
            x='Component', y='Cumulative % Variance Explained', data=ve, ax=ax2, color=linecolor2)

        line1_legend = mpatches.Patch(color=linecolor1, label='PVE')
        line2_legend = mpatches.Patch(color=linecolor2, label='Cumulative PVE')
        ax[1, 1].legend(handles=[line1_legend, line2_legend], fontsize='small')

        fig.suptitle("Principal Components Analysis Scree Plot")
        plt.show()

    def score_plot(self, n=2, group_var=None):
        scores = self.score_samples()

        # Establish plot dimensions
        if n:
            if n < 2 or n > scores.shape[0]:
                raise Exception("n must be between 2 and the number of features")
            else:
                n_plots = n-1
        else:
            n_plots = scores.shape[1]-1
        
        ncols = min(2, n_plots)
        nrows = math.ceil(np.sqrt(n_plots/ncols))

        # Initialize plot and set aesthetics            
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,8))
        sns.set(style = "whitegrid", font_scale = 1)
        sns.set_palette("Set1")

        for p in range(n_plots):
            c = p % 2 
            r = math.floor(np.sqrt(p/2))

            if (n_plots < 2):
                if group_var is None:
                    sns.scatterplot(x=scores.iloc[:,p], y=scores.iloc[:,p+1], ax=ax)
                else:
                    sns.scatterplot(x=scores.iloc[:,p], y=scores.iloc[:,p+1], ax=ax, hue=group_var, cmap="Set1")
                ax.set_xlabel('Principal Component {}'.format(p+1))
                ax.set_ylabel('Principal Component {}'.format(p+2))
                ax.set_title('Components {}'.format(p+1) + ' and {}'.format(p+2))
                ax.set_facecolor('white')
            elif (n_plots < 3):
                if group_var is None:
                    sns.scatterplot(x=scores.iloc[:,p], y=scores.iloc[:,p+1], ax=ax[p])
                else:
                    sns.scatterplot(x=scores.iloc[:,p], y=scores.iloc[:,p+1], ax=ax[p], hue=group_var, cmap="Set1")
                ax[p].set_xlabel('Principal Component {}'.format(p+1))
                ax[p].set_ylabel('Principal Component {}'.format(p+2))
                ax[p].set_title('Components {}'.format(p+1) + ' and {}'.format(p+2))
                ax[p].set_facecolor('white')
            else:             
                if group_var is None:
                    sns.scatterplot(x=scores.iloc[:,p], y=scores.iloc[:,p+1], ax=ax[r,c])
                else:
                    sns.scatterplot(x=scores.iloc[:,p], y=scores.iloc[:,p+1], ax=ax[r,c], hue=group_var, cmap="Set1")
                ax[r,c].set_xlabel('Principal Component {}'.format(p+1))
                ax[r,c].set_ylabel('Principal Component {}'.format(p+2))
                ax[r,c].set_title('Components {}'.format(p+1) + ' and {}'.format(p+2))
                ax[r,c].set_facecolor('white')

        if (c == 0 & n_plots > 2):
            fig.delaxes(ax[r,c+1])

        fig.suptitle("Principal Component Analysis Scores Plot", y=1.05)
        plt.tight_layout()

    def biplot(self, a=1, b=2, group_var=None):
        loadings = self.loadings()
        scores = self.score_samples()
        labels = loadings.index.tolist()

        # Extract variables of interest
        xs = scores.iloc[:,a-1]
        ys = scores.iloc[:,b-1]
        n_features = loadings.shape[0]

        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())

        scaled_xs = xs * scalex
        scaled_ys = ys * scaley

        fig, ax = plt.subplots(figsize=(12,8))        
        if group_var is not None:
            sns.scatterplot(scaled_xs, scaled_ys, hue = group_var, cmap="Set1", ax=ax, s=10)
        else:
            sns.scatterplot(scaled_xs, scaled_ys, ax=ax, s=10)
        for i in range(n_features):
            ax.arrow(0, 0, loadings.iloc[i,a-1], loadings.iloc[i,b-1], color = 'b',alpha = 0.5)
            ax.text(loadings.iloc[i,a-1]* 1.15, loadings.iloc[i,b-1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
        ax.set_title("Principal Components Analysis Biplot")
        ax.set_xlim(min(scaled_xs),max(scaled_xs))
        ax.set_ylim(min(scaled_ys),max(scaled_ys))
        ax.set_xlabel("PC{}".format(a))
        ax.set_ylabel("PC{}".format(b))

    def heatmap(self):
        fig, ax = plt.subplots(figsize=(12,8))     
        sns.heatmap(np.log(self._pca.inverse_transform(np.eye(self.X.shape[1]))), 
                    cmap="coolwarm", cbar=True, ax=ax)
        

# %%
# import data
# df = data.read()
# vars = ['gender', 'age', 'age_o', 'race','race_o', 'attractive',
#         'sincere', 'intelligence', 'funny', 'ambitious', 'match']
# df = df[vars]
# df = df.dropna()
# columns = df.columns
# match = df['match']

# # Label categorical variables
# le = LabelEncoder()
# ss = StandardScaler()
# df = df.apply(le.fit_transform)
# df = ss.fit_transform(df)
# df = pd.DataFrame(df, columns=columns)

# # Recombine data frames
# X = df.drop(columns='match')
# y = match
# pca = PrincipalComponents(X.shape[1])
# pca.fit(X)
# pca.heatmap()
