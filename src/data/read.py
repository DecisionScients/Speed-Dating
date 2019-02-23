#region
# =========================================================================== #
#                                     READ                                    #
# =========================================================================== #
# BSD License

# Copyright (c) 2019, John James
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
# ---------------------------------------------------------------------------- #
#endregion
'''
Module that reads the data from the interim folder.  Created to remove 
clutter from notebooks that require the data.
'''
#%%
# --------------------------------------------------------------------------- #
#                                   IMPORTS                                   #
# --------------------------------------------------------------------------- #
import os
import sys
import inspect

import numpy as np
import pandas as pd

from config import settings

# --------------------------------------------------------------------------- #
#                                READ MODULE                                  #
# --------------------------------------------------------------------------- #

class Speed8a():
    """Reads data from the Speed Dating Dataset 

    Class for reading the Speed Dating Dataset downloaded from the Kaggle
    website.  Methods allow for reading features and one, two, or three
    of the target variables. The targets are:
        dec - the decision of the subject
        dec_o - the decision of the partner
        match - indicates whether the subject and partner were a match

    Argument:
    ---------
    filepath (str): The file path to the file to be read

    Reference:
    https://www.kaggle.com/annavictoria/speed-dating-experiment

    """

    def __init__(self, filepath=None):
        """Instantiates class and designates the filepath for the data
        
        Initializes the class, sets the file path, and reads the codebook

        Argument:
        ---------
        filepath (str): the file path to the file to be read
        
        """

        self._speed8a = None
        if filepath is None:
            self._filepath = settings.filepath['raw']
        else:
            self._filepath = filepath
        self._codebook = pd.read_csv(settings.filepath['codebook'])
        self._codebook = self._codebook.loc[self._codebook['sel'].isin(['x', 'y'])]
    
    def _load(self):
        """Loads the data

        Reads the data from the designated file path.

        """
        self._speed8a = pd.read_csv(self._filepath, encoding='latin_1')
        return(self._speed8a)
    
    def features(self, cat_no=None):
        """Reads all features or features by numbered category
        
        Reads and returns the features of the dataset. The features are in 
        categories numbered from 1 to 22. If the category number is provided
        only those features in that category are returned.

        Arguments:
        ----------
        cat_no (int): An integer from 1 to 22 indicating the category of 
            features to return

        Returns:
        --------
        DataFrame: Dataframe of features.
        
        """  
        if isinstance(cat_no, int):
            features = list(self._codebook[self._codebook['cat_no']==cat_no]['column'])
        elif cat_no is None:
            features = list(self._codebook[self._codebook['sel'] =='x']['column'])
        else:
            raise Exception("Optional parameter cat_no must be an integer")
        if self._speed8a is None:
            self._load()            
        return(self._speed8a[features])

    def target(self, target='dec'):
        """Returns a target variable
        
        Returns the designated target variable. The default is 'dec' the 
        decision of the subject. The decision of the partner 'dec_o' or the 
        match variable can also be selected.  'dec' is the default.

        Argument:
        ---------
        target (str): String containing the name of the target variable.

        Returns:
        --------
        Series: Series containing the target variable data.

        """
        if self._speed8a is None:
            self._load()            
        return(self._speed8a[target])


    def codebook(self, what='all'):
        """Reads and returns the codebook

        Reads and returns the codebook in one of three forms.  

            1. List of all variables including category, description and data type
            2. List of categories
            3. List of variables for a designated category

        Arguments:
        ----------
        what (int str): 'all', or 'cat'.  If 'all' is designated, a five column 
            dataframe will be returned containing the category number, the
            category, the variable, the variable description and data type.  
            If 'cat' is selected, a two column dataframe is returned 
            containing just the category numbers and the categories. If
            an int, the variables for that category are returned.

        Returns:
        --------
        DataFrame. The full codebook or the categories.

        """
        vars_all = ['cat_no', 'category', 'column', 'description', 'type']
        vars_cat = ['cat_no', 'category']
        if what == 'all':            
            return(self._codebook[vars_all])
        elif what == 'cat':            
            return(self._codebook[vars_cat].drop_duplicates())  
        elif isinstance(what, int):            
            return(self._codebook[self._codebook['cat_no']==what][vars_all])

    


#%%
speed = Speed8a()
df = speed.features()
cb = speed.codebook(what=14)
target = speed.target()
print(target.head())
