# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 16:36:54 2017

@author: xiaog
"""

from pandas import read_table,read_csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


'''
Define an imputer function

'''
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


print("modification")



print("\nRescale using sklearn")

# import data and define the column names
filename_income = "C:/Users/xiaog/Dropbox/quant stuff/class/AI - bittiger/Week 1/adult_data.txt"
names_income = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
         "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country",
         "income"]
data_income = read_csv(filename_income,names=names_income)
data_income = pd.DataFrame(data_income)
# dataset = read_csv(filename,header=None)

# List a couple of data （first 50 data）
data_income.head(50)

# The summary of this data
print(data_income.describe())

# Set the values of the data to array
array_income = data_income.values
data_income = data_income.applymap(lambda x: np.NaN if "?" in str(x) else x)
data_income.head(50)
print("\n The summary of number of null data for each feature")
print(data_income.isnull().sum())
print("\nThe shape of the data frame is %s" % (data_income.shape,))

# Copy data to the data_income_drop and drop the NaN data
data_income_drop = data_income
data_income_drop.dropna(inplace=True)

# The shap
print("\nThe shape of the data frame after dropping missiong data is %s" % (data_income_drop.shape,))

# Copy data to the data_income_impute and fill the NaN values using method meadian
# The following using numpy

data_income_impute_np = data_income
data_income_impute_np.fillna(data_income_impute_np.median(),inplace=True)
print(data_income_impute_np.isnull().sum())
data_income_impute_np.head(50)

# Copy data to the data_income_impute and fill the NaN values
# The following using sklearn and DataFrameImputer
data_income_impute= data_income

data_income_transformed = DataFrameImputer().fit_transform(data_income_impute)
print(data_income_transformed.isnull().sum())
data_income_transformed.dtypes
data_income.head()

## Next we scale the target data

scaler = MinMaxScaler(feature_range=(0,1))

# rescaling 
for c in data_income_drop:
    if data_income_drop[c].dtype == np.dtype('O'):
        data_income_drop[c] = data_income_drop[c].astype("category")
    else:
        data_income_drop[c] = scaler.fit_transform(data_income_drop[c].values.reshape(-1,1))

# check the data
data_income_drop.head()
data_income_res_d = data_income_drop["income"]
data_income_train_d = data_income_drop.drop('income',axis=1)

# map the response into 0 and 1

data_income_train_d.head()

# create the list containing numeric value
name_numeric = [c for c in data_income_train_d if data_income_train_d[c].dtype == 
                np.dtype('float64')]

data_income_train_final = data_income_train_d[name_numeric]
data_income_train_final.head()

# convert the categorical features into dummy representations
for c in data_income_train_d:
    if data_income_train_d[c].dtype != np.dtype('float64'):
        dummy_ranks = pd.get_dummies(data_income_train_d[c], 
                                     prefix = c, drop_first = True)
        data_income_train_final = data_income_train_final.join(dummy_ranks.loc[:,])

data_income_train_final['intercept'] = 1.0
# convert the response into 0-1 coding
# labeling <=50K as 0 and >50K as 1
le =LabelEncoder()
data_income_res_df = pd.DataFrame(le.fit_transform(data_income_res_d),
                                  columns=["income"],index=data_income_train_final.index) 


# Set up the logistic regression
# The setting is correct but the singular matrix error raised since this
# design matrix of this dataset, after rescaling the distilling the dummy variables, produces
# dependent columns. For this reason, we have to select features at first
# One can perform this analysis in R, which would be much convenient since
# that algorithm has already implemented methods to tackle this case
logit_drop = sm.Logit(data_income_res_df,data_income_train_final)

result = logit_drop.fit()




