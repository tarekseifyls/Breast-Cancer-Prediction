# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:00:29 2022

@author: safwanshamsir99

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import pickle
import joblib
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

#%% function
def plot_cat(df,categorical_col):
    '''
    This function is to generate plots for categorical columns

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    categorical_col : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    for i in categorical_col:
        plt.figure() 
        sns.countplot(df[i]) 
        plt.show()

def plot_con(df,continuous_col):
    '''
    This function is to generate plots for continuous columns

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    continuous_col : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    for j in continuous_col:
        plt.figure()
        sns.distplot(df[j])
        plt.show()

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,         
        Journal of the Korean Statistical Society 42 (2013): 323-328    
    """    
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n    
    r,k = confusion_matrix.shape    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% STATIC
CSV_PATH = os.path.join(os.getcwd(),'breast-cancer.csv')

#%% DATA LOADING
df = pd.read_csv(CSV_PATH)

#%% DATA INSPECTION
df.info()
df.duplicated().sum() # no duplicated value
df = df.drop(labels='id',axis=1) # drop id column
stats= df.describe().T # no outlier
df.isna().sum()  #no null values
df.boxplot() # area mean, area se and area worst have outlier
df.columns

#%% DATA CLEANING
con_data = df.columns[df.dtypes=='float64']
plot_con(df, con_data)

cat_data = df.columns[df.dtypes=='object']
plot_cat(df, cat_data) # unbalance categorical data

#%% FEATURES SELECTION
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
LE_PATH = os.path.join(os.getcwd(),'le_breast.pkl')
with open(LE_PATH,'wb') as file:
    pickle.dump(le,file)

# categorical vs categorical(target) using cramer's V
'''
Since the only categorical data is Class column and the target data is 
Class column, so no need to check for the correlation
'''

# continuous features vs categorical target using LogisticRegression
for con in con_data:
    logreg = LogisticRegression()
    logreg.fit(np.expand_dims(df[con],axis=-1), df['diagnosis'])
    print(con)
    print(logreg.score(np.expand_dims(df[con],axis=-1),df['diagnosis'])) # accuracy

'''
Since radius_mean, texture_mean,  perimeter_mean, area_mean, concavity_mean, 
radius_se, perimeter_se, area_se, radius_worst, texture_worst, 
perimeter_worst, area_worst, compactness_worst, concavity_worst, 
concave points_worst has high percentage relation (>0.7) when trained with 
Logistic Regression, they will be chosen as the features.
'''

#%% PREPROCESSING
df_features=['radius_mean','texture_mean','perimeter_mean','area_mean',
             'concavity_mean','radius_se','perimeter_se','area_se', 
             'radius_worst','texture_worst','perimeter_worst','area_worst', 
             'compactness_worst','concavity_worst','concave points_worst']

X = df.loc[:,df_features]
y = df.loc[:,'diagnosis']

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=3)

#%% MODEL DEVELOPMENT
#LogisticRegression,RandomForest,DecisionTree,KNeighbors,SVC
# LR
pl_std_lr = Pipeline([('Standard Scaler',StandardScaler()),
                      ('LogClassifier',LogisticRegression())]) 

pl_mm_lr = Pipeline([('Min Max Scaler',MinMaxScaler()),
                     ('LogClassifier',LogisticRegression())])

#RF
pl_std_rf = Pipeline([('Standard Scaler',StandardScaler()),
                      ('RFClassifier',RandomForestClassifier())]) 

pl_mm_rf = Pipeline([('Min Max Scaler',MinMaxScaler()),
                     ('RFClassifier',RandomForestClassifier())]) 

# Decision Tree
pl_std_tree = Pipeline([('Standard Scaler',StandardScaler()),
                        ('DTClassifier',DecisionTreeClassifier())]) 

pl_mm_tree = Pipeline([('Min Max Scaler',MinMaxScaler()),
                       ('DTClassifier',DecisionTreeClassifier())]) 

# KNeighbors
pl_std_knn = Pipeline([('Standard Scaler',StandardScaler()),
                       ('KNClassifier',KNeighborsClassifier())]) 

pl_mm_knn = Pipeline([('Min Max Scaler',MinMaxScaler()),
                      ('KNClassifier',KNeighborsClassifier())])

# SVC
pl_std_svc = Pipeline([('Standard Scaler',StandardScaler()),
                       ('SVClassifier',SVC())]) 

pl_mm_svc = Pipeline([('Min Max Scaler',MinMaxScaler()),
                      ('SVClassifier',SVC())])

# create pipeline
pipelines = [pl_std_lr,pl_mm_lr,pl_std_rf,pl_mm_rf,pl_std_tree,
             pl_mm_tree,pl_std_knn,pl_mm_knn,pl_std_svc,pl_mm_svc]

# fitting the data
for pipe in pipelines:
    pipe.fit(X_train,y_train)

pipe_dict = {0:'SS+LR', 
             1:'MM+LR',
             2:'SS+RF',
             3:'MM+RF',
             4:'SS+Tree',
             5:'MM+Tree',
             6:'SS+KNN',
             7:'MM+KNN',
             8:'SS+SVC',
             9:'MM+SVC'}
best_accuracy = 0

# model evaluation
for i,model in enumerate(pipelines):
    print(model.score(X_test,y_test))
    if model.score(X_test, y_test) > best_accuracy:
        best_accuracy = model.score(X_test,y_test)
        best_pipeline = model
        best_scaler = pipe_dict[i]
        
print('The best pipeline for breast dataset will be {} with accuracy of {}'
      .format(best_scaler, best_accuracy))

#%% Fine tune the model (MM+LR)
'''
Based on the pipeline, model with highest accuracy is Min Max Scaler 
and Logistic Regression with accuracy of 0.971.
'''
pl_mm_lr = Pipeline([('Min Max Scaler',MinMaxScaler()),
                     ('LogClassifier',LogisticRegression())])

# number of trees
grid_param = [{'LogClassifier':[LogisticRegression()],
               'LogClassifier__solver':['lbfgs','liblinear','newton-cg'],
               'LogClassifier__C':[0.2,0.5,1.0,1.5]}]

gridsearch = GridSearchCV(pl_mm_lr,grid_param,cv=5,verbose=1,n_jobs=1)
best_model = gridsearch.fit(X_train, y_train)
print(best_model.score(X_test,y_test))
print(best_model.best_index_)
print(best_model.best_params_)

# saving the best pipeline
BEST_PIPE_PATH = os.path.join(os.getcwd(),'breast_fine_tune.pkl')
with open(BEST_PIPE_PATH,'wb') as file:
    pickle.dump(best_model,file)

'''
Since the accuracy is about the same(0.976 (added 0.006)) but different in 
params,the ml model will use the fine tune parameters.
'''

#%% MODEL EVALUATION
pl_mm_lr = Pipeline([('Min Max Scaler',MinMaxScaler()),
                     ('LogClassifier',LogisticRegression(C=1.5,solver='lbfgs'))])
pl_mm_lr.fit(X_train,y_train)

# saving the best model
joblib.dump(pl_mm_lr, 'best_model_breast.sav')


#%% MODEL ANALYSIS
y_true = y_test
y_pred = pl_mm_lr.predict(X_test)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print('Accuracy score: ' + str(accuracy_score(y_true, y_pred)))


