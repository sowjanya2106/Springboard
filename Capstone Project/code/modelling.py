#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:14:44 2018

@author: sowjanya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

import pandas as pd
import io

filtered_loans_df = pd.read_csv('final_filtered.csv',index_col=0)
filtered_loans_df

#Normalize the data and split it into trianing set and test set.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

#Target Variable
y = filtered_loans_df.loan_status.values
#print(y)

y[y==3] = 1
#Independent variables
X = filtered_loans_df.drop(['loan_status'],axis=1)

#Normalize the data
normalizer = Normalizer()
scaled_X = normalizer.fit_transform(X)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.2, random_state=42, stratify=y)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12, ratio = 1.0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

len(X_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def accuracy_metrics(y_test,y_pred):
    print("#--------------Confusion Matrix--------------------")
    print(confusion_matrix(y_test, y_pred))
    print("#--------------Classification Report--------------------")
    print(classification_report(y_test, y_pred))
    print("#--------------Accuracy Score --------------------")
    print(accuracy_score(y_test, y_pred))
    print("#--------------ROC AUC score---------------------- ")
    print(roc_auc_score(y_test, y_pred))
    
        

def classify(clf,X_train,y_train,X_test,y_test) :
    # Fit the classifier to the training data
    clf.fit(X_train,y_train)
    # Predict the labels of the test set: y_pred
    y_pred = clf.predict(X_test)
    #Calculate the accuracy of the model
    print("#--------------Accuarcy Score---------------------- ")
    accuracy_metrics(y_test,y_pred)

    print("#--------------cross validation---------------------- ")
    cv_scores_3 = cross_val_score(clf,X_train,y_train,cv=3)
    print("3 fold Cross-Validation score : ", cv_scores_3 )
    print("Mean of 3 fold Cross-Validation score : ", np.mean(cv_scores_3) )
    print("#------------------------------------ ")
    #cv_scores_5 = cross_val_score(clf,X_train,y_train,cv=5)
   # print("5 fold Cross-Validation : " , cv_scores_5)
    #print("Mean of 5 fold Cross-Validation score : ", np.mean(cv_scores_5) )
   # print("#------------------------------------ ")

def classify_with_parameter_tuning(clf,param_grid,X_train,y_train,X_test,y_test) :
    
    clf_cv = GridSearchCV(clf,param_grid,n_jobs=-1)
    # Fit the classifier to the training data
    clf_cv.fit(X_train,y_train)
    # Predict the labels of the test set: y_pred
    y_pred = clf_cv.predict(X_test)
    
    # Print the optimal parameters and best score
    print("Tuned Classifier Parameter: {}".format(clf_cv.best_params_))
    print("Tuned Classifieer Regression Accuracy: {}".format(clf_cv.best_score_))
    accuracy_metrics(y_test,y_pred)
    

def plot_roc_graph(clf,X_train,y_train,X_test,y_test) :
    # Compute predicted probabilities: y_pred_prob
    logreg.fit(X_train,y_train)
    y_pred_prob =  logreg.predict_proba(X_test)[:,1]

    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import xgboost as xgb
#st = datetime.now()
from xgboost import XGBClassifier
clf = XGBClassifier(
    objective="binary:logistic", 
    learning_rate=0.05, 
    seed=9616, 
    max_depth=20, 
    gamma=10, 
    n_estimators=500,
n_jobs=-1)

#clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", verbose=True)
clf.fit(X_train, y_train)

#print(datetime.now()-st)

y_pred = clf.predict(X_test)