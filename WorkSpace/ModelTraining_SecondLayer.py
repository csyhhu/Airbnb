# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:12:46 2016

@author: aurelius

This program performs the second layer learning and predict:
1) It combine the output prediction probability from the first layer.
2) Then use Sandro's classifier(EN_optA, EN_optB, cc_optA, cc_optB) to make prediction.

In our final case we just perform the first stage, but not using Sandro's classifier, 
instead, we use other classifier to perform the second layer prediction. You may refer to
ModelTraining_SecondLayer2.py for more details.
"""

import numpy as np
import pandas as pd
import os
import xgboost as xgb
#import 

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator
from scipy.optimize import minimize

from sklearn.externals import joblib

os.chdir('/media/aurelius/New Volume/Kaggle/Airbnb') # Set your workspace here

#Prepare the training set
X_second = []
X_second_LR = np.load('Data/Eval/LR_second_prob.npy')
X_second_GBM = np.load('Data/Eval/GBM_second_prob.npy')
X_second_RF = np.load('Data/Eval/RF_second_prob.npy')
X_second_KNN = np.load('Data/Eval/KNN_second_prob.npy')
X_second_ETC = np.load('Data/Eval/ETC_second_prob.npy')

X_second.append(X_second_LR)
X_second.append(X_second_GBM)
X_second.append(X_second_RF)
X_second.append(X_second_KNN)
X_second.append(X_second_ETC)

X_second_feature = np.hstack(X_second)

#Prepare the testing set
X_second_test = []
X_second_LR_test = np.load('Data/Eval/LGR_prob.npy')
X_second_GBM_test = np.load('Data/Eval/GBM_prob.npy')
X_second_RF_test = np.load('Data/Eval/RF_prob.npy')
X_second_KNN_test = np.load('Data/Eval/knn_prob.npy')
X_second_ETC_test = np.load('Data/Eval/ETC_prob.npy')

X_second_test.append(X_second_LR_test)
X_second_test.append(X_second_GBM_test)
X_second_test.append(X_second_RF_test)
X_second_test.append(X_second_KNN_test)
X_second_test.append(X_second_ETC_test)

x_second_test_feature = np.hstack(X_second_test)

import WorkSpace.EN_optA
import WorkSpace.EN_optB

n_classes = 12

y = np.load('Data/y.npy')
#EN_optA
enA = WorkSpace.EN_optA.EN_optA(n_classes)
enA.fit(X_second_feature, y)
w_enA = enA.w
y_enA = enA.predict_proba(x_second_test_feature)
#print('{:20s} {:2s} {:1.7f}'.format('EN_optA:', 'logloss  =>', log_loss(y_test, y_enA)))
    
#Calibrated version of EN_optA 
cc_optA = CalibratedClassifierCV(enA, method='isotonic')
cc_optA.fit(X_second_feature, y)
y_ccA = cc_optA.predict_proba(x_second_test_feature)
#print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optA:', 'logloss  =>', log_loss(y_test, y_ccA)))
        
#EN_optB
enB = WorkSpace.EN_optB.EN_optB(n_classes) 
enB.fit(X_second_feature, y)
w_enB = enB.w
y_enB = enB.predict_proba(x_second_test_feature)
#print('{:20s} {:2s} {:1.7f}'.format('EN_optB:', 'logloss  =>', log_loss(y_test, y_enB)))

#Calibrated version of EN_optB
cc_optB = CalibratedClassifierCV(enB, method='isotonic')
cc_optB.fit(X_second_feature, y)
y_ccB = cc_optB.predict_proba(x_second_test_feature)  
#print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optB:', 'logloss  =>', log_loss(y_test, y_ccB)))
print('')

y_3l = (y_enA * 4./9.) + (y_ccA * 2./9.) + (y_enB * 2./9.) + (y_ccB * 1./9.)
#print('{:20s} {:2s} {:1.7f}'.format('3rd_layer:', 'logloss  =>', log_loss(y_test, y_3l)))