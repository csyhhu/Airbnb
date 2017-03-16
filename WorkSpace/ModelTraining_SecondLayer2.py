# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:53:10 2016

@author: aurelius
This program performs the second layer learning and predict
"""

import numpy as np
import pandas as pd
import os
import xgboost as xgb
#import 

from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
#from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

os.chdir('/media/aurelius/New Volume/Kaggle/Airbnb') # Set your workspace here

clf_svm = SVC(probability=True, random_state = 1)
clf_svm.fit(X_second_feature, y)
svm_pred = clf_svm.predict_proba(x_second_test_feature)

clf_LR = LogisticRegression(random_state = 1)
clf_LR.fit(X_second_feature, y)
LR_pred = clf_LR.predict_proba(x_second_test_feature)

clf_GBM = GradientBoostingClassifier(n_estimators=50, random_state = 1)
clf_GBM.fit(X_second_feature, y)
GBM_pred = clf_GBM.predict_proba(x_second_test_feature)
#cc_svm = CalibratedClassifierCV(clf_svm, method='isotonic')
#cc_svm.fit(X_second_feature, y)
#svm_pred_cc = cc_svm.predict_proba(x_second_test_feature)

np.save('Data/Eval/svm_second_output_pred', svm_pred)
np.save('Data/Eval/LR_second_output_pred', LR_pred)
np.save('Data/Eval/GBM_second_output_pred', GBM_pred)