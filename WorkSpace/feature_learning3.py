# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:08:49 2016

@author: Arthur
This code use xgboost to train the model
The score is pretty bad: 0.34529
"""
import numpy as np
import xgboost as xgb
import os
import pandas as pd

os.chdir('/home/aurelius/Kaggle/Airbnb') # Set your workspace here

# The 'trainX.csv' data file and etc. can be generated from the preprocess.R
train_X_NDF = np.matrix(pd.read_csv('trainX.csv').ix[1:,2:])
train_Y_NDF = np.matrix(pd.read_csv('trainY_NDF.csv').ix[1:,1:])
train_Y_NDF = np.ravel(train_Y_NDF)
xg_train = xgb.DMatrix( train_X_NDF, label = train_Y_NDF)

param = {}
param['objective'] = 'binary:logistic'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4

#watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 50
bst_NDF = xgb.train(param, xg_train, num_round)
bst_NDF.save_model('bst_NDF.model')
print ('Train Completed')

#bst2 = xgb.Booster(model_file='bst_NDF.model')

# Read in the test data
test_X_NDF = np.matrix(pd.read_csv('testX.csv').ix[1:,2:])
xg_test = xgb.DMatrix(test_X_NDF)
pred_NDF =  bst_NDF.predict(xg_test)

NDF_index = np.array([], dtype = 'int')
NONDF_index = np.array([], dtype = 'int')
for index, pred in enumerate(pred_NDF):
    if pred > 0.5:
        NDF_index = np.append(NDF_index, index)
    else:
        NONDF_index = np.append(NONDF_index, index)

###################################################################################################
# Attain the US train data
train_X_US = np.matrix(pd.read_csv('trainX_US.csv').ix[1:,3:])
train_Y_US = np.matrix(pd.read_csv('trainY_US.csv').ix[1:,1:])
train_Y_US = np.ravel(train_Y_US)
xg_train = xgb.DMatrix( train_X_US, label = train_Y_US) 

bst_US = xgb.train(param, xg_train, num_round)
bst_US.save_model('bst_US.model')

# Attain the NON-NDF test data
test_X_US = test_X_NDF[NONDF_index]
xg_test = xgb.DMatrix(test_X_US)
pred_US =  bst_US.predict(xg_test)

US_index = np.array([], dtype = 'int')
NOUS_index = np.array([], dtype = 'int')
for index, pred in enumerate(pred_US):
    if pred > 0.5:
        US_index = np.append(US_index, index)
    else:
        NOUS_index = np.append(NOUS_index, index)
        
##################################################################################################
# Attain the NON-US train data
train_X_Others = np.matrix(pd.read_csv('trainX_Others.csv').ix[1:,3:])
train_Y_Others = np.matrix(pd.read_csv('trainY_Others.csv').ix[1:,1:])
train_Y_Others = np.ravel(train_Y_Others)
train_Y_Others = train_Y_Others.astype('int')

xg_train = xgb.DMatrix( train_X_Others, label = train_Y_Others)

param = {}
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 10
num_round = 30
bst_Others = xgb.train(param, xg_train, num_round)
bst_Others.save_model('bst_Others.model')
print ('Train Completed')

# Attain the others test data
test_X_Others = test_X_US[NOUS_index]
xg_test = xgb.DMatrix(test_X_Others)
pred_Others = bst_Others.predict(xg_test)
mapping_rules = {0:'other', 1:'FR', 2:'CA', 3:'GB', 4:'ES',
                 5:'IT', 6:'PT', 7:'NL', 8:'DE', 9:'AU'}
                 
##################################################################################################
# Generate Final Result  
test_X_ID = np.matrix(pd.read_csv('test_users.csv').ix[1:,0])
submision = file('submission.csv','w')
submision.write('id,country\n')
test_NDF_ID = test_X_ID[0, NDF_index]
for i in range(len(NDF_index)):
    submision.write(test_NDF_ID[0,i] + ',' + 'NDF\n')

test_NONNDF_ID = test_X_ID[0, NONDF_index]
test_US_ID = test_NONNDF_ID[0, US_index]
for i in range(len(US_index)):
    submision.write(test_US_ID[0,i] + ',' + 'US\n')

test_Other_ID = test_NONNDF_ID[0, NOUS_index]
for i in range(len(NOUS_index)):
    submision.write(test_Other_ID[0, i] + ',' + mapping_rules[np.argmax(pred_Others[i])] + '\n')

submision.close()