# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 11:25:03 2016

@author: aurelius
This script use Sandro's feature to train the classifier.
This feature seems good, it can reach to 0.88 score simply using a xgboost
"""
import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

os.chdir('/home/aurelius/Kaggle/Airbnb') # Set your workspace here

# Read in the data
X = np.load('Data/X.npy')
y = np.load('Data/y.npy')
X_test = np.load('Data/X_test.npy')

xg_train = xgb.DMatrix(X, label = y)

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 12

num_round = 5
################Multi-label test##############################
# Prepare the test set
xg_test_X = xgb.DMatrix(X_test)

param['objective'] = 'multi:softprob'
xg_train = xgb.DMatrix(X, label = y)
bst = xgb.train(param, xg_train, num_round);

pred_prob = bst.predict( xg_test_X ).reshape( X_test.shape[0], 12 )

# Set the mapping rules for target
train_users_path = 'train_users_2.csv'
df_train = pd.read_csv(train_users_path)
target = np.unique(df_train['country_destination'])
le = LabelEncoder()
y_target = le.fit_transform(target)

# Get the test ID
test_users_path = 'test_users.csv'
df_test = pd.read_csv(test_users_path)
test_id = df_test['id']

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries

for i in range(X_test.shape[0]):
    idx = test_id[i]
    ids += [idx] * 5
    order = np.argsort(-pred_prob[i,])[0: 5]
    cts = np.append(cts, target[order])

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)
    