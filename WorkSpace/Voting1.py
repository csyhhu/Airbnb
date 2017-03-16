# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:35:59 2016

@author: aurelius

This program perform voting machainsic to get the final result
"""
import numpy as np
import os
import pandas as pd

os.chdir('/home/aurelius/Kaggle/Airbnb') # Set your workspace here

knn_prob = np.load('Data/knn_prob.npy')
LGR_prob = np.load('Data/LGR_prob.npy')
RF_prob = np.load('Data/RF_prob.npy')
xgboost_prob = np.load('Data/xgboost_prob.npy')

# Set the mapping rules for target
train_users_path = 'train_users_2.csv'
df_train = pd.read_csv(train_users_path)
target = np.unique(df_train['country_destination'])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_target = le.fit_transform(target)

X_test = np.load('Data/X_test.npy')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_target = le.fit_transform(target)

# Get the test ID
test_users_path = 'test_users.csv'
df_test = pd.read_csv(test_users_path)
test_id = df_test['id']

#############################################First Vote Method###############################
# Score: 0.87916
# Score: 0.87984 -- when double the xgboost probability
all_prob = knn_prob + LGR_prob + RF_prob + 2 * xgboost_prob

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries

for i in range(X_test.shape[0]):
    idx = test_id[i]
    ids += [idx] * 5
    order = np.argsort(-all_prob[i,])[0: 5]
    cts = np.append(cts, target[order])

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('Data/vote3.csv',index=False)

#############################################Second Vote Method###############################
# Score: 0.87903
#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries

all_prob_dict = {0: knn_prob, 1:LGR_prob, 2:RF_prob, 3:xgboost_prob}
for i in range(X_test.shape[0]):
    rank = np.zeros(12)
    for j in range(len(all_prob_dict)):
        order = np.argsort(-all_prob_dict[j][i])
        rank[order] += np.arange(12)[: :-1]
    idx = test_id[i]
    ids += [idx] * 5
    order2 = np.argsort(-rank)[0: 5]
    cts = np.append(cts, target[order2])

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('Data/vote2.csv',index=False)