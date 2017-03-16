# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 10:52:02 2016

@author: shangyu

This code use different classifer to check its abililties and build a final second layer classifer
"""

import os
import numpy as np
import pandas as pd

os.chdir("/Users/shangyu/Documents/Kaggle/Airbnb") #Set your WorkSpace here
X = np.load("Data/X.npy")
y = np.load("Data/y.npy")
X_test = np.load("Data/X_test.npy")

# Set the mapping rules for target
train_users_path = 'train_users_2.csv'
df_train = pd.read_csv(train_users_path)
target = np.unique(df_train['country_destination'])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_target = le.fit_transform(target)

# Get the test ID
test_users_path = 'test_users.csv'
df_test = pd.read_csv(test_users_path)
test_id = df_test['id']

##############################Multi-layer Perceptron classifier####################
#from sklearn.neural_network import MLPClassifier
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(300, 100, 50), random_state=1)
#clf.fit(X, y)

##############################Logestic Regression####################
#score: 0.87725
from sklearn.linear_model import LogisticRegression
clf_LGR = LogisticRegression()
clf_LGR.fit(X, y)
pred_prob = clf_LGR.predict_proba(X_test)

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
sub.to_csv('Data/sub_lgr.csv',index=False)

##############################knn####################
#score: 0.85675
from sklearn.neighbors import KNeighborsClassifier
clf_KNN = KNeighborsClassifier(n_neighbors=30);
clf_KNN.fit(X, y)
pred_prob = clf_KNN.predict_proba(X_test)

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
sub.to_csv('Data/sub_knn.csv',index=False)

##############################Random Forest####################
# score: 0.88005
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state = 1)
clf_RF.fit(X, y)
pred_prob_RF = clf_RF.predict_proba(X_test)

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries

for i in range(X_test.shape[0]):
    idx = test_id[i]
    ids += [idx] * 5
    order = np.argsort(-pred_prob_RF[i,])[0: 5]
    cts = np.append(cts, target[order])

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('Data/sub_RF.csv',index=False)

##############################SVM####################
from sklearn.svm import SVC
clf_SVM = SVC(probability=True, random_state = 1)
clf_SVM.fit(X, y)
pred_prob_SVM = clf_SVM.predict_proba(X_test)

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries

for i in range(X_test.shape[0]):
    idx = test_id[i]
    ids += [idx] * 5
    order = np.argsort(-pred_prob_SVM[i,])[0: 5]
    cts = np.append(cts, target[order])

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('Data/sub_SVM.csv',index=False)