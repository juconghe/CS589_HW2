import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from Library.kaggle import *


def read_data(data_path):
    temp_X = None
    temp_Y = np.array([])
    with open(data_path, 'r') as train_data:
        for data in train_data:
            raw = data.split(' ')
            temp_Y = np.append(temp_Y, raw[0])
            temp = np.zeros(124)
            for feature in raw[1:]:
                index, value = int(feature.split(':')[0]), float(feature.split(':')[1])
                temp[index - 1] = value
            if temp_X is None:
                temp_X = np.array([temp])
            else:
                temp_X = np.append(temp_X, [temp], axis=0)
    return temp_X, temp_Y


data_dir = '/Users/Jucong/Documents/CS589/COMPSCI-589-HW2/Data/'
train_file_name = 'HW2.train.txt'
test_file_name = 'HW2.test.txt'
kaggle_file_name = 'HW2.kaggle.txt'

print('reading training data')
X_train = None
Y_train = None
if 'X_train.pickle' in os.listdir():
    X_train = pickle.load(open('X_train.pickle','rb'))
    Y_train = pickle.load(open('Y_train.pickle','rb'))
else:
    X_train, Y_train = read_data(data_dir + train_file_name)
    pickle.dump(X_train, open('X_train.pickle','wb'))
    pickle.dump(Y_train, open('Y_train.pickle','wb'))

print('finished reading training data')
print('X_train data shape is ', X_train.shape)
print('Y_train shape is ', Y_train.shape)

X_test = None
Y_test = None
print('reading testing data')
if 'X_test.pickle' in os.listdir():
    X_test = pickle.load(open('X_test.pickle','rb'))
    Y_test = pickle.load(open('Y_test.pickle','rb'))
else:
    X_test, Y_test = read_data(data_dir + test_file_name)
    pickle.dump(X_train, open('X_test.pickle','wb'))
    pickle.dump(Y_train, open('Y_test.pickle','wb'))
print('finished reading training data')
print('testing_X data shape is ', X_test.shape)
print('Testing_Y shape is ', Y_test.shape)

X_kaggle, Y_kaggle = read_data(data_dir + kaggle_file_name)

hyper_parameters = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
param_result = defaultdict(list)
kf = KFold(n_splits=5, shuffle=True)
kf.get_n_splits(X_train)

print('Hyper parameter tuning')
for train_index, test_index in kf.split(X_train):
    x_train, x_test = X_train[train_index], X_train[test_index]
    y_train, y_test = Y_train[train_index], Y_train[test_index]
    for est in hyper_parameters:
        rf = RandomForestClassifier(n_estimators=est)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        param_result[est].append(score)
print('Hyper parameter tuning Done')

param_avg = {k:np.mean(param_result[k]) for k in param_result.keys()}
best_param = max(param_avg, key=param_avg.get)
print('best param', best_param)

testing_score = []
best_rf = None
best_score = 0
print('Getting testing score for all hyper parameters')
for est in hyper_parameters:
    rf = RandomForestClassifier(n_estimators=est)
    rf.fit(X_train, Y_train)
    y_pred = rf.predict(X_test)
    score = accuracy_score(Y_test, y_pred)
    testing_score.append(score)
    print(est, score)
    if score > best_score:
        best_rf = rf
        best_score = score

print('plotting result')
plt_labels = ['Train', 'Test']
plt.plot(hyper_parameters, list(param_avg.values()),'or-')
plt.plot(hyper_parameters, testing_score,'sb-')
plt.grid(True)
plt.ylabel('Accuracy')
plt.xlabel('n_estimators')
plt.title('Random Forest Hyper Parameter Tuning Result')
plt.legend(plt_labels)
plt.savefig('result.png')
plt.close()

kaggle_pred = best_rf.predict(X_kaggle)
kaggleize(kaggle_pred, 'kaggle.csv')