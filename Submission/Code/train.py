import os
import numpy as np
import time
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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

classfiers = [RandomForestClassifier(), LogisticRegression(), SVC()]
best_pred_score = 0
best_classifer = None

for c in classfiers:
    print('training ', c)
    train_then = time.time()
    c.fit(X_train, Y_train)
    train_now = time.time()
    diff = train_now - train_then
    print('total training time is {} minute, {} second'.format(diff // 60, diff % 60))

    pred_then = time.time()
    y_pred = c.predict(X_test)
    pred_now = time.time()
    diff = pred_now - pred_then
    print('total training time is {} minute, {} second'.format(diff // 60, diff % 60))

    score = accuracy_score(Y_test, y_pred)
    print('accuracy is ', score)
    if score > best_pred_score:
        best_pred_score = score
        best_classifer = c

print('best classifier is ', best_classifer)
X_kaggle = None
Y_kaggle = None
if 'X_kaggle.pickle' in os.listdir():
    X_kaggle = pickle.load(open('X_kaggle.pickle','rb'))
    Y_kaggle = pickle.load(open('Y_kaggle.pickle','rb'))
else:
    X_kaggle, Y_kaggle = read_data(data_dir + kaggle_file_name)
    pickle.dump(X_train, open('X_kaggle.pickle','wb'))
    pickle.dump(Y_train, open('Y_kaggle.pickle','wb'))

kaggle_pred = best_classifer.predict(X_kaggle)
kaggleize(kaggle_pred, 'kaggle.csv')
# kf = KFold(n_splits=5, shuffle=True)
# kf.get_n_splits(X_train)
#
# print('KFold')
# for c in classfiers:
#     print('fitting data on ', c)
#     for train_index, test_index in kf.split(X_train):
#         X_train, X_test = X_train[train_index], X_train[test_index]
#         y_train, y_test = Y_train[train_index], Y_train[test_index]
#         c.fit(X_train, y_train)
#         y_pred = c.predict(X_test)
#         print(classification_report(y_test, y_pred, target_names=['1', '-1']))
