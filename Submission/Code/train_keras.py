import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt


from Library.kaggle import *
from keras.models import Sequential
from keras.layers import Dense, Activation

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


def plot_result(name, x, y_train, y_test, param_name):
    print('plotting result')
    plt_labels = ['Train', 'Test']
    plt.plot(x, y_train,'or-')
    plt.plot(x, y_test,'sb-')
    plt.grid(True)
    plt.ylabel('Accuracy')
    plt.xlabel(param_name)
    plt.title('Random Forest Hyper Parameter Tuning Result')
    plt.legend(plt_labels)
    plt.savefig(name + '.png')
    plt.close()

data_dir = '/Users/Jucong/Documents/CS589/COMPSCI-589-HW2/Data/'
train_file_name = 'HW2.train.txt'
test_file_name = 'HW2.test.txt'
kaggle_file_name = 'HW2.kaggle.txt'

print('reading training data')
X_train, Y_train = read_data(data_dir + train_file_name)
print('finished reading training data')
print('X_train data shape is ', X_train.shape)
print('Y_train shape is ', Y_train.shape)

print('reading testing data')
X_test, Y_test = read_data(data_dir + test_file_name)
print('finished reading testing data')
print('testing_X data shape is ', X_test.shape)
print('Testing_Y shape is ', Y_test.shape)


X_kaggle, Y_kaggle = read_data(data_dir + kaggle_file_name)


model = Sequential()
model.add(Dense(64, input_dim=124, activation='sigmoid'))
model.add(Dense(128,activation='sigmoid'))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=100)
#
# kaggle_pred = best_rf.predict(X_kaggle)
# kaggleize(kaggle_pred, 'kaggle.csv')