import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_svmlight_file
from Library.kaggle import *

def read_data(FILE, multilabel=None):

    if (multilabel): # for kaggle data
        data = load_svmlight_file(FILE, multilabel=True)
        return data[0].toarray()
    # for training and testing data
    data = load_svmlight_file(FILE)
    return data[0].toarray(), data[1]


def plot_result(name,title, x, y_train, param_name):
    print('plotting result')
    plt.plot(x, y_train,'or-')
    plt.grid(True)
    plt.ylabel('Accuracy')
    plt.xlabel(param_name)
    plt.title(title)
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


X_kaggle = read_data(data_dir + kaggle_file_name, multilabel=True)


tuned_parameters = {'n_estimators': [50, 100, 150, 200, 250, 300],
                    'max_features': ['sqrt', 'auto'],
                    }

clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, verbose=5, n_jobs=-1, return_train_score=True)

mean_train_score = clf.cv_result['mean_train_score']
mean_test_score = clf.cv_result['mean_test_score']

clf.fit(X_train, Y_train)
print('Getting the result of the CV')
plot_result('result_sqrt', 'Random Forest Training Error With Sqrt Max Feature',tuned_parameters['n_estimators'],
            mean_train_score[:6],'n_estimators')
plot_result('result_auto', 'Random Forest Training Error With Auto Max Feature',tuned_parameters['n_estimators'],
            mean_train_score[6:], 'n_estimators')

y_pred = clf.predict(X_test)
print(accuracy_score(Y_test, y_pred))

kaggle_pred = clf.predict(X_kaggle)
kaggleize(kaggle_pred, 'kaggle.csv')
