"""
功能：用7帧照片的嘴唇外轮廓的平均值作为特征，训练svm分类器
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.externals import joblib

# load dataset
dataframe = pd.read_csv(r"dataset\dataset.csv", header=None)
dataset = dataframe.values
labelframe = pd.read_csv(r"dataset\labelset.csv", header=None)
labelset = labelframe.values
X = dataset[1:, 1:].astype(float)
X_re = np.reshape(X, newshape=[np.shape(X)[0], 2, 12])
Y = labelset[1:, 1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# parameters = {'kernel': ('linear', 'rbf'), 'C': [0.5, 0.8, 1, 2, 5], 'gamma': [0.125, 0.25, 0.5, 1, 2, 4]}
clf = SVC(C=0.7, kernel='linear', gamma=1, decision_function_shape='ovr')  # C=0.7, kernel='linear', gamma=1, decision_function_shape='ovr'
# clf = GridSearchCV(svm, parameters, n_jobs=-1,  iid=True, cv=5, return_train_score=True)
clf.fit(X_train, Y_train)
# cv_result = pd.DataFrame.from_dict(clf.cv_results_)
# with open('cv_result.csv', 'w') as f:
#     cv_result.to_csv(f)
# print('The parameters of best model are:')
# print(clf.best_params_)

joblib.dump(clf, r'model\svm_model.m')
pred = clf.predict(X_test)

print('训练集精度：' + str(clf.score(X_train, Y_train)))  # 精度
print('测试集精度：' + str(clf.score(X_test, Y_test)))

for m in range(np.shape(X_re)[0]):
    if Y[m] == '1':
        color = 'b'
        marker = 'o'
    elif Y[m] == '5':
        color = 'g'
        marker = '.'
    elif Y[m] == '8':
        color = 'y'
        marker = 's'
    else:
        color = 'r'
        marker = 'v'
    plt.scatter(X_re[m, 0, :], X_re[m, 1, :], c=color, marker=marker)
plt.show()

for m in range(np.shape(X)[0]):
    if Y[m] == '1':
        color = 'b'
        marker = 'o'
    elif Y[m] == '5':
        color = 'g'
        marker = '.'
    elif Y[m] == '8':
        color = 'y'
        marker = 's'
    else:
        color = 'r'
        marker = 'v'
    plt.scatter(range(24), X[m, :], c=color, marker=marker)
plt.show()

# encode class values as integers
# encoder = LabelEncoder()
# encoded_Y = encoder.fit_transform(Y)
# # convert integers to dummy variables (one hot encoding)
# dummy_y = np_utils.to_categorical(encoded_Y)
#
#





