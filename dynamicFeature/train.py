"""
功能：
"""
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.externals import joblib

# load dataset
dataframe = pd.read_csv(r"dataset\dynamic_dataset.csv", header=None)
dataset = dataframe.values
labelframe = pd.read_csv(r"dataset\dynamic_labelset.csv", header=None)
labelset = labelframe.values
X = dataset[1:, 1:].astype(float)
# X_re = np.reshape(X, newshape=[np.shape(X)[0], 12, 2, 12])
Y = labelset[1:, 1]
class_names = ['sil', '1', '5', '8']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

clf = SVC(C=0.04, kernel='linear', gamma=0.2, decision_function_shape='ovr')  # C=0.7, kernel='linear', gamma=1, decision_function_shape='ovr'
clf.fit(X_train, Y_train)

joblib.dump(clf, r'model\svm_model.m')
pred = clf.predict(X_test)

print('训练集精度：' + str(clf.score(X_train, Y_train)))  # 精度
print('测试集精度：' + str(clf.score(X_test, Y_test)))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# 计算混淆矩阵
cnf_matrix = confusion_matrix(Y_test, pred)
np.set_printoptions(precision=2)

# 画未归一化的混淆矩阵
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

# 画归一化的混淆矩阵
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()
# for m in range(np.shape(X_re)[0]):
#     if Y[m] == '1':
#         color = 'b'
#         marker = 'o'
#     elif Y[m] == '5':
#         color = 'g'
#         marker = '.'
#     elif Y[m] == '8':
#         color = 'y'
#         marker = 's'
#     else:
#         color = 'r'
#         marker = 'v'
#     plt.scatter(X_re[m, 0, :], X_re[m, 1, :], c=color, marker=marker)
# plt.show()
#
# for m in range(np.shape(X)[0]):
#     if Y[m] == '1':
#         color = 'b'
#         marker = 'o'
#     elif Y[m] == '5':
#         color = 'g'
#         marker = '.'
#     elif Y[m] == '8':
#         color = 'y'
#         marker = 's'
#     else:
#         color = 'r'
#         marker = 'v'
#     plt.scatter(range(24), X[m, :], c=color, marker=marker)
# plt.show()
