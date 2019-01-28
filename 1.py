import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load dataset
dataframe = pd.read_csv(r"dataset\dataset.csv", header=None)
dataset = dataframe.values
labelframe = pd.read_csv(r"dataset\labelset.csv", header=None)
labelset = labelframe.values
X = dataset[1:, 1:].astype(float)
X_re = np.reshape(X, newshape=[np.shape(X)[0], 2, 12])
X_add = X_re[:, :, 0]
X_plot = np.dstack((X_re, X_add))
Y = labelset[1:, 1]

plt.figure(1)
plt.xlim(-25, 25)
plt.ylim(-20, 20)
number = '8'
for m in range(np.shape(X_plot)[0]):
    if Y[m] == number:
        color = 'b'
        marker = 'o'
        plt.plot(X_plot[m, 0, :], X_plot[m, 1, :], c=color, marker=marker)
    # elif Y[m] == '5':
    #     color = 'g'
    #     marker = '.'
    # elif Y[m] == '8':
    #     color = 'y'
    #     marker = 's'
    # else:
    #     color = 'r'
    #     marker = 'v'
    # plt.plot(X_plot[m, 0, :], X_plot[m, 1, :], c=color, marker=marker)
plt.title("lip's outer contour of {}".format(number))
plt.grid()
plt.show()

