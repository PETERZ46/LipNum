"""
功能：实现实时异步预测
"""

import cv2
import numpy as np
import dlib
from function import get_frames_mouth
from sklearn.externals import joblib
import pandas as pd

# 基本绘图
cv2.namedWindow("Image")  # 创建窗口
# 抓取摄像头视频图像
cap = cv2.VideoCapture(cv2.CAP_DSHOW)  # 创建内置摄像头变量
cap.set(3, 352)  # 设置分辨率
cap.set(4, 288)

# load dataset
dataframe = pd.read_csv(r"dataset\dataset.csv", header=None)
dataset = dataframe.values
labelframe = pd.read_csv(r"dataset\labelset.csv", header=None)
labelset = labelframe.values
X = dataset[1:, 1:].astype(float)
X_re = np.reshape(X, newshape=[np.shape(X)[0], 2, 12])
Y = labelset[1:, 1]

FACE_PREDICTOR_PATH = r'D:\LipNum\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACE_PREDICTOR_PATH)
svmmodel = joblib.load(r'model\svm_model.m')

k = 0
while cap.isOpened():  # isOpened()  检测摄像头是否处于打开状态
    m = 0
    frames = []
    ret, img = cap.read()  # 把摄像头获取的图像信息保存之img变量
    if ret:  # 如果摄像头读取图像成功
        print(np.shape(img))
        cv2.imshow('Image', img)

    while m < 10:
        ret, img = cap.read()  # 把摄像头获取的图像信息保存之img变量
        if ret:  # 如果摄像头读取图像成功
            cv2.imshow('Image', img)
            cv2.imwrite(r'image\image{}.jpg'.format(m), img)
            frame = np.array(img)
            frames.append(frame)
            k = cv2.waitKey(5)
            m += 1

    mouth_points = get_frames_mouth(detector, predictor, frames)
    if np.shape(mouth_points) != (np.shape(mouth_points)[0], 2, 12):
        continue
    mouth_points_mean = np.mean(mouth_points, axis=0)
    # print(mouth_points_mean.shape)
    mouth_points_mean_re = np.reshape(mouth_points_mean, newshape=[1, 24])
    # print(mouth_points_mean_re.shape)
    # mouth_points_mean_re.reshape(-1, 1)
    # print(mouth_points_mean_re.shape)

    pred = svmmodel.predict(mouth_points_mean_re)
    print('我认为你现在说了 ' + pred)

    # plt.figure(1)
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
    # plt.scatter(mouth_points_mean[0, :], mouth_points_mean[1, :], c='k', marker='x')
    # plt.show()
    k = cv2.waitKey(1)
    if k == 27:  # esc==27
        break
cap.release()  # 关闭摄像头
cv2.destroyWindow("Image")
