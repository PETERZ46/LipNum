from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QImage
from sklearn.externals import joblib
import numpy as np
import cv2
import time
import dlib
import random
import speech
import os
from function import get_frames_mouth


class ShowThread(QtCore.QThread):
    breakSignal = QtCore.pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        super(ShowThread, self).__init__(parent)
        self.cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        # self.cap.set(cv2.CAP_PROP_FPS, 15)

    def run(self):
        while True:
            # QtCore.QMutex.lock()
            ret, image = self.cap.read()
            # QtCore.QMutex.unlock()
            # convert image to RGB format
            cv2.waitKey(50)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # get image infos
            height, width, channel = image.shape
            step = channel * width
            # create QImage from image
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            # show image in img_label
            self.breakSignal.emit(QPixmap.fromImage(qImg))


# 训练线程
class TrainThread(QtCore.QThread):
    order_value = QtCore.pyqtSignal(str)
    pre_value = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(TrainThread, self).__init__(parent)
        self.cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        self.order_value.emit('')
        self.pre_value.emit('')
        FACE_PREDICTOR_PATH = 'D:\LipNum\shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(FACE_PREDICTOR_PATH)
        # model_path = os.path.join(os.path.abspath('..'), 'model', 'svm_model.m')
        self.svmmodel = joblib.load('D:\LipNum\model\svm_model.m')

    def run(self):
        state = ['s', '1', '5', '8']
        ind = random.randint(0, 11)
        self.order_value.emit(str(state[ind % 4]))
        if str(state[ind % 4]) == 's':
            speech.say("请保持沉默。")
        else:
            speech.say("请说" + str(state[ind % 4]))
        print('请说 ' + str(state[ind % 4]))
        while True:
            m = 0
            frames = []
            speech.say(3)
            speech.say(2)
            speech.say(1)
            while m < 3:
                ret, img = self.cap.read()
                cv2.waitKey(33)
                if ret:  # 如果摄像头读取图像成功
                    frame = np.array(img)
                    frames.append(frame)
                    # cv2.waitKey(60)
                    m += 1

            mouth_points = get_frames_mouth(self.detector, self.predictor, frames)
            print(np.shape(mouth_points))
            if np.shape(mouth_points) != (np.shape(mouth_points)[0], 2, 12):
                continue
            mouth_points_mean = np.mean(mouth_points, axis=0)
            mouth_points_mean_re = np.reshape(mouth_points_mean, newshape=[1, 24])

            pred = self.svmmodel.predict(mouth_points_mean_re)
            self.pre_value.emit(pred[0])
            print('我认为你说了 ' + pred[0])
            if pred == state[ind % 4]:
                true = True
            else:
                true = False
            time.sleep(2)
            self.order_value.emit('')
            self.pre_value.emit('')

            if true:
                print('正确!\n')
                speech.say("正确。")
                ind = random.randint(0, 11)
                self.order_value.emit(str(state[ind % 4]))
                if str(state[ind % 4]) == 's':
                    speech.say("请保持沉默。")
                else:
                    speech.say("请说" + str(state[ind % 4]))
                print('请说 ' + str(state[ind % 4]))
            else:
                speech.say("我认为你说错了，请继续说" + str(state[ind % 4]))
                print('错误！请继续说 ' + str(state[ind % 4]))
                self.order_value.emit(str(state[ind % 4]))

    def stop(self):
        print("训练线程停止中...")
        self.order_value.emit('')
        self.pre_value.emit('')
        self.terminate()


# 预测线程
class PreThread(QtCore.QThread):
    predict = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(PreThread, self).__init__(parent)
        self.cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        FACE_PREDICTOR_PATH = 'D:\LipNum\shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(FACE_PREDICTOR_PATH)
        # model_path = os.path.join(os.path.abspath('..'), 'model', 'svm_model.m')
        self.svmmodel = joblib.load('D:\LipNum\model\svm_model.m')

    def run(self):
        while self.cap.isOpened():  # isOpened()  检测摄像头是否处于打开状态
            m = 0
            frames = []
            while m < 2:
                ret, img = self.cap.read()  # 把摄像头获取的图像信息保存之img变量
                cv2.waitKey(33)
                if ret:  # 如果摄像头读取图像成功
                    frame = np.array(img)
                    frames.append(frame)
                    m += 1

            mouth_points = get_frames_mouth(self.detector, self.predictor, frames)
            print(np.shape(mouth_points))
            if np.shape(mouth_points) != (np.shape(mouth_points)[0], 2, 12):
                continue
            mouth_points_mean = np.mean(mouth_points, axis=0)
            mouth_points_mean_re = np.reshape(mouth_points_mean, newshape=[1, 24])

            pred = self.svmmodel.predict(mouth_points_mean_re)
            self.predict.emit(pred[0])
            print('我认为你现在说了 ' + pred[0])
            # time.sleep(1)

    def stop(self):
        print("预测线程停止中...")
        self.predict.emit('')
        self.terminate()
