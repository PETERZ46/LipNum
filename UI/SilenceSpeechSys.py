# coding:utf-8

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import cv2
import numpy as np
from function import get_frames_mouth
from sklearn.externals import joblib
import random
import dlib
import time
import os
from PIL import Image
import qtawesome


# class Video:
#     def __init__(self, capture):
#         # 从gui的init中获取摄像头图片，存为capture
#         self.capture = capture
#         self.currentFrame = np.array([])
#
#     def captureFrame(self):
#         # 读取capture，获取frame，返回名readFrame
#         ret, readFrame = self.capture.read()
#         return readFrame
#
#     def captureNextFrame(self):
#         # 正确读取一帧后，进行颜色转换，并存入初始化的self.currentFrame中
#         ret, readFrame = self.capture.read()
#         if ret:
#             self.currentFrame = cv2.cvtColor(readFrame, cv2.COLOR_BGR2RGB)
#
#     def convertFrame(self):
#         try:
#             height, width = self.currentFrame.shape[:2]
#             img = QImage(self.currentFrame, width, height, QImage.Format_RGB888)
#             img = QPixmap.fromImage(img)
#             self.previousFrame = self.currentFrame
#             return img
#         except:
#             return None


class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # self.keyValue = QtCore.pyqtSignal(int)
        self.init_ui()
        self.showThread = ShowThread()
        self.showThread.breakSignal.connect(self.showVideo)
        self.showThread.start()
        # self.video = Video(cv2.VideoCapture(0))
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.play)
        # self.timer.start(27)
        # self.update()

    def init_ui(self):
        self.setFixedSize(960, 700)
        self.main_widget = QtWidgets.QWidget()  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局

        self.left_widget = QtWidgets.QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout)  # 设置左侧部件布局为网格

        self.right_widget = QtWidgets.QWidget()  # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout)  # 设置右侧部件布局为网格

        self.main_layout.addWidget(self.left_widget, 0, 0, 12, 1)  # 左侧部件在第0行第0列，占8行3列
        self.main_layout.addWidget(self.right_widget, 0, 1, 12, 10)  # 右侧部件在第0行第3列，占8行9列
        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        # 左侧的分布
        self.left_close = QtWidgets.QPushButton("")  # 关闭按钮
        self.left_visit = QtWidgets.QPushButton("")  # 空白按钮
        self.left_mini = QtWidgets.QPushButton("")  # 最小化按钮

        self.start_button = QtWidgets.QPushButton("开始训练")
        self.start_button.setObjectName('start_button')
        self.left_label_1 = QtWidgets.QLabel("请说:")
        self.left_label_1.setObjectName('left_label')
        self.left_label_2 = QtWidgets.QLabel("你说了：")
        self.left_label_2.setObjectName('left_label')
        self.stop_button = QtWidgets.QPushButton("停止训练")
        self.start_button.setObjectName('start_button')

        self.left_show_1 = QtWidgets.QLabel("")
        self.left_show_1.setObjectName('left_show')
        self.left_show_1.setAlignment(QtCore.Qt.AlignCenter)
        self.left_show_2 = QtWidgets.QLabel("")
        self.left_show_2.setObjectName('left_show')
        self.left_show_2.setAlignment(QtCore.Qt.AlignCenter)

        self.left_layout.addWidget(self.left_mini, 0, 0, 1, 1)
        self.left_layout.addWidget(self.left_visit, 0, 1, 1, 1)
        self.left_layout.addWidget(self.left_close, 0, 2, 1, 1)
        self.left_layout.addWidget(self.start_button, 1, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_1, 2, 0, 1, 3)
        self.left_layout.addWidget(self.left_show_1, 3, 0, 2, 3)
        self.left_layout.addWidget(self.left_label_2, 5, 0, 1, 3)
        self.left_layout.addWidget(self.left_show_2, 6, 0, 2, 3)
        self.left_layout.addWidget(self.stop_button, 8, 0, 1, 3)

        self.left_layout.setAlignment(Qt.AlignAbsolute)

        # 左侧部件的美化
        self.left_close.setFixedSize(15, 15)  # 设置关闭按钮的大小
        self.left_visit.setFixedSize(15, 15)  # 设置按钮大小
        self.left_mini.setFixedSize(15, 15)  # 设置最小化按钮大小

        self.left_close.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet(
            '''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet(
            '''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')
        self.left_widget.setStyleSheet('''
            QWidget#left_widget{
            background:gray;
            border-top:1px solid white;
            border-bottom:1px solid white;
            border-left:1px solid white;
            border-top-left-radius:10px;
            border-bottom-left-radius:10px;
            }
            QPushButton#start_button:hover{border-left:4px solid red;font-weight:700;}
            QLabel{border:none;color:black;}
            QLabel#left_label{
                border:none;
                border-bottom:1px solid white;
                font-size:18px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
            QLabel#left_show{
                border:none;
                border-bottom:1px solid white;
                font-size:18px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
        ''')

        # 右侧显示摄像头内容
        self.videoFrame = QLabel("VideoCapture")
        self.videoFrame.setObjectName('videoFrame')

        self.right_layout.addWidget(self.videoFrame, 0, 0, 1, 9)

        # 右侧部件的美化
        self.right_widget.setStyleSheet('''
            QWidget#right_widget{
                color:#232C51;
                background:white;
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-right:1px solid darkGray;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
            }
            QLabel#videoFrame{
                border:none;
                font-size:16px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
        ''')

        # self.setWindowOpacity(0.9)  # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        # 去除窗口边框
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.main_layout.setSpacing(0)

        # 实现窗口关闭、最小化
        self.left_close.clicked.connect(QApplication.instance().quit)
        self.left_mini.clicked.connect(self.showMinimized)
        if self.isMaximized():
            self.left_visit.clicked.connect(self.showNor)
        else:
            self.left_visit.clicked.connect(self.showMax)

        # 开始训练
        self.start_button.clicked.connect(self.start_TrainThread)
        # 停止训练
        self.stop_button.clicked.connect(self.stop_TrainThread)

    def showNor(self):
        self.showNormal()
        self.left_visit.clicked.connect(self.showMax)

    def showMax(self):
        self.showMaximized()
        self.left_visit.clicked.connect(self.showNor)

    def showVideo(self, qImg):
        self.videoFrame.setPixmap(qImg)
        self.videoFrame.setScaledContents(True)

    def start_TrainThread(self):
        self.trainThread = TrainThread(parent=None)
        self.trainThread.start()
        self.trainThread.order_value.connect(self.set_orderValue)
        self.trainThread.pre_value.connect(self.set_preValue)

    def stop_TrainThread(self):
        try:
            self.trainThread.stop()
        except:
            pass

    def set_orderValue(self, string):
        self.left_show_1.setText(string)

    def set_preValue(self, string):
        self.left_show_2.setText(string)

    # 以下三个函数实现窗口拖动
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_drag = True
            self.m_DragPosition = event.globalPos() - self.pos()
            event.accept()

    def mouseMoveEvent(self, QMouseEvent):
        if QMouseEvent.buttons() and Qt.LeftButton:
            self.move(QMouseEvent.globalPos() - self.m_DragPosition)
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_drag = False

    # def play(self):
    #     try:
    #         self.video.captureNextFrame()
    #         self.videoFrame.setPixmap(self.video.convertFrame())
    #         self.videoFrame.setScaledContents(True)
    #     except TypeError:
    #         print('No Frame')

    # 检测键盘按键，名字不能改，这是重写函数
    # def keyPressEvent(self, event):
    #     print("按下了：" + event.key())
        # if event.key() == Qt.Key_K:
        #     img = self.video.captureFrame()
        #     img.save(r'D:\LipNum\image\1.jpg')


class ShowThread(QtCore.QThread):
    breakSignal = QtCore.pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        super(ShowThread, self).__init__(parent)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def run(self):
        while True:
            # QtCore.QMutex.lock()
            ret, image = self.cap.read()
            # QtCore.QMutex.unlock()
            # convert image to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # get image infos
            height, width, channel = image.shape
            step = channel * width
            # create QImage from image
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            # show image in img_label
            self.breakSignal.emit(QPixmap.fromImage(qImg))


class TrainThread(QtCore.QThread):
    order_value = QtCore.pyqtSignal(str)
    pre_value = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(TrainThread, self).__init__(parent)

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        FACE_PREDICTOR_PATH = r'D:\LipNum\shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(FACE_PREDICTOR_PATH)
        modelpath = os.path.join(os.path.abspath('..'), 'model', 'svm_model.m')
        self.svmmodel = joblib.load(modelpath)

        true = True
        state = ['s', '1', '5', '8']
        ind = random.randint(0, 11)
        self.order_value.emit(str(state[ind % 4]))
        print('请说 ' + str(state[ind % 4]))
        while cap.isOpened():  # isOpened()  检测摄像头是否处于打开状态
            m = 0
            frames = []

            if true:
                print('正确!\n')
                ind = random.randint(0, 11)
                self.order_value.emit(str(state[ind % 4]))
                print('请说 ' + str(state[ind % 4]))
            else:
                print('错误！请继续说 ' + str(state[ind % 4]))
                self.order_value.emit(str(state[ind % 4]))

            # if k == 107:  # ASCLL(k)==127 keep
            time.sleep(1)
            while m < 10:
                ret, img = cap.read()  # 把摄像头获取的图像信息保存之img变量
                if ret:  # 如果摄像头读取图像成功
                    cv2.imwrite(r'image\image{}.jpg'.format(m), img)
                    frame = np.array(img)
                    frames.append(frame)
                    cv2.waitKey(60)
                    m += 1

            mouth_points = get_frames_mouth(detector, predictor, frames)
            mouth_points_mean = np.mean(mouth_points, axis=0)
            # print(mouth_points_mean.shape)
            mouth_points_mean_re = np.reshape(mouth_points_mean, newshape=[1, 24])
            # print(mouth_points_mean_re.shape)
            # mouth_points_mean_re.reshape(-1, 1)
            # print(mouth_points_mean_re.shape)

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

    def stop(self):
        print("线程停止中...")
        self.order_value.emit('')
        self.pre_value.emit('')
        self.terminate()


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
