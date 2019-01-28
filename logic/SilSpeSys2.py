# coding:utf-8
"""
功能：实现界面2.0的逻辑功能
作者：ZhengQi
"""

import sys
from PyQt5.QtCore import Qt
from UI import SilenceSpeechSys2
from PyQt5 import QtCore, QtWidgets
from logic import my_thread
import multiprocessing


class MouseDrag(QtWidgets.QMainWindow):
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


class MainWindow(object):
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        self.MainWindow = MouseDrag()
        self.ui = SilenceSpeechSys2.Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        # file = QtCore.QFile('base.qss')
        # file.open(QtCore.QFile.ReadOnly)
        # styleSheet = file.readAll()
        # styleSheet = unicode(styleSheet, encoding='utf8')
        # self.MainWindow.setStyleSheet(styleSheet)
        with open(r'D:\LipNum\logic\base.qss') as file:
            string = file.readlines()
            string = ''.join(string).strip('\n')
            app.setStyleSheet(string)

        # self.MainWindow.setWindowOpacity(0.9)  # 设置窗口透明度
        # self.MainWindow.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        # 去除窗口边框
        self.MainWindow.setWindowFlag(QtCore.Qt.FramelessWindowHint)

        self.close_window()
        self.visit_window()
        self.mini_window()
        self.show_function()
        self.start_and_stop_train_sys()
        self.start_and_stop_pre_sys()

        self.MainWindow.show()
        sys.exit(app.exec_())

    # 实现窗口关闭按钮功能
    def close_window(self):
        self.ui.closeButton.clicked.connect(QtWidgets.QApplication.instance().quit)

    # 实现窗口最小化按钮功能
    def mini_window(self):
        self.ui.miniButton.clicked.connect(self.MainWindow.showMinimized)

    # 实现窗口最大化与正常化按钮功能
    def visit_window(self):
        if self.MainWindow.isMaximized():
            self.ui.visitButton.clicked.connect(self.show_normal)
        else:
            self.ui.visitButton.clicked.connect(self.show_max)

    def show_normal(self):
        self.MainWindow.showNormal()
        self.ui.visitButton.clicked.connect(self.show_max)

    def show_max(self):
        self.MainWindow.showMaximized()
        self.ui.visitButton.clicked.connect(self.show_normal)

    # 实现实时显示摄像头内容功能
    def show_function(self):
        self.showThread = my_thread.ShowThread()
        self.showThread.breakSignal.connect(self.show_video)
        self.showThread.start()

    def show_video(self, qImg):
        self.ui.videoTrainLabel.setPixmap(qImg)
        self.ui.videoTrainLabel.setScaledContents(True)
        self.ui.videoPrelabel.setPixmap(qImg)
        self.ui.videoPrelabel.setScaledContents(True)

    # 实现训练系统的开始与停止
    def start_and_stop_train_sys(self):
        self.ui.startTrain.clicked.connect(self.start_train_thread)
        self.ui.stopTrain.clicked.connect(self.stop_train_thread)

    def start_train_thread(self):
        try:
            if hasattr(self, 'preThread'):
                self.preThread.stop()
        except AttributeError:
            pass
        self.trainThread = my_thread.TrainThread()
        self.trainThread.start()
        self.trainThread.order_value.connect(self.set_order_value)
        self.trainThread.pre_value.connect(self.set_pre_value)

    def stop_train_thread(self):
        try:
            self.trainThread.stop()
        except:
            pass

    def set_order_value(self, string):
        self.ui.showLabel_1.setText(string)

    def set_pre_value(self, string):
        self.ui.showLabel_2.setText(string)

    # 实现预测系统的开始与停止
    def start_and_stop_pre_sys(self):
        self.ui.startPreButton.clicked.connect(self.start_pre_thread)
        self.ui.stopPreButton.clicked.connect(self.stop_pre_thread)

    def start_pre_thread(self):
        try:
            if hasattr(self, 'trainThread'):
                self.preThread.stop()
        except AttributeError:
            pass
        self.preThread = my_thread.PreThread()
        self.preThread.start()
        self.preThread.predict.connect(self.set_predict)

    def stop_pre_thread(self):
        try:
            self.preThread.stop()
        except:
            pass

    def set_predict(self, string):
        self.ui.preNumLabel.setText(string)


if __name__ == "__main__":
    MainWindow()
