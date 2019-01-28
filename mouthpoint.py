# # coding=utf-8
# import cv2
# import dlib
#
# path = "1.jpg"
# img = cv2.imread(path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 人脸分类器
# detector = dlib.get_frontal_face_detector()
# # 获取人脸检测器
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#
# dets = detector(gray, 1)
# for face in dets:
#     shape = predictor(img, face)  # 寻找人脸的68个标定点
#     # 遍历所有点，打印出其坐标，并圈出来
#     for pt in shape.parts():
#         pt_pos = (pt.x, pt.y)
#         cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
#     cv2.imshow("image", img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import fnmatch  # filename match,主要作用是文件名称的匹配
import os
import skvideo.io
import numpy as np
import dlib
import cv2
import pandas as pd
from scipy.misc import imresize

SOURCE_PATH = r'C:\Users\zhengqi\Desktop\silence'
SOURCE_EXTS = r'*.mp4'
TARGET_PATH = r'C:\Users\zhengqi\Desktop\test\s\s1'
FACE_PREDICTOR_PATH = r'D:\LipNum\shape_predictor_68_face_landmarks.dat'


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename  # yield 的作用就是把一个函数变成一个generator


def get_video_frames(path):
    videogen = skvideo.io.vreader(path)
    frame = np.array([frame for frame in videogen])
    return frame


def get_frames_mouth(detector_x, predictor_x, frames_x):
    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 50
    HORIZONTAL_PAD = 0.19
    normalize_ratio = None
    mouth_frames_x = []
    for frame in frames_x:
        dets = detector_x(frame, 1)
        shape = None
        for k, d in enumerate(dets):
            shape = predictor_x(frame, d)
        if shape is None:  # Detector doesn't detect face, just return as is
            return frames_x
        mouth_points = []
        i = -1
        for part in shape.parts():
            i += 1
            if i < 48:  # Only take mouth region
                continue
            mouth_points.append((part.x, part.y))
        np_mouth_points = np.array(mouth_points)

        # print('np_mouth_points shape: '+str(np.shape(np_mouth_points)))

        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

        if normalize_ratio is None:
            mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
        resized_img = imresize(frame, new_img_shape)
        # cv2.imshow("image", resized_img)
        # cv2.waitKey(0)

        mouth_centroid_norm = mouth_centroid * normalize_ratio
        mouth_norm = np_mouth_points * normalize_ratio

        # print('mouth_norm shape: ' + str(np.shape(mouth_norm)))

        mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
        # mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
        mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
        # mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

        mouth_rel_x = mouth_norm[:, 0] - mouth_l
        mouth_rel_y = mouth_norm[:, 1] - mouth_t
        mouth_rel = np.hstack((mouth_rel_x, mouth_rel_y))
        # print('mouth_rel shape: ' + str(np.shape(mouth_rel)))
        # print(mouth_rel)
        # mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]

        mouth_frames_x.append(mouth_rel)
        # print('mouth_frames_x shape: ' + str(np.shape(mouth_frames_x)))
    np.array(mouth_frames_x)
    return mouth_frames_x


data = []
label = []
for filepath in find_files(SOURCE_PATH, SOURCE_EXTS):
    print("Processing: {}".format(filepath))
    frames = get_video_frames(filepath)
    filepath_wo_ext = os.path.basename(filepath)  # 获得文件名
    label.append(filepath_wo_ext[0])

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACE_PREDICTOR_PATH)
    mouth_frames = get_frames_mouth(detector, predictor, frames)
    mouth_point_mean = np.mean(mouth_frames, axis=0)
    # print('mouth_point_mean shape: ' + str(np.shape(mouth_point_mean)))
    # print(mouth_point_mean)
    data.append(mouth_point_mean)
# print(label)
# print(data)
dataset = pd.DataFrame(data=data)
dataset.to_csv('datasil.csv')
name = ['number']
label = pd.DataFrame(columns=name, data=label)
label.to_csv('labelsil.csv')
