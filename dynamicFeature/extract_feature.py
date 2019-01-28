import fnmatch  # filename match,主要作用是文件名称的匹配
import os
import skvideo.io
from skimage import io
import numpy as np
import dlib
import matplotlib.pyplot as plt
import pandas as pd

SOURCE_PATH = r'C:\Users\zhengqi\Desktop\s1 (1)'
SOURCE_EXTS = r'*.mp4'
FACE_PREDICTOR_PATH = r'D:\LipNum\shape_predictor_68_face_landmarks.dat'


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename  # yield 的作用就是把一个函数变成一个generator


def get_video_frames(path):
    videogen = skvideo.io.vreader(path)
    # io.imsave(r'image\zyy.jpg', [frame for frame in videogen][0])
    frame = np.array([frame for frame in videogen])
    return frame


# 返回n*2*12的数组，表示连续n帧嘴唇外轮廓的坐标
def get_frames_mouth(detector_x, predictor_x, frames_x):
    MOUTH_WIDTH = 100
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
            if i < 48 or i > 59:  # Only take mouth region
                continue
            mouth_points.append((part.x, part.y))
        np_mouth_points = np.array(mouth_points)

        # print('np_mouth_points shape: '+str(np.shape(np_mouth_points)))

        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

        if normalize_ratio is None:
            mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        # new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
        # resized_img = imresize(frame, new_img_shape)
        # # cv2.imshow("image", resized_img)
        # # cv2.waitKey(0)

        mouth_centroid_norm = mouth_centroid * normalize_ratio
        mouth_norm = np_mouth_points * normalize_ratio

        # print('mouth_norm shape: ' + str(np.shape(mouth_norm)))

        # mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
        # mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
        # mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
        # mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

        mouth_rel_x = mouth_norm[:, 0] - mouth_centroid_norm[0]
        mouth_rel_y = mouth_centroid_norm[1] - mouth_norm[:, 1]
        mouth_rel = np.vstack((mouth_rel_x, mouth_rel_y))
        # print('mouth_rel shape: ' + str(np.shape(mouth_rel)))
        # print(mouth_rel)
        # mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]

        mouth_frames_x.append(mouth_rel)
        # print('mouth_frames_x shape: ' + str(np.shape(mouth_frames_x)))
    mouth_pos = np.array(mouth_frames_x)
    return mouth_pos


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
    print(np.shape(mouth_frames))
    # mouth_frames_mean = np.mean(mouth_frames, axis=0)
    mouth_frames_re = np.reshape(mouth_frames, newshape=[288, ])
    data.append(mouth_frames_re)

#     if filepath_wo_ext[0] == '1':
#         color = 'b'
#         marker = 'o'
#     elif filepath_wo_ext[0] == '5':
#         color = 'g'
#         marker = '.'
#     elif filepath_wo_ext[0] == '8':
#         color = 'y'
#         marker = 's'
#     else:
#         color = 'r'
#         marker = 'v'
#     plt.scatter(mouth_frames_mean[0, :], mouth_frames_mean[1, :], c=color, marker=marker)
# plt.show()

dataset = pd.DataFrame(data=data)
dataset.to_csv(r'dataset\dynamic_dataset.csv')
name = ['state']
labelset = pd.DataFrame(columns=name, data=label)
labelset.to_csv(r'dataset\dynamic_labelset.csv')
