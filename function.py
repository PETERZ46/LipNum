import numpy as np


# 提取一串照片（n）的嘴唇的外轮廓（12个点），返回一个n*2*12的矩阵
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

        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

        if normalize_ratio is None:
            mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        mouth_centroid_norm = mouth_centroid * normalize_ratio
        mouth_norm = np_mouth_points * normalize_ratio

        mouth_rel_x = mouth_norm[:, 0] - mouth_centroid_norm[0]
        mouth_rel_y = mouth_norm[:, 1] - mouth_centroid_norm[1]
        mouth_rel = np.vstack((mouth_rel_x, mouth_rel_y))

        mouth_frames_x.append(mouth_rel)
    mouth_pos = np.array(mouth_frames_x)
    return mouth_pos
