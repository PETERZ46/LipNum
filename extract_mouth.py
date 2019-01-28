""""
extract_mouth.py
    This script will extract mouth crop of every single video inside source directory
    while preserving the overall structure of the source directory content.

Usage:
    python extract_mouth.py [source directory] [pattern] [target directory] [face predictor path]

    pattern: *.avi, *.mpg, etc

Example:
    python scripts/extract_mouth.py evaluation/samples/GRID/ *.mpg TARGET/ common/predictors/shape_predictor_68_face_landmarks.dat

    Will make directory TARGET and process everything inside evaluation/samples/GRID/ that match pattern *.mpg.
"""

import errno
import fnmatch  # filename match,主要作用是文件名称的匹配
import os
import sys
from videos import Video
from skimage import io

SOURCE_PATH = r'C:\Users\zhengqi\Desktop\test\s\s1'
SOURCE_EXTS = r'*.mp4'
TARGET_PATH = r'C:\Users\zhengqi\Desktop\test\s\s1'
FACE_PREDICTOR_PATH = r'D:\LipNum\shape_predictor_68_face_landmarks.dat'

# SOURCE_PATH = sys.argv[1]
# SOURCE_EXTS = sys.argv[2]
# TARGET_PATH = sys.argv[3]
# FACE_PREDICTOR_PATH = sys.argv[4]


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename  # yield 的作用就是把一个函数变成一个generator


for filepath in find_files(SOURCE_PATH, SOURCE_EXTS):
    print("Processing: {}".format(filepath))
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH).from_video(filepath)

    filepath_wo_ext = os.path.splitext(filepath)[0]  # splitext:将文件名和扩展名分开
    target_dir = os.path.join(TARGET_PATH, filepath_wo_ext)
    mkdir_p(target_dir)

    i = 0
    for frame in video.mouth:
        io.imsave(os.path.join(target_dir, "mouth_{0:03d}.png".format(i)), frame)
        i += 1
