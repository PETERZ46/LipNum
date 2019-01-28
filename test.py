# 识别眼睛、嘴巴、人脸
import cv2

# image = cv2.imread('timg.jpg')
# 加载算法
face_detector = cv2.CascadeClassifier('D:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('D:\opencv\sources\data\haarcascades\haarcascade_eye.xml')
mouth_detector = cv2.CascadeClassifier('D:\opencv\sources\data\haarcascades\haarcascade_mcs_mouth.xml')
cap = cv2.VideoCapture('p.mp4')

while cap.isOpened():
    ok, cv_img = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(cv_img, code=cv2.COLOR_BGR2BGRA)

    face_zone = face_detector.detectMultiScale(gray, 1.3, 3, minSize=(80, 80))
    if face_zone is None:
        continue
    print(face_zone)
    for x, y, w, h in face_zone:
        cv2.rectangle(cv_img, pt1=(x, y), pt2=(x + w, y + h), color=[0, 0, 255], thickness=2)

    #  人脸切分
    h_up = int(face_zone[0, -1] * 0.6)
    x, y, w, h = face_zone.reshape(-1)
    # 头部
    head = gray[y:y + h, x:x + w]
    head_up = head[0:h_up]
    head_down = head[h_up:]
    # 检测眼睛
    eye_zone = eye_detector.detectMultiScale(head_up, 1.3, 3, minSize=(10, 10))
    for ex, ey, ew, eh in eye_zone:
        cv2.rectangle(cv_img, pt1=(ex + x, ey + y), pt2=(ex + ew + x, ey + eh + y), color=[0, 255, 0], thickness=1)

    # 检查嘴
    mouth_zone = mouth_detector.detectMultiScale(head_down, 1.3, 3, minSize=(10, 10))
    for mx, my, mw, mh in mouth_zone:
        cv2.rectangle(cv_img, pt1=(mx + x, my + y + h_up), pt2=(mx + mw + x, my + mh + y + h_up), color=[255, 0, 0],
                      thickness=1)

    cv2.imshow('liyong', cv_img)
    cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
