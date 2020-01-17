import cv2
import numpy as np
import time
import imutils

def absdiff_demo(image_1, image_2, sThre):
    gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)  # 灰度化
    gray_image_1 = cv2.GaussianBlur(gray_image_1, (3, 3), 0)  #高斯滤波
    gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    gray_image_2 = cv2.GaussianBlur(gray_image_2, (3, 3), 0)
    d_frame = cv2.absdiff(gray_image_1, gray_image_2)
    ret, d_frame = cv2.threshold(d_frame, sThre, 255, cv2.THRESH_BINARY)
    return d_frame

capture = cv2.VideoCapture('image/jump_skip01.mov')
#capture = cv2.VideoCapture("image/front_jump_skip.mp4")
capture = cv2.VideoCapture("image/front_jeston_1.mp4")
#capture = cv2.VideoCapture('image/side_jump_skip.mp4')
capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture("side_jump_skip.mp4")
sThre = 5  # sThre表示像素阈值
i = 0
while True:
    ret, frame = capture.read()
    if i == 0:
        cv2.waitKey(660)
    i = i + 1
    ret_2, frame_2 = capture.read()
    gray_1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
    # frame = cv2.cvtColor(gray_1, cv2.COLOR_GRAY2BGR)
    # frame_2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    d_frame = absdiff_demo(frame, frame_2, sThre)
    # 去掉那些小点，kernel的尺寸自行选择
    d_frame = cv2.morphologyEx(d_frame, cv2.MORPH_OPEN, np.ones((3, 1), np.uint8))
    cnts = cv2.findContours(d_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 10:
            continue
        cv2.rectangle(d_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
    cv2.putText(d_frame, str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

    '''
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(d_frame, kernel, iterations=1)
    lines = cv2.HoughLines(edges, 1, np.pi / 500, 20)  # rho=1，theta=np.pi/180
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(frame_2, (x1, y1), (x2, y2), (0, 0, 255), 2)
    '''
    # [27,33,40,51,82,87,88,103,104,105,113,114,180,181]
    # time.sleep(0.5)
    cv2.imshow('diff', d_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()



'''
————————————————
版权声明：本文为CSDN博主「weixin_41987641」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_41987641/article/details/81910450
'''