import cv2
import numpy as np
import imutils
import sys
from os.path import join, dirname, realpath
'''
边缘检测后，二值化，膨胀操作以后, 找区域，适合抠尺子
'''

sys.path.append(join(dirname(realpath(__file__)), ".."))
from utils.math_utils import iou, IOU, get_board_area

# image_path = 'bending_frame.jpg'
# image_path = 'new_image.jpg'
image_path = 'jump_raw.jpg'
# image_path = '_image.jpg'
# image_path = 'test.png'

cap = cv2.VideoCapture('jump_skip01.mov')

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.Canny(frame, 60, 200, apertureSize=3)
    canny = cv2.Canny(frame, 10, 200, apertureSize=3)
    print(canny.shape)
    # cv2.imshow('canny', canny)
    # cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)
    #cv2.imshow('dilation2', dilate)
    #cv2.waitKey(0)
    cv2.imshow('frame', dilate)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''
#ruler_area = [244, 247, 396, 256]  # 左上，右q下坐标
ruler_area = [327, 193, 476, 200]  # new_image

image = cv2.imread(image_path)  # 读取成灰度图
print(image.shape)
frame = image.copy()

cv2.imshow('image', image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)

# 中值滤波
medianBlur = cv2.medianBlur(gray, 7)

cv2.imshow('medianBlur', medianBlur)
cv2.waitKey(0)

# 二值化操作
ret, binary = cv2.threshold(medianBlur, 90, 255, cv2.THRESH_BINARY)

cv2.imshow('binary', binary)
cv2.waitKey(0)

# 膨胀操作，把尺子分开的区域连起来
kernel = np.ones((3, 3), np.uint8)
dilate = cv2.dilate(binary, kernel, iterations=1)

cv2.imshow('dilation', dilate)
cv2.waitKey(0)

canny = cv2.Canny(gray, 60, 200, apertureSize=3)
cv2.imshow('canny', canny)
cv2.waitKey(0)


cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# 返回交并比，后面匹配tracker的时候也会用到
iou_threshold = 0
ruler_box = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w * h < 10:
        continue
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    box = [x, y, x+w, y+h]
    iou_value = IOU(ruler_area, box)
    print(iou_value, x, y, w, h)

    if iou_value > iou_threshold:
        ruler_box = [x, y, w, h]
        iou_threshold = iou_value

x, y, w, h = ruler_box
cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
cv2.rectangle(image, (ruler_area[0], ruler_area[1]), (ruler_area[2], ruler_area[3]), (0, 0, 255), 1)


cv2.imshow('image2', image)
cv2.waitKey(0)

ruler_box = [x, y, x+w+1, y+h]
get_board_area(ruler_area, frame)
'''





