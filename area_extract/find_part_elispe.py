#!/usr/bin/env python
# -*- coding:utf-8 -*-
# -------------------------------------------------------------------------
#
# 这个代码主要目的是寻找近椭圆曲线，难点在于曲线只是部分曲线，抠出来就不好抠，抠出来以后
# 再找也不好找。
#
# -------------------------------------------------------------------------
# @author: kai yang
# @file: find_part_ellipse.py
# @time: 2019-09-11 21:35
# -------------------------------------------------------------------------
import cv2
import imutils
import numpy as np
from skimage import draw,transform,feature

__author__ = 'yang kai'
'''
这个代码基于main.py，找出区域以后，用cv2.fitEllipse找到区域的最大内接圆
https://jingyan.baidu.com/article/9113f81b5e9a3c2b3214c731.html
'''

image_path = './line_on_head.png'
# image_path = './no_line.png'
image_path = './line_body.png'
image_path = './plant.png'
# image_path = './jump_raw.jpg'

image = cv2.imread(image_path)

cv2.imshow('image', image)
cv2.waitKey(0)
# image = cv2.GaussianBlur(image, (3, 3), 0) # 如果进行了这一步很容易把绳子弄没，但不进行会有很多无关边缘
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)
# 中值滤波
medianBlur = cv2.medianBlur(gray, 7)
cv2.imshow('medianBlur', medianBlur)
cv2.waitKey(0)

# 二值化
ret, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow('binary', binary)
cv2.waitKey(0)

# 为了保证绳子更好的被识别出来，应该对图像做膨胀操作或者开运算
kernel = np.ones((3, 3), np.uint8)
dilate = cv2.dilate(gray, kernel, iterations=1)
cv2.imshow('dilation', dilate)
cv2.waitKey(0)

canny = cv2.Canny(gray, 10, 200, apertureSize=3)
cv2.imshow('canny', canny)
cv2.waitKey(0)

kernel = np.ones((3, 3), np.uint8)
dilate = cv2.dilate(canny, kernel, iterations=1)
cv2.imshow('dilation2', dilate)
cv2.waitKey(0)


cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]


for cnt in cnts:
    print(len(cnts))
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    ellipse = cv2.fitEllipse(cnt)
    img = cv2.ellipse(image, ellipse, (0, 255, 0), 2)

    '''
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    img = cv2.line(image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
    '''
    circles = cv2.HoughCircles(dilate, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=40, minRadius=10, maxRadius=50)
    print(circles)
    # 整数化，#把circles包含的圆心和半径的值变成整数
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # 画出外边圆
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # 画出圆心
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

# 到这里已经有了线，现在需要找椭圆了
cv2.imshow('dilation3', image)
cv2.waitKey(0)
