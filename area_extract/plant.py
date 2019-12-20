import cv2
import numpy as np
from skimage import draw,transform,feature
# 参数解释 https://blog.csdn.net/weixin_42904405/article/details/82814768
'''
这个代码是使用cv2.HoughCircles找圆
'''

planets = cv2.imread('./plant.png')
planets = cv2.imread('./line_on_head.png')
gray_img = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
# img = cv2.medianBlur(gray_img, 5)
canny = cv2.Canny(gray_img, 10, 200, apertureSize=3)

kernel = np.ones((3, 3), np.uint8)
dilate = cv2.dilate(canny, kernel, iterations=1)
cv2.imwrite('dilate.jpg', dilate)

cv2.imshow('canny', dilate)
cv2.waitKey(0)

print(dilate)
dilate1 = dilate > 0
print(dilate1.dtype.name)

circles = cv2.HoughCircles(dilate1, cv2.HOUGH_GRADIENT, 1, 120,
                            param1=100, param2=10, minRadius=10, maxRadius=50)

circles = np.uint16(np.around(circles))
print(circles)
for i in circles[0, :]:
    print('i', i[2])
    # draw the outer circle
    cv2.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imwrite("planets_circles.jpg", planets)
cv2.imshow("HoughCirlces", planets)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = transform.hough_ellipse(dilate, accuracy=20, threshold=250, min_size=50, max_size=30)
#result.sort(order='accumulator')  # 根据累加器排序
print(result)
