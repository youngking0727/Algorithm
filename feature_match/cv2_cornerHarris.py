# 特征检测是计算机对一张图像中最为明显的特征进行识别检测并将其勾画出来。大多数特征检测都会涉及图像的角点、边和斑点的识别、或者是物体的对称轴。
# 角点检测 是由Opencv的cornerHarris函数实现，其他函数参数说明如下：
'''
cv2.cornerHarris(src=gray, blockSize=9, ksize=23, k=0.04)
# cornerHarris参数：
# src - 数据类型为 float32 的输入图像。
# blockSize - 角点检测中要考虑的领域大小。
# ksize - Sobel 求导中使用的窗口大小
# k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06].
————————————————
版权声明：本文为CSDN博主「Xy-Huang」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/HuangZhang_123/article/details/80660688
'''
import cv2
import numpy as np

img = cv2.imread('image/target.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cornerHarris函数图像格式为 float32 ，因此需要将图像转换 float32 类型
gray = np.float32(gray)
# cornerHarris参数：
# src - 数据类型为 float32 的输入图像。
# blockSize - 角点检测中要考虑的领域大小。
# ksize - Sobel 求导中使用的窗口大小
# k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06].
dst = cv2.cornerHarris(src=gray, blockSize=9, ksize=23, k=0.04)
# 变量a的阈值为0.01 * dst.max()，如果dst的图像值大于阈值，那么该图像的像素点设为True，否则为False
# 将图片每个像素点根据变量a的True和False进行赋值处理，赋值处理是将图像角点勾画出来
a = dst > 0.01 * dst.max()
img[a] = [0, 0, 255]
# 显示图像
while True:
  cv2.imshow('corners', img)
  if cv2.waitKey(120) & 0xff == ord("q"):
    break
  cv2.destroyAllWindows()
