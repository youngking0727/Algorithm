# -*- coding: cp936 -*-
import matplotlib.pyplot as plt
import cv2
from skimage import data,draw,color,transform,feature,io,img_as_uint
'''
这个代码是使用transform.hough_ellipse寻找椭圆，速度很难，适合寻找已知形状的椭圆
accuracy=20, threshold=5, min_size=5, max_size=50，其中threshold是累加器累加的数量，变小会极大提升速度
'''

# 加载图片，转换成灰度图并检测边缘
image = data.coffee()
cv2.imwrite('coffee.jpg', image)
image_rgb = data.coffee()[0:220, 160:420]  # 裁剪原图像，不然速度非常慢
# image_rgb = io.imread('./coffee.jpg')
# image_rgb = image_rgb[0:220, 160:420]
image_rgb = io.imread('plant.png')
image_gray = color.rgb2gray(image_rgb)
print(image_rgb.shape)
print(image_gray.dtype)
'''
print(image_rgb.dtype.name) #uint8
print(image_gray.dtype.name) #float64 
print image_gray.shape #(220L, 260L)
#io.imshow(image_gray)
'''
edges = feature.canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
print(edges.dtype.name)  # bool
#print(edges)
# edges = image_gray > 0

'''
gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray_img, 100, 200, apertureSize=3)
'''

# 执行椭圆变换
result = transform.hough_ellipse(edges, accuracy=20, threshold=5, min_size=5, max_size=50)
result.sort(order='accumulator')  # 根据累加器排序
result = result[-5:-1]
print(result[-1])
# 估计椭圆参数
for best in result:
    print(best)
    # best = list(result[-1])  # 排完序后取最后一个
    best = list(best)
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    print(a, b)
    orientation = best[5]

    # 在原图上画出椭圆
    try:
        cy, cx = draw.ellipse_perimeter(yc, xc, a, b, orientation)
        image_rgb[cy, cx] = (0, 0, 255)  # 在原图中用蓝色表示检测出的椭圆
        print('haha')
    except Exception as e:
        print(e)
    # print image_rgb
    # io.imshow(image_rgb)


# 分别用白色表示canny边缘，用红色表示检测出的椭圆，进行对比
# print edges.shape #(220L, 260L)
edges = color.gray2rgb(edges)
print(edges.dtype.name)  # bool

edges = img_as_uint(edges)  # 转化类型
io.imshow(edges)
print(edges.shape)  # (220L, 260L, 3L)

edges[cy, cx] = (250, 0, 0)
# io.imshow(edges)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))

ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()
