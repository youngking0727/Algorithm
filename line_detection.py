import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

'''根据极角极径参数在原图像中画线'''


def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    for line in lines:
        for x1, y1, x2, y2 in line:
            # 在图像中设置俩条水平线，只有当直线于其相交，才绘制该直线，用于筛选直线
            top_horizon_line = ([0, img.shape[0] * 0.6], [img.shape[1], img.shape[0] * 0.7])
            bottom_horizon_line = ([0, img.shape[0]], [img.shape[1], img.shape[0]])
            line_intersection_top = line_intersection(top_horizon_line, ([x1, y1], [x2, y2]))
            line_intersection_bottom = line_intersection(bottom_horizon_line, ([x1, y1], [x2, y2]))
            #if line_intersection_top == None or line_intersection_bottom == None:
            #    return
            # 绘制直线
            cv2.line(img, line_intersection_top, line_intersection_bottom, color, thickness)


'''判断俩条线段是否相交'''


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)

    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)


image = cv2.imread("1.jpg")  # 读取源图像
showImg = True
# image = np.array(image)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 转为灰度图
print(grayscale_image)
kernel_size = 5
# 高斯平滑，大小为5的高斯核
gaussian_blur_image = cv2.GaussianBlur(grayscale_image, (kernel_size, kernel_size), 0)
cv2.imwrite("gauss.jpg", gaussian_blur_image)
# 设置阈值，进行canny边缘提取
canny_low_threshold = 100
canny_high_threshold = 150
edge_image = cv2.Canny(gaussian_blur_image, canny_low_threshold, canny_high_threshold)
# ROI可视区域选择，本程序选取左测公路区域
image_shape = edge_image.shape
cv2.imwrite("edge.jpg", edge_image)
print(image_shape)
x_offset = 200
y_offset = 90
v1 = (0, image_shape[0] - y_offset / 2)
v2 = (int(image_shape[1] / 4 + x_offset), int(image_shape[0] / 2 + y_offset))
v3 = (int(image_shape[1] / 2 - x_offset), int(image_shape[0] / 2 + y_offset))
v4 = (image_shape[1] / 2 + x_offset, image_shape[0] - y_offset / 2)
print(v1,v2,v3,v4)
vert = np.array([[v1, v2, v3, v4]], dtype=np.int32)
print(vert)
vert = np.array([[0,120],[0,307], [500,307],[500,120]], dtype=np.int32)
print(vert)
mask = np.zeros_like(edge_image)
print(mask.shape)
cv2.imwrite("raw_mask.jpg",mask)
print("sadadads", len(edge_image.shape))
if len(edge_image.shape) > 2:
    channel_count = edge_image.shape[2]
    ignore_mask_color = (255,) * channel_count
    print(f"ignore_mask_color: {ignore_mask_color}")
else:
    ignore_mask_color = 255
    print(f"ignore_mask_color >>>> : {ignore_mask_color}")
# ROI可视区填充，在用mask与灰度图进行与运算，即在灰度图中得可视区
cv2.fillPoly(mask, [vert], ignore_mask_color)
cv2.imwrite("mask.jpg", mask)
masked_edge_image = cv2.bitwise_and(edge_image, mask)
print("fafasfafafs")
# 显示可视区的边缘提取二值图像
cv2.imwrite("masked_edge.jpg", masked_edge_image)
s = cv2.getStructuringElement(cv2.M)

print(np.array([[v1, v2, v3, v4]], dtype=np.int32))
# 霍夫线变换
rho = 2  # 设置极径分辨率
theta = (np.pi) / 180  # 设置极角分辨率
threshold = 10  # 设置检测一条直线所需最少的交点
min_line_len = 30  # 设置线段最小长度
max_line_gap = 200  # 设置线段最近俩点的距离
lines = cv2.HoughLinesP(masked_edge_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                        maxLineGap=max_line_gap)
for i,line in enumerate(lines):
    print(i)
hough_line_image = np.zeros((masked_edge_image.shape[0], masked_edge_image.shape[1], 3), dtype=np.uint8)

# 绘制检测到的直线
draw_lines(hough_line_image, lines)
# 将直线与原图像合成为一幅图像
sync_image = cv2.addWeighted(image, 0.8, hough_line_image, 1, 0)
# 显示图像
if showImg:
    plt.imshow(sync_image)
    plt.show()
    cv2.imshow("lane", sync_image)
    cv2.waitKey(0)
