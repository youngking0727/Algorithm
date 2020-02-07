import cv2
import numpy as np
"""


"""


def get_hline(image, kernel_size=(2, 20), dx=5, dy=2):
    up_line_list = []
    down_line_list = []
    up_y_list = []
    down_y_list = []
    shape = image.shape
    area_size = kernel_size[0] * kernel_size[1]
    w = shape[1]
    h = shape[0]
    for y in range(0, w - kernel_size[0], dy):
        for x in range(0, h - kernel_size[1], dx):
            dit_sum = np.sum(image[y:y+kernel_size[0], x:x+kernel_size[1]] == 255)
            if (dit_sum / area_size) > 0.8 and not (x > 0.8 * w and y < 0.2 * h):  # 手指部分不考虑
                if y < 0.2 * h:  # 只考虑接近框边缘的线
                    if y in up_y_list:
                        ids = up_y_list.index(y)
                        x = min(x, up_line_list[ids][0])
                        x_max = max(x+kernel_size[1], up_line_list[ids][2])
                        up_line_list[ids] = [x, y, x_max, y]
                    else:
                        up_line_list.append([x, y, x+kernel_size[1], y])
                        up_y_list.append(y)
                elif y > 0.8 * h:
                    if y in down_y_list:
                        ids = down_y_list.index(y)
                        x = min(x, down_line_list[ids][0])
                        x_max = max(x+kernel_size[1], down_line_list[ids][2])
                        down_line_list[ids] = [x, y, x_max, y]
                    else:
                        down_line_list.append([x, y, x+kernel_size[1], y])
                        down_y_list.append(y)

    return up_line_list[:2], down_line_list[:2]


def get_vline(image, kernel_size=(20, 2), dx=2, dy=5):
    left_line_list = []
    right_line_list = []
    left_x_list = []
    right_x_list = []
    shape = image.shape
    area_size = kernel_size[0] * kernel_size[1]
    w = shape[1]
    h = shape[0]
    for x in range(0, w - kernel_size[1], dx):
        for y in range(0, h - kernel_size[0], dy):
            dit_sum = np.sum(image[y:y+kernel_size[0], x:x+kernel_size[1]]==255)
            if (dit_sum / area_size) > 0.8 and not (x > 0.8 * w and y < 0.2 * h):
                if x < 0.2 * w:
                    if x in left_x_list:
                        ids = left_x_list.index(x)
                        y = min(y, left_line_list[ids][1])
                        y_max = max(y + kernel_size[0], left_line_list[ids][3])
                        left_line_list[ids] = [x, y, x, y_max]
                    else:
                        left_line_list.append([x, y, x, y+kernel_size[0]])
                        left_x_list.append(x)
                elif x > 0.8 * w:
                    if x in right_x_list:
                        ids = right_x_list.index(x)
                        y = min(y, right_line_list[ids][1])
                        y_max = max(y + kernel_size[0], right_line_list[ids][3])
                        right_line_list[ids] = [x, y, x, y_max]
                    else:
                        right_line_list.append([x, y, x, y + kernel_size[0]])
                        right_x_list.append(x)

    return left_line_list, right_line_list


def get_card(line_lists):
    up_line_list, down_line_list, left_line_list, right_line_list = line_lists
    card_box = True
    card_ratio = 1.58 # 信用卡长宽比
    card = []
    index_list = []
    ratio_threshold = 0.1
    error_threshold = 1

    for i, lines in enumerate(line_lists):
        if len(lines) == 0:
            card_box = False
        else:
            index_list.append(i)

    if card_box:
        card_chose = []
        for a, up in enumerate(up_line_list):
            for b, down in enumerate(down_line_list):
                for c, right in enumerate(right_line_list):
                    for d, left in enumerate(left_line_list):
                        ratio = (right[0] - left[0]) / (down[1] - up[1])
                        error = abs(card_ratio - ratio)
                        if error < error_threshold:
                            error_threshold = error
                            card_chose = [left[0], up[1], right[0], down[1]]
        if error_threshold < ratio_threshold:
            card = card_chose

    return card


def get_threshold(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (3, 3), 0)
    gradX = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    ret, thresh = cv2.threshold(gradient, 150, 255, cv2.THRESH_BINARY)
    return thresh





cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # grey = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (3, 3), 0)
    canny = cv2.Canny(grey, 80, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    opening = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel)

    # 用Sobel算子计算x，y方向上的梯度，之后在x方向上减去y方向上的梯度，
    # 通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域。
    # https://blog.csdn.net/liqiancao/article/details/55670749 抠蜜蜂
    # 梯度抠信用卡 https://blog.csdn.net/g11d111/article/details/78094687
    gradX = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    ret, thresh1 = cv2.threshold(gradient, 200, 255, cv2.THRESH_BINARY)
    # thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))

    gradient_o = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)
    # gradient_o = cv2.morphologyEx(gradient_o, cv2.MORPH_CLOSE, kernel2)

    # 用自己写的方法提取水平线
    lines = get_vline(thresh1)
    if len(lines) != 0:
        for i in range(len(lines)):
            cv2.line(img, (lines[i][0], lines[i][1]), (lines[i][2], lines[i][3]), (0, 0, 255), 3, cv2.LINE_AA)

    # 提取水平线
    hline = cv2.getStructuringElement(cv2.MORPH_RECT, ((int(img.shape[1] / 16)), 1), (-1, -1))
    dst = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, hline)
    dst = cv2.bitwise_not(dst)

    # 提取垂直线
    vline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, (int(img.shape[1] / 16))), (-1, -1))
    dst = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, vline)
    dst = cv2.bitwise_not(dst)

    # 在这里做一些腐蚀膨胀操作提
    dilated = cv2.dilate(gradient, kernel, iterations=1)

    result = cv2.bitwise_and(gradient, grey)

    blurred = cv2.GaussianBlur(grey, (3, 3), 0)
    # cnts = cv2.findContours(opening.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # cv2.drawContours(opening, [cnts], -1, 255, -1)
    # _, thresh = cv2.threshold(gradient, 90, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(gradient, 50, 200)

    img_2, contours, hei = cv2.findContours(opening, mode=cv2.RETR_EXTERNAL,
                                            method=cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
    for contour in contours:
        if 100 < cv2.contourArea(contour) < 40000:
            x, y, w, h = cv2.boundingRect(contour)  # 找方框
            cv2.rectangle(grey, (x, y), (x + w, y + h), (255, 255, 255), 3)

    '''
    minLineLength = 10
    lines = cv2.HoughLinesP(image=grey, rho=1, theta=np.pi / 180, threshold=1000, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=10)
    if lines is None:
        continue
    a, b, c = lines.shape
    for i in range(a):
        cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    '''
    cv2.imshow('1', img)
    if cv2.waitKey(50) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()




