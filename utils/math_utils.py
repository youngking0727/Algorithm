import numpy as np
import math
import cv2
import imutils

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (yi2 - yi1) * (xi2 - xi1)  # 不相交的时候是负值

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area

    return iou


def isPointinPolygon(point, area_points):  # [[0,0],[1,1],[0,1],[0,0]] [1,0.8]顺序是左上，右上，右下，左下
    # 判断是否在外包矩形内，如果不在，直接返回false
    rangelist = []
    for i in area_points:
        rangelist.append(i)
    rangelist.append(rangelist[0]) # 把第一个点再加进去，形成封闭形状
    lnglist = []
    latlist = []
    for i in range(len(rangelist) - 1):
        lnglist.append(rangelist[i][0])
        latlist.append(rangelist[i][1])

    maxlng = max(lnglist)
    minlng = min(lnglist)
    maxlat = max(latlist)
    minlat = min(latlist)

    # 先判断是否在外接矩形里，不再直接就返回了
    if (point[0] > maxlng or point[0] < minlng or
            point[1] > maxlat or point[1] < minlat):
        return False

    # 在内部，射线交点是奇数，在内部交点是偶数
    count = 0
    point1 = rangelist[0]
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        # 点与多边形顶点重合
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            return False
        # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
        if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
            # 求线段与射线交点 再和lat比较
            point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1])
            # 点在多边形边上
            if (point12lng == point[0]):
                return False
            if (point12lng < point[0]):
                count += 1
        point1 = point2
    if count % 2 == 0:
        return False
    else:
        return True


def IOU(box1, box2):
    """
    :param box1:[x1,y1,x2,y2] 左上角的坐标与右下角的坐标
    :param box2:[x1,y1,x2,y2]
    :return: iou_ratio--交并比
    """
    width1 = abs(box1[2] - box1[0])
    height1 = abs(box1[1] - box1[3]) # 这里y1-y2是因为一般情况y1>y2，为了方便采用绝对值
    width2 = abs(box2[2] - box2[0])
    height2 = abs(box2[1] - box2[3])
    x_max = max(box1[0],box1[2],box2[0],box2[2])
    y_max = max(box1[1],box1[3],box2[1],box2[3])
    x_min = min(box1[0],box1[2],box2[0],box2[2])
    y_min = min(box1[1],box1[3],box2[1],box2[3])
    iou_width = x_min + width1 + width2 - x_max
    iou_height = y_min + height1 + height2 - y_max
    if iou_width <= 0 or iou_height <= 0:
        iou_ratio = 0
    else:
        iou_area = iou_width * iou_height  # 交集的面积
        box1_area = width1 * height1
        box2_area = width2 * height2
        iou_ratio = iou_area / (box1_area + box2_area - iou_area) # 并集的面积
    return iou_ratio


def get_board_area(ruler_area, frame):
    kernel = np.ones((3, 3), np.uint8)
    try:
        l, t, r, b = ruler_area
        ruler = frame[t:b, l:r, :]
        cv2.imshow('ruler', ruler)
        cv2.waitKey(0)
        hsv = cv2.cvtColor(ruler, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
        cv2.imshow('hsv', hsv)
        cv2.waitKey(0)
        lower_hsv = np.array([0, 0, 130])  # 提取颜色的低值
        high_hsv = np.array([80, 80, 255])  # 提取颜色的高值
        # bgr 图像的高低值
        #lower_hsv = np.array([0, 80, 46])  # 提取颜色的低值
        #high_hsv = np.array([200, 255, 255])  # 提取颜色的高值
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
        mask = cv2.inRange(ruler, lowerb=lower_hsv, upperb=high_hsv)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        mask = cv2.dilate(mask, kernel, iterations=1)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        target_c = []
        area = 10000
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if abs(w*h-70) < area:
                area = abs(w*h-70)

                target_c = [x, y, w, h]
        x, y, w, h = target_c
        print('board box', x, y, w, h)
        cv2.rectangle(ruler, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('ruler', ruler)
        cv2.waitKey(0)
    except Exception as e:
        print(e)
    pass
