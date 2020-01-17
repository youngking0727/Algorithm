#! ./venv/bin/python3.6
import cv2
import numpy as np
import matplotlib
import os

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import time

class fixed_list(list):
    def __init__(self, Length):
        # 固定长度的列表
        self.fa = super(fixed_list, self).__init__()
        self.Length_Constant = Length
        self.Counter = 0

    def increase(self, object):
        if not self.Counter < self.Length_Constant:
            self.pop(0)
            self.append(object)
        else:
            self.append(object)
            self.Counter += 1


class Line():
    def __init__(self, pos):
        # 找到的那一根
        self.line = None
        self.k = None  # 斜率
        self.angles = fixed_list(6)  # 与x轴夹角
        self.b = None
        self.__pos = pos

    def two_points_ensure_line_equation(self, y=None, x=None):
        two_points = self.line
        assert (two_points[2] - two_points[0]) != 0 and (two_points[3] - two_points[1]) != 0

        if x or y:
            if x:
                return self.k * x + self.b
            if y:
                # 与另一条同高
                if self.__pos == 'l':
                    self.line[0] = (y - self.b) / self.k
                elif self.__pos == 'r':
                    self.line[-2] = (y - self.b) / self.k

        else:
            k = (two_points[3] - two_points[1]) / (two_points[2] - two_points[0])
            b = two_points[1] - k * two_points[0]
            self.k, self.b = k, b

    def angle_calcu(self):
        if self.k:
            self.angles.increase(math.atan(self.k))

    def set_Line(self, line):
        self.line = line
        self.k = None  # 斜率
        self.b = None
        self.two_points_ensure_line_equation()
        if self.__pos == 'l':
            self.to_y = line[1]
        elif self.__pos == 'r':
            self.to_y = line[-1]


class Direction():
    def __init__(self, Length, range=0.):
        self.L = Length
        self.angles_ratio_lr = fixed_list(Length)
        self.__range = range  # 转向灵敏度，越小越容易转，<0.1

    def inference_direction(self, l_k, r_k):
        # 1左，-1右，0中间
        if l_k and r_k:
            ratio = math.atan(r_k) / abs(math.atan(l_k))  # 和x轴的夹角，右转<==>变小
            self.angles_ratio_lr.increase(ratio)
        else:
            if len(self.angles_ratio_lr) > 0:
                self.angles_ratio_lr.increase(self.angles_ratio_lr[-1])
        #
        rot = 0
        if len(self.angles_ratio_lr) >= 2:
            if self.angles_ratio_lr[-1] > self.angles_ratio_lr[-2] + self.__range:
                rot = 1
            elif self.angles_ratio_lr[-1] + self.__range < self.angles_ratio_lr[-2]:
                rot = -1
        return rot


class GaussianBlur():
    def __init__(self):
        self.kernel_size = 5
        self.sigmaX = 0

    def blur(self, img):
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigmaX)


class Canny():
    def __init__(self):
        self.low_threshold = 50
        self.high_threshold = 150

    def canny(self, img):
        return cv2.Canny(img, self.low_threshold, self.high_threshold)


class ROI_Based_Edge_Filtering():
    def __init__(self, edges):
        # 梯形ROI
        edges = edges[:, :, 0]
        self.mask = np.zeros_like(edges)
        self.ignore_mask_color = 255
        self.imshape = edges.shape
        # print(self.imshape)
        self.vertices = np.array([[(150, 200), (280, 75), (285, 75), (self.imshape[1], 200)]], dtype=np.int32)
        cv2.fillPoly(self.mask, self.vertices, self.ignore_mask_color)

    def bitwise_and(self, img):
        return cv2.bitwise_and(img, self.mask)


class Image_Morphology():
    def __init__(self):
        # 腐蚀 膨胀
        self.dkernel = np.ones((3, 3), np.uint8)
        self.ekernel = np.ones((2, 2), np.uint8)

    def open_close(self, img):
        img = cv2.dilate(img, self.dkernel, iterations=2)
        img = cv2.erode(img, self.ekernel, iterations=3)
        return img


class HoughLinesP():
    def __init__(self):
        # 霍夫
        self.rho = 1  # distance resolution in pixels of the Hough grid
        self.theta = np.pi / 180  # angular resolution in radians of the Hough grid
        self.threshold = 30  # minimum number of votes (intersections in Hough grid cell)
        self.min_line_length = 30  # minimum number of pixels making up a line
        self.max_line_gap = 30  # maximum gap in pixels between connectable line segments
        # line_image = np.copy(image)*0 # creating a blank to draw lines on

    def houghlinesP(self, img):
        # return (n,4)
        try:
            return cv2.HoughLinesP(img, self.rho, self.theta, self.threshold, np.array([]), self.min_line_length,
                                   self.max_line_gap)[:, 0, :]
        except TypeError:
            return None


def get_suitable_line(lines):
    # 通过斜率和长度筛选出合适的线段，并且补齐
    # 全部线的斜率
    deltax = (lines[:, 0] - lines[:, 2])
    deltay = (lines[:, 1] - lines[:, 3])
    k = deltay / (deltax + 0.001)
    # 将线分成左右两部分，左侧斜率小于0，右侧大于0
    l_lines_inds = np.where(k < 0)[0]
    l_lines = lines[l_lines_inds]
    r_lines_inds = np.where(k > 0)[0]
    r_lines = lines[r_lines_inds]
    # print('right', r_lines_inds,'left',l_lines_inds)

    if len(r_lines_inds) and len(l_lines_inds):
        # 一把算出线段长度
        delta = np.concatenate(((lines[:, 0] - lines[:, 2]).reshape((1, -1)),
                                (lines[:, 1] - lines[:, 3]).reshape((1, -1))))
        lines_length = np.linalg.norm(delta, axis=0)
        ## 左 右 分开
        l_lines_length = lines_length[l_lines_inds]
        r_lines_length = lines_length[r_lines_inds]
        l_lines_length_max_indice = np.argmax(l_lines_length)
        r_lines_length_max_indice = np.argmax(r_lines_length)
        l_line = l_lines[l_lines_length_max_indice]
        r_line = r_lines[r_lines_length_max_indice]

        return l_line, r_line
    else:
        return [None], [None]
    pass


class DrawDraw():
    def __init__(self, img_row, scope0, turtle='turtle.png'):
        # 跟绘图有关的
        self.scope = self.__define_scope(img_row, scope0)  # y0 y1 x0 x1 (650, 960, 0, 536)
        self.turtle = cv2.imread(turtle)  # 读入小乌龟
        self.__rotate_turtle()

        self.position4turtle = ((int(0.8 * self.scope[1] + 0.2 * self.scope[0]) - self.turtle.shape[0] // 2,
                                 (self.scope[2] + self.scope[3]) // 2 - self.turtle.shape[1] // 2),
                                (int(0.8 * self.scope[1] + 0.2 * self.scope[0]) + self.turtle.shape[0] // 2,
                                 (self.scope[2] + self.scope[3]) // 2 + self.turtle.shape[1] // 2))
        # print(self.position4turtle)

    def set_turtle(self, img_row, direction=0):
        # 将原图换成透明小海龟
        small_img = img_row[self.position4turtle[0][0]:self.position4turtle[1][0],
                    self.position4turtle[0][1]:self.position4turtle[1][1]]

        # print(img_row.shape,small_img.shape,self.turtle.shape)
        # assert small_img.shape==self.turtle.shape
        turtle = self.turtle
        if direction == 1:
            turtle = self.turtle_l
        elif direction == -1:
            turtle = self.turtle_r
        # print(small_img.shape,turtle.shape)
        small_img = cv2.addWeighted(small_img, 1, turtle, 0.9, 0)
        img_row[self.position4turtle[0][0]:self.position4turtle[1][0],
        self.position4turtle[0][1]:self.position4turtle[1][1]] \
            = small_img
        pass

    def __rotate_turtle(self):
        shape = self.turtle.shape
        rotate_matrix = cv2.getRotationMatrix2D((shape[1] // 2, shape[0] // 2), 45, 0.8)
        self.turtle_l = cv2.warpAffine(self.turtle, rotate_matrix, shape[1::-1])
        rotate_matrix = cv2.getRotationMatrix2D((shape[1] // 2, shape[0] // 2), -45, 0.8)
        self.turtle_r = cv2.warpAffine(self.turtle, rotate_matrix, shape[1::-1])
        pass

    def xxxx(self, img_row, l_line, r_line):
        # 画个交叉线
        assert len(l_line) == 4 and len(r_line) == 4
        cv2.line(img_row, (self.scope[2] + l_line[0], self.scope[0] + l_line[1]),
                 (self.scope[2] + l_line[2], self.scope[0] + l_line[3]), (200, 0, 200), 2)
        cv2.line(img_row, (self.scope[2] + r_line[0], self.scope[0] + r_line[1]),
                 (self.scope[2] + r_line[2], self.scope[0] + r_line[3]), (200, 0, 200), 2)

    def __define_scope(self, img_row, scope):
        # print(type(img_row))

        shape = img_row.shape
        # print(type(img_row))
        shape = [0, shape[0], 0, shape[1]]
        for i, sc in enumerate(scope):
            if sc == None:
                scope[i] = shape[i]
        return tuple(scope)


class MySetting():
    def __init__(self, img_row):
        scope0 = [650, None, None, None]
        self.small_dd = DrawDraw(img_row, scope0)  # 画图的
        self.scope = self.small_dd.scope
        imginit = img_row[self.scope[0]:self.scope[1],
                  self.scope[2]:self.scope[3]]
        self.gb = GaussianBlur().blur
        self.canny = Canny().canny
        self.roi = ROI_Based_Edge_Filtering(imginit).bitwise_and
        self.oc = Image_Morphology().open_close
        self.houghlinesp = HoughLinesP().houghlinesP


def main():
    # 第一次ROI范围

    cap = cv2.VideoCapture(0)

    _, img = cap.read()
    cv2.imwrite("2.jpg", img)
    mvname = 'laotie.mp4'
    if not os.path.isfile(mvname):
        cvwriter = cv2.VideoWriter(mvname, cv2.VideoWriter_fourcc(*'MP4V'), 30, img.shape[-2::-1])
    #cv2.namedWindow('', cv2.WINDOW_KEEPRATIO)
    #cv2.resizeWindow('', 480, 640)
    #cv2.moveWindow('', 0, 0)
    cv = MySetting(img)
    left, right = Line('l'), Line('r')
    direc = Direction(6, range=0.06)
    rot = 0
    while True:
        # t0 = time.time()
        _, img = cap.read()
        cv2.imwrite("3.jpg", img)
        print(img)
        cv2.imshow('1', img)
        #img = cv2.imread("2.jpg")
        print(img.shape)
        img = img[cv.scope[0]:cv.scope[1], cv.scope[2]:cv.scope[3]]
        print('print(img.shape)', img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('gray.jpg', gray)
        blur_gray = cv.gb(gray)
        edges = cv.canny(blur_gray)
        print('edge', edges.shape)
        masked_edges = cv.roi(edges)
        ocimg = cv.oc(masked_edges)
        #cv2.imshow('1', img)
        # 霍夫
        lines = cv.houghlinesp(ocimg)
        print('lines', lines)
        if lines is None:
            continue
        # 整理图像
        # erosion_rgb = cv2.cvtColor(ocimg, cv2.COLOR_GRAY2RGB)
        # 斜率确定直线
        l_line, r_line = get_suitable_line(lines)
        # print('left:', l_line, 'right:', r_line)
        if np.any(l_line) and np.any(r_line):
            left.set_Line(l_line), right.set_Line(r_line)
            # print('left:', l_line, 'right:', r_line)
            # 延长两边y点的高度，到一样
            left.two_points_ensure_line_equation(y=right.to_y) if right.to_y > left.to_y else \
                right.two_points_ensure_line_equation(y=left.to_y)
            # 计算方向
            # 画图
            #
            cv.small_dd.xxxx(img_row, left.line, right.line)
            rot = direc.inference_direction(left.k, right.k)
        cv.small_dd.set_turtle(img_row, rot)
        cv2.imshow('', img_row)
        flag = cv2.waitKey(20)
        if flag == 27:
            break
        elif flag == 32:
            while flag != 32:
                flag = cv2.waitKey(0)
        try:
            cvwriter.write(img_row)
        except UnboundLocalError:
            continue

        # print(1/(time.time()-t0))#32FPS，36ms
    pass


if __name__ == '__main__':
    main()
