import cv2
import numpy as np
import os
"""
通过调试发现，frame_20200121.jpg这张图，在kernel——size是（1，20），阈值是0.8的时候是可以检测出右侧线的，当然这个时候干扰线也多
kernel宽度：发现是1的时候，检出的更多，把宽度改成2，同样的阈值下，检出的线就会少一些，我们需要更多的线，然后再去处理，所以kenel应该选1
"""


class line:
    def __init__(self):
        self.confidence = 0
        self.ltrb = [0, 0, 0, 0]
        self.counts = 0


def get_hline(image, kernel_size=(1, 30), dx=5, dy=2, threshold=0.8, confirm_threshold=0.28, confirm_ratio=0.8):

    up_line_list = []
    down_line_list = []
    up_y_list = []
    down_y_list = []
    shape = image.shape
    area_size = kernel_size[0] * kernel_size[1]
    w = shape[1]
    h = shape[0]
    for y in range(2, h - kernel_size[0]-2, dy):
        for x in range(2, w - kernel_size[1] - 2, dx):
            dit_sum = np.sum(image[y:y+kernel_size[0], x:x+kernel_size[1]] == 255)
            if (dit_sum / area_size) > threshold and not (x > 0.7 * w and y < 0.3 * h):
                if y < 0.2 * h:  # 只考虑接近框边缘的线
                    if y in up_y_list:
                        ids = up_y_list.index(y)
                        # 距离超过10个像素我们就不再进行合并了，没有意义了
                        if x - up_line_list[ids][2] < 10:
                            x_min = min(x, up_line_list[ids][0])
                            x_max = max(x+kernel_size[1], up_line_list[ids][2])
                            up_line_list[ids] = [x_min, y, x_max, y]
                        else:
                            up_line_list.append([x, y, x + kernel_size[1], y])
                    else:
                        up_line_list.append([x, y, x+kernel_size[1], y])
                        up_y_list.append(y)
                elif y > 0.8 * h:
                    if y in down_y_list:
                        ids = down_y_list.index(y)
                        # 距离超过10个像素我们就不再进行合并了，没有意义了
                        if x - down_line_list[ids][2] < 10:
                            x_min = min(x, down_line_list[ids][0])
                            x_max = max(x+kernel_size[1], down_line_list[ids][2])
                            down_line_list[ids] = [x_min, y, x_max, y]
                        else:
                            down_line_list.append([x, y, x + kernel_size[1], y])
                    else:
                        down_line_list.append([x, y, x+kernel_size[1], y])
                        down_y_list.append(y)

    up_line_list_choose = []
    up_line_list_confirm = []
    down_line_list_choose = []
    down_line_list_confirm = []
    confirm_line_count = 0

    # todo 先进行最近的线的合并,暂时不用，评估效果决定是否添加

    # 对检出的线开始进行判断是不是确定线，如果是确定线，应该把之前的删掉，只加入确认线，并且继续添加
    for line in up_line_list:
        up_line_list_choose.append(line)
        length = line[2] - line[0]
        # 如果有很长的线而且255像素值总量大于一个阈值，我们就认为是确认线

        if length > confirm_threshold * shape[1] and (np.sum(image[line[1], line[0]:line[2]] == 255)) / (line[2] - line[0]) > confirm_ratio:
            confirm_line_count += 1
            up_line_list_confirm.append(line)  # 添加确认线
        num = 4
        if kernel_size[0] == 1:
            num = 2
        if confirm_line_count == num:  # 最多4根，说明这根线足够宽
            break

    confirm_line_count = 0
    down_line_list.reverse()
    for line in down_line_list:
        down_line_list_choose.append(line)
        length = line[2] - line[0]
        # 如果有很长的线而且255像素值总量大于一个阈值，我们就认为是确认线

        if length > confirm_threshold * shape[1] and (np.sum(image[line[1], line[0]:line[2]] == 255)) / (line[2] - line[0]) > confirm_ratio:
            confirm_line_count += 1
            down_line_list_confirm.append(line)  # 添加确认线
        num = 4
        if kernel_size[0] == 1:
            num = 2
        if confirm_line_count == num:  # 最多4根，说明这根线足够宽
            break

    return up_line_list_choose, down_line_list_choose, up_line_list_confirm, down_line_list_confirm


def get_vline(image, kernel_size=(30, 1), dx=2, dy=5, threshold=0.8, confirm_threshold=0.2, confirm_ratio=0.8):
    left_line_list = []
    right_line_list = []
    left_x_list = []
    right_x_list = []
    shape = image.shape
    area_size = kernel_size[0] * kernel_size[1]
    w = shape[1]
    h = shape[0]
    for x in range(2, w - kernel_size[1]-2, dx):
        for y in range(2, h - kernel_size[0]-2, dy):
            dit_sum = np.sum(image[y:y+kernel_size[0], x:x+kernel_size[1]]==255)
            if (dit_sum / area_size) > threshold and not (x > 0.8 * w and y < 0.2 * h):
                if x < 0.2 * w:
                    if x in left_x_list:
                        ids = left_x_list.index(x)
                        if y - left_line_list[ids][3] < 10:
                            y_min = min(y, left_line_list[ids][1])
                            y_max = max(y + kernel_size[0], left_line_list[ids][3])
                            left_line_list[ids] = [x, y_min, x, y_max]
                        else:
                            left_line_list.append([x, y, x, y + kernel_size[0]])
                    else:
                        left_line_list.append([x, y, x, y+kernel_size[0]])
                        left_x_list.append(x)
                elif x > 0.8 * w:
                    if x in right_x_list:
                        ids = right_x_list.index(x)
                        if y - right_line_list[ids][3] < 10:
                            y_min = min(y, right_line_list[ids][1])
                            y_max = max(y + kernel_size[0], right_line_list[ids][3])
                            right_line_list[ids] = [x, y_min, x, y_max]
                        else:
                            right_line_list.append([x, y, x, y + kernel_size[0]])
                    else:
                        right_line_list.append([x, y, x, y + kernel_size[0]])
                        right_x_list.append(x)

    left_line_list_choose = []
    left_line_list_confirm = []
    right_line_list_choose = []
    right_line_list_confirm = []
    confirm_line_count = 0

    for line in left_line_list:
        left_line_list_choose.append(line)
        length = line[3] - line[1]
        # print('debug', line, length, (np.sum(image[line[1]:line[3], line[0]] == 255)), confirm_threshold * shape[0])
        if length > confirm_threshold * shape[0] and (np.sum(image[line[1]:line[3], line[0]] == 255)) / (line[3] - line[1]) > confirm_ratio:
            confirm_line_count += 1
            left_line_list_confirm.append(line)  # 添加确认线
        num = 4
        if kernel_size[1] == 1:
            num = 2
        if confirm_line_count == num:  # 最多4根，说明这根线足够宽
            break

    confirm_line_count = 0
    right_line_list.reverse()
    for line in right_line_list:
        right_line_list_choose.append(line)
        length = line[3] - line[1]
        if length > confirm_threshold * shape[0] and (np.sum(image[line[1]:line[3], line[0]] == 255)) / (line[3] - line[1]) > confirm_ratio:
            confirm_line_count += 1
            right_line_list_confirm.append(line)  # 添加确认线
        num = 4
        if kernel_size[1] == 1:
            num = 2
        if confirm_line_count == num:  # 最多4根，说明这根线足够宽
            break

    return left_line_list_choose, right_line_list_choose, left_line_list_confirm, right_line_list_confirm


def get_card(line_lists, is_adjust=False):

    up_line_list, down_line_list, left_line_list, right_line_list = line_lists
    print('up', up_line_list)
    print('down', down_line_list)
    print('left', left_line_list)
    print('right', right_line_list)
    card_box = True
    card_ratio = 85.60 / 53.98  # 信用卡长宽比
    card = []
    index_list = []
    ratio_threshold = 0.08  # todo 最佳就是0.02就可以了，而且返回策略不是以比例来衡量了
    adjust_threshold = 0.3
    error_threshold = 1
    box_ratio = 0

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
                        # print('卡的比例是', box_ratio, [left[0], up[1], right[0], down[1]])
                        error = abs(card_ratio - ratio)
                        if error < error_threshold:
                            error_threshold = error
                            box_ratio = ratio
                            card_chose = [left[0], up[1], right[0], down[1]]
                            print('选的卡是', [left[0], up[1], right[0], down[1]], ratio, error_threshold)

        if error_threshold < ratio_threshold:
            card = card_chose
        elif is_adjust and error_threshold < adjust_threshold:
            card = card_chose


    return card


def get_threshold(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (3, 3), 0)
    gradX = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    ret, thresh = cv2.threshold(gradient, 160, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    mediu = cv2.medianBlur(hsv, 5)  # 中值滤波，去除椒盐噪声
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # gray = cv2.cvtColor(mediu, cv2.COLOR_RGB2GRAY)
    sobeledx = cv2.Sobel(hsv, cv2.CV_64F, 1, 0, ksize=5)
    sobeledy = cv2.Sobel(hsv, cv2.CV_64F, 0, 1, ksize=5)
    laplacian = cv2.Laplacian(hsv, cv2.CV_64F)
    xgrad = cv2.Sobel(img, cv2.CV_16SC1, 1, 0)
    ygrad = cv2.Sobel(img, cv2.CV_16SC1, 0, 1)
    edge_map = cv2.Canny(xgrad, ygrad, 10, 80)
    cv2.imshow('edge_map', edge_map)
    cv2.waitKey(0)

    xgrad = cv2.Sobel(hsv, cv2.CV_16SC1, 1, 0)
    ygrad = cv2.Sobel(hsv, cv2.CV_16SC1, 0, 1)
    edge_map_hsv = cv2.Canny(xgrad, ygrad, 50, 150)
    cv2.imshow('edge_map_hsv', edge_map_hsv)
    cv2.waitKey(0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.imshow('hsv', hsv)
    cv2.waitKey(0)

    grey = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (3, 3), 0)
    gradX = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(grey, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    ret, thresh_hsv = cv2.threshold(gradient, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow('hsv_thresh', thresh_hsv)
    cv2.waitKey(0)

    return thresh, edge_map, thresh_hsv, edge_map_hsv


def draw_line(img, line_list):

    for i, lines in enumerate(line_list):
        if i == 0:
            color = (0, 0, 255)
        elif i == 1:
            color = (255, 0, 0)
        elif i == 2:
            color = (255, 255, 0)
        else:
            color = (0, 255, 0)
        if len(lines) != 0:
            for i in range(len(lines)):
                cv2.line(img, (lines[i][0], lines[i][1]),
                         (lines[i][2], lines[i][3]), color, 2, cv2.LINE_AA)


def file_name(file_dir):
    file_list = []
    dirs_list = []
    file_name_list = []
    for root, dirs, files in os.walk(file_dir):
        print(root, dirs, files)
        dirs_list.append(dirs)
        for file in files:
            if '.DS_Store' in file:
                pass
            else:
                file_list.append(file)
                file_name_list.append(os.path.join(root, file))
    return file_list, file_name_list, dirs_list


'''
def get_confirm_line(line_lists,  confirm_line_lists):

    confirm_line_list = [[], [], [], []]

    for i in len(confirm_line_lists):

        list_one = confirm_line_lists[0][i]
        list_two = confirm_line_lists[1][i]
        list_thr = confirm_line_lists[2][i]
        list_fou = confirm_line_lists[3][i]

        line_list = list_one + list_two + list_thr + list_fou
        # 把所有的坐标拿出来加一下，然后做差，最后再循环一遍，把不合适的删掉
        for line in line_list:
            if i < 2:
            idx =

        confirm_line_list[i] = line_list

    up_line_list = [line_lists[0], ]
    down_line_list = [line_lists[0], ]
    left_line_list = [line_lists[0], ]
    right_line_list = [line_lists[0], ]
'''

def get_result(img):
    thresh_image, canny, hsv, hsv_canny = get_threshold(img)

    img_copy = img.copy()
    img_copy2 = img.copy()
    img_copy3 = img.copy()
    img_copy4 = img.copy()
    img_copy5 = img.copy()
    img_copy6 = img.copy()
    img_copy7 = img.copy()
    img_copy8 = img.copy()
    # 使用canny和大的kernel去获得确认的线, 这里设置小的kernel，因为逻辑里有并线操作
    # 也不用太大，因为要组装，最后判断组装的直线是不是我们要的直线
    # 因为kernel_size是1，所以不会出现复线的情况
    up_line_list, down_line_list, up_line_list_confirm, down_line_list_confirm = get_hline(canny, kernel_size=(1, 30), dx=3, dy=1, threshold=0.75, confirm_threshold=0.32, confirm_ratio=0.82)
    left_line_list, right_line_list, left_line_list_confirm, right_line_list_confirm = get_vline(canny, kernel_size=(25, 1), dx=1, dy=3, threshold=0.75, confirm_threshold=0.25, confirm_ratio=0.82)

    canny_line_lists = [up_line_list, down_line_list, left_line_list, right_line_list]
    draw_line(img_copy, canny_line_lists)
    # cv2.imwrite('line[1,20].jpg', frame)
    cv2.imshow('frame_small_line', img_copy)
    cv2.waitKey(0)

    canny_confirm_line_lists = [up_line_list_confirm, down_line_list_confirm, left_line_list_confirm, right_line_list_confirm]
    draw_line(img_copy2, canny_confirm_line_lists)
    # cv2.imwrite('line[1,20].jpg', frame)
    cv2.imshow('frame_small_confirm', img_copy2)
    cv2.waitKey(0)

    up_line_list_hsv, down_line_list_hsv, up_line_list_hsv_confirm, down_line_list_hsv_confirm = get_hline(hsv_canny, kernel_size=(1, 30), dx=3, dy=1, threshold=0.75, confirm_threshold=0.32, confirm_ratio=0.82)
    left_line_list_hsv, right_line_list_hsv, left_line_list_hsv_confirm, right_line_list_hsv_confirm = get_vline(hsv_canny, kernel_size=(25, 1), dx=1, dy=3, threshold=0.75, confirm_threshold=0.25, confirm_ratio=0.82)

    hsv_line_lists = [up_line_list_hsv, down_line_list_hsv, left_line_list_hsv, right_line_list_hsv]
    draw_line(img_copy4, hsv_line_lists)
    # cv2.imwrite('line[1,20].jpg', frame)
    cv2.imshow('frame_small_line_hsv_canny', img_copy4)
    cv2.waitKey(0)

    hsv_confirm_line_lists = [up_line_list_hsv_confirm, down_line_list_hsv_confirm, left_line_list_hsv_confirm, right_line_list_hsv_confirm]
    draw_line(img_copy3, hsv_confirm_line_lists)
    # cv2.imwrite('line[1,20].jpg', frame)
    cv2.imshow('frame_small_confirm_hsv_canny', img_copy3)
    cv2.waitKey(0)


    # 这里应该进行一下组装
    if len(up_line_list_confirm) == 0:
        up_line_list_confirm.extend(up_line_list_hsv_confirm)

    if len(down_line_list_confirm) == 0:
        down_line_list_confirm.extend(down_line_list_hsv_confirm)

    if len(left_line_list_confirm) == 0:
        left_line_list_confirm.extend(left_line_list_hsv_confirm)

    if len(right_line_list_confirm) == 0:
        right_line_list_confirm.extend(right_line_list_hsv_confirm)

    up_line_list.extend(up_line_list_hsv)
    down_line_list.extend(down_line_list_hsv)
    left_line_list.extend(left_line_list_hsv)
    right_line_list.extend(right_line_list_hsv)

    up_line_list_th, down_line_list_th, up_line_list_th_confirm, down_line_list_th_confirm = get_hline(thresh_image,
                                                                                           kernel_size=(2, 35),
                                                                                           dx=3, dy=1,
                                                                                           threshold=0.8, confirm_threshold=0.35, confirm_ratio=0.85)
    left_line_list_th, right_line_list_th, left_line_list_th_confirm, right_line_list_th_confirm = get_vline(thresh_image,
                                                                                                 kernel_size=(25, 2), dx=1, dy=3,
                                                                                                 threshold=0.8, confirm_threshold=0.25, confirm_ratio=0.85)

    th_line_lists = [up_line_list_th, down_line_list_th, left_line_list_th, right_line_list_th]
    draw_line(img_copy5, th_line_lists)
    # cv2.imwrite('line[1,20].jpg', frame)
    cv2.imshow('frame_small_line_th', img_copy5)
    cv2.waitKey(0)

    th_confirm_line_lists = [up_line_list_th_confirm, down_line_list_th_confirm, left_line_list_th_confirm, right_line_list_th_confirm]
    draw_line(img_copy6, th_confirm_line_lists)
    # cv2.imwrite('line[1,20].jpg', frame)
    cv2.imshow('frame_small_confirm_th', img_copy6)
    cv2.waitKey(0)

    up_line_list_hsvth, down_line_list_hsvth, up_line_list_hsvth_confirm, down_line_list_hsvth_confirm = get_hline(hsv,
                                                                                                                        kernel_size=(2, 35),
                                                                                                                        dx=3,
                                                                                                                        dy=1,
                                                                                                                        threshold=0.8,
                                                                                                                        confirm_threshold=0.35,
                                                                                                                        confirm_ratio=0.88)
    left_line_list_hsvth, right_line_list_hsvth, left_line_list_hsvth_confirm, right_line_list_hsvth_confirm = get_vline(hsv,
                                                                                                                         kernel_size=(25, 2),
                                                                                                                         dx=1,
                                                                                                                         dy=3,
                                                                                                                         threshold=0.8,
                                                                                                                         confirm_threshold=0.25,
                                                                                                                         confirm_ratio=0.88)

    hsvth_line_lists = [up_line_list_hsvth, down_line_list_hsvth, left_line_list_hsvth, right_line_list_hsvth]
    draw_line(img_copy7, hsvth_line_lists)
    # cv2.imwrite('line[1,20].jpg', frame)
    cv2.imshow('frame_small_line_hsvth', img_copy7)
    cv2.waitKey(0)

    hsvth_confirm_line_lists = [up_line_list_hsvth_confirm, down_line_list_hsvth_confirm, left_line_list_hsvth_confirm,
                                right_line_list_hsvth_confirm]
    draw_line(img_copy8, hsvth_confirm_line_lists)
    # cv2.imwrite('line[1,20].jpg', frame)
    cv2.imshow('frame_small_confirm_hsvth', img_copy8)
    cv2.waitKey(0)


    # 继续组装
    if len(up_line_list_confirm) == 0:
        up_line_list_confirm.extend(up_line_list_th_confirm)
        up_line_list_confirm.extend(up_line_list_hsvth_confirm)

    if len(down_line_list_confirm) == 0:
        down_line_list_confirm.extend(down_line_list_th_confirm)
        down_line_list_confirm.extend(down_line_list_hsvth_confirm)

    if len(left_line_list_confirm) == 0:
        left_line_list_confirm.extend(left_line_list_th_confirm)
        left_line_list_confirm.extend(left_line_list_hsvth_confirm)

    if len(right_line_list_confirm) == 0:
        right_line_list_confirm.extend(right_line_list_th_confirm)
        right_line_list_confirm.extend(right_line_list_hsvth_confirm)

    is_adjust = True

    if len(up_line_list_confirm) == 0:
        is_adjust = False
        if len(up_line_list) == 0:
            up_line_list.extend(up_line_list_th)
            up_line_list.extend(up_line_list_hsvth)
        up_line_list_confirm = up_line_list

    if len(down_line_list_confirm) == 0:
        is_adjust = False
        if len(down_line_list) == 0:
            down_line_list.extend(down_line_list_th)
            down_line_list.extend(down_line_list_hsvth)
        down_line_list_confirm = down_line_list

    if len(left_line_list_confirm) == 0:
        is_adjust = False
        if len(left_line_list) == 0:
            left_line_list.extend(left_line_list_th)
            left_line_list.extend(left_line_list_hsvth)
        left_line_list_confirm = left_line_list

    if len(right_line_list_confirm) == 0:
        is_adjust = False
        if len(right_line_list) == 0:
            right_line_list.extend(right_line_list_th)
            right_line_list.extend(right_line_list_hsvth)
        right_line_list_confirm = right_line_list

    line_lists = [up_line_list_confirm, down_line_list_confirm, left_line_list_confirm, right_line_list_confirm]
    # todo 调试完去除画线
    draw_line(img, line_lists)
    # cv2.imwrite('line[1,20].jpg', frame)
    cv2.imshow('frame_small', img)
    cv2.waitKey(0)

    card_rect = get_card(line_lists, is_adjust)
    print('检测到的卡片', card_rect, is_adjust)
    if len(card_rect) != 0:
        cv2.rectangle(img, (card_rect[0], card_rect[1]), (card_rect[2], card_rect[3]), (0, 0, 255), 2)
        cv2.imshow('frame_small', img)
        cv2.waitKey(0)

    line_lists = [canny_line_lists, hsv_line_lists, th_line_lists, hsvth_line_lists]
    confirm_line_lists = [canny_confirm_line_lists, hsv_confirm_line_lists, th_confirm_line_lists, hsvth_confirm_line_lists]

    """
    finally_line_lists = get_confirm_line(line_lists,  confirm_line_lists)
    card_rect = get_card(finally_line_lists)
    print('检测到的卡片', card_rect)
    if len(card_rect) != 0:
        cv2.rectangle(img, (card_rect[0], card_rect[1]), (card_rect[2], card_rect[3]), (0, 0, 255), 2)
        cv2.imshow('frame_small', img)
        cv2.waitKey(0)
    """

file_list, file_name_list, _ = file_name('data/')

for i, file in enumerate(file_name_list):
    frame = cv2.imread(file)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    print(i)
    # frame = frame[139:320, 240:515, :]
    if i in [4, 5, 7, 18, 22, 26, 27, 29]:
        frame_small = frame[47:190, 160:380, :]
    elif i in [20]:
        frame_small = frame[19:170, 126:346, :]
    elif i in [24]:
        frame_small = frame[28:162, 117:309, :]
    elif i in [25]:
        frame_small = frame[17:110, 77:217, :]
    elif i in [31]:
        frame_small = frame[1:185, 111:388, :]
    elif i in [35]:
        frame_small = frame[15:95, 63:185, :]
    elif i in [32]:
        frame_small = frame[22:106, 74:199, :]
    elif i in [3]:
        frame_small = frame[24:163, 127:350, :]
    elif frame.shape[1] > 270:
        frame_small = frame[30:166, 132:322, :]
    else:
        frame_small = frame[6:99, 65:209, :]
    cv2.imshow('frame_small', frame_small)
    cv2.waitKey(0)

    get_result(frame_small)

file_list, file_name_list, _ = file_name('data2/')

for i, file in enumerate(file_name_list):
    frame = cv2.imread(file)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    print(i)

    frame_small = frame[139:320, 240:515, :]
    if i in [4, 5, 0]:
        frame_small = frame[38:208, 139:413, :]
    elif i in [1]:
        frame_small = frame[79:263, 219:542, :]
    elif i in [2, 3]:
        frame_small = frame[38:111, 95:240, :]
    elif i in [6, 10]:
        frame_small = frame[30:117, 91:232, :]
    elif i in [7]:
        frame_small = frame[31:112, 91:217, :]
    elif i in [8, 9]:
        frame_small = frame[70:271, 225:540, :]
    elif i in [11, 14]:
        frame_small = frame[90:280, 242:570, :]
    elif i in [12]:
        frame_small = frame[70:256, 225:522, :]
    elif i in [13]:
        frame_small = frame[77:270, 230:580, :]
    elif i in [15]:
        frame_small = frame[43:190, 152:403, :]

    cv2.imshow('frame_small', frame_small)
    cv2.waitKey(0)

    get_result(frame_small)
