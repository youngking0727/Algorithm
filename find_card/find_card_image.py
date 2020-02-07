import cv2
import numpy as np
import os
"""
通过调试发现，frame_20200121.jpg这张图，在kernel——size是（1，20），阈值是0.8的时候是可以检测出右侧线的，当然这个时候干扰线也多
kernel宽度：发现是1的时候，检出的更多，把宽度改成2，同样的阈值下，检出的线就会少一些，我们需要更多的线，然后再去处理，所以kenel应该选1
"""


def get_hline(image, kernel_size=(1, 30), dx=5, dy=2, threshold=0.8):

    up_line_list = []
    down_line_list = []
    up_y_list = []
    down_y_list = []
    shape = image.shape
    area_size = kernel_size[0] * kernel_size[1]
    w = shape[1]
    h = shape[0]
    for y in range(0, h - kernel_size[0], dy):
        for x in range(0, w - kernel_size[1], dx):
            dit_sum = np.sum(image[y:y+kernel_size[0], x:x+kernel_size[1]] == 255)
            if (dit_sum / area_size) > threshold and not (x > 0.7 * w and y < 0.3 * h):
                if y < 0.2 * h:  # 只考虑接近框边缘的线
                    if y in up_y_list:
                        ids = up_y_list.index(y)
                        print(y, ids, x, up_line_list[ids][0])
                        x_min = min(x, up_line_list[ids][0])
                        x_max = max(x+kernel_size[1], up_line_list[ids][2])
                        print(up_line_list[ids])
                        up_line_list[ids] = [x_min, y, x_max, y]
                        print(up_line_list[ids])
                    else:
                        up_line_list.append([x, y, x+kernel_size[1], y])
                        up_y_list.append(y)
                elif y > 0.8 * h:
                    if y in down_y_list:
                        ids = down_y_list.index(y)
                        x_min = min(x, down_line_list[ids][0])
                        x_max = max(x+kernel_size[1], down_line_list[ids][2])
                        down_line_list[ids] = [x_min, y, x_max, y]
                    else:
                        down_line_list.append([x, y, x+kernel_size[1], y])
                        down_y_list.append(y)

    up_line_list_choose = []
    down_line_list_choose = []
    for line in up_line_list:
        up_line_list_choose.append(line)
        length = line[2] - line[0]
        # 如果有很长的线就直接break了，或者达到3根线
        '''
        if length > 0.35 * shape[1]:
            break
        elif len(up_line_list_choose) == 3:
            break
        '''
    down_line_list.reverse()
    for line in down_line_list:
        down_line_list_choose.append(line)
        length = line[2] - line[0]
        '''
        if length > 0.35 * shape[1]:
            break
        elif len(up_line_list_choose) == 3:
            break
        '''

    return up_line_list_choose, down_line_list_choose


def get_vline(image, kernel_size=(30, 1), dx=2, dy=5, threshold=0.8):
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
            if (dit_sum / area_size) > threshold and not (x > 0.8 * w and y < 0.2 * h):
                if x < 0.2 * w:
                    if x in left_x_list:
                        ids = left_x_list.index(x)
                        y_min = min(y, left_line_list[ids][1])
                        y_max = max(y + kernel_size[0], left_line_list[ids][3])
                        left_line_list[ids] = [x, y_min, x, y_max]
                    else:
                        left_line_list.append([x, y, x, y+kernel_size[0]])
                        left_x_list.append(x)
                elif x > 0.8 * w:
                    if x in right_x_list:
                        ids = right_x_list.index(x)
                        y_min = min(y, right_line_list[ids][1])
                        y_max = max(y + kernel_size[0], right_line_list[ids][3])
                        right_line_list[ids] = [x, y_min, x, y_max]
                    else:
                        right_line_list.append([x, y, x, y + kernel_size[0]])
                        right_x_list.append(x)

    left_line_list_choose = []
    right_line_list_choose = []
    for line in left_line_list:
        left_line_list_choose.append(line)
        length = line[2] - line[0]
        '''
        if length > 0.35 * shape[1]:
            break
        elif len(left_line_list_choose) == 3:
            break
        '''
    right_line_list.reverse()
    for line in right_line_list:
        right_line_list_choose.append(line)
        length = line[2] - line[0]
        '''
        if length > 0.35 * shape[1]:
            break
        elif len(right_line_list_choose) == 3:
            break
        '''
    return left_line_list_choose, right_line_list_choose


def get_card(line_lists):

    up_line_list, down_line_list, left_line_list, right_line_list = line_lists
    card_box = True
    card_ratio = 1.58  # 信用卡长宽比
    card = []
    index_list = []
    ratio_threshold = 0.03  # todo 最佳就是0.02就可以了，而且返回策略不是以比例来衡量了
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
                        print('卡的比例是', box_ratio)
                        error = abs(card_ratio - ratio)
                        if error < error_threshold:
                            error_threshold = error
                            box_ratio = ratio
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
    ret, thresh_hsv = cv2.threshold(gradient, 160, 255, cv2.THRESH_BINARY)
    cv2.imshow('hsv_thresh', thresh_hsv)
    cv2.waitKey(0)

    return thresh, edge_map, thresh_hsv, edge_map_hsv


def draw_line(img, line_list):

    for lines in line_list:
        if len(lines) != 0:
            for i in range(len(lines)):
                cv2.line(img, (lines[i][0], lines[i][1]),
                         (lines[i][2], lines[i][3]), (0, 0, 255), 1, cv2.LINE_AA)


def file_name(file_dir):
    file_list = []
    dirs_list = []
    file_name_list = []
    for root, dirs, files in os.walk(file_dir):
        print(root, dirs, files)
        dirs_list.append(dirs)
        for file in files:
            file_list.append(file)
            file_name_list.append(os.path.join(root, file))
    return file_list, file_name_list, dirs_list


def get_result(img):
    thresh_image, canny, hsv, hsv_canny = get_threshold(img)

    up_line_list, down_line_list = get_hline(thresh_image, kernel_size=(2, 35), dx=3, dy=1, threshold=0.7)
    left_line_list, right_line_list = get_vline(thresh_image, kernel_size=(35, 2), dx=1, dy=3, threshold=0.7)

    line_lists = [up_line_list, down_line_list, left_line_list, right_line_list]
    # todo 调试完去除画线
    img_copy = img.copy()
    img_copy2 = img.copy()
    img_copy3 = img.copy()
    draw_line(img, line_lists)
    # cv2.imwrite('line[1,20].jpg', frame)
    cv2.imshow('frame_small', img)
    cv2.waitKey(0)

    card_rect = get_card(line_lists)
    print('检测到的卡片', card_rect)
    if len(card_rect) != 0:
        cv2.rectangle(img, (card_rect[0], card_rect[1]), (card_rect[2], card_rect[3]), (0, 0, 255), 2)
        cv2.imshow('frame_small', img)
        cv2.waitKey(0)

    up_line_list, down_line_list = get_hline(canny, kernel_size=(1, 20), dx=3, dy=1, threshold=0.65)
    left_line_list, right_line_list = get_vline(canny, kernel_size=(20, 1), dx=1, dy=3, threshold=0.65)

    line_lists = [up_line_list, down_line_list, left_line_list, right_line_list]
    draw_line(img_copy, line_lists)
    cv2.imshow('frame_small_copy', img_copy)
    cv2.waitKey(0)

    card_rect = get_card(line_lists)
    print('检测到的卡片', card_rect)
    if len(card_rect) != 0:
        cv2.rectangle(img_copy, (card_rect[0], card_rect[1]), (card_rect[2], card_rect[3]), (0, 0, 255), 2)
        cv2.imshow('frame_small_copy', img_copy)
        cv2.waitKey(0)

    up_line_list, down_line_list = get_hline(hsv, kernel_size=(2, 35), dx=3, dy=1, threshold=0.7)
    left_line_list, right_line_list = get_vline(hsv, kernel_size=(35, 2), dx=1, dy=3, threshold=0.7)

    line_lists = [up_line_list, down_line_list, left_line_list, right_line_list]
    draw_line(img_copy2, line_lists)
    cv2.imshow('frame_small_copy_hsv', img_copy2)
    cv2.waitKey(0)

    card_rect = get_card(line_lists)
    print('检测到的卡片', card_rect)
    if len(card_rect) != 0:
        cv2.rectangle(img_copy2, (card_rect[0], card_rect[1]), (card_rect[2], card_rect[3]), (0, 0, 255), 2)
        cv2.imshow('frame_small_copy_hsv', img_copy2)
        cv2.waitKey(0)

    up_line_list, down_line_list = get_hline(hsv_canny, kernel_size=(1, 20), dx=3, dy=1, threshold=0.65)
    left_line_list, right_line_list = get_vline(hsv_canny, kernel_size=(20, 1), dx=1, dy=3, threshold=0.65)

    line_lists = [up_line_list, down_line_list, left_line_list, right_line_list]
    draw_line(img_copy3, line_lists)
    cv2.imshow('frame_small_canny_hsv', img_copy3)
    cv2.waitKey(0)

    card_rect = get_card(line_lists)
    print('检测到的卡片', card_rect)
    if len(card_rect) != 0:
        cv2.rectangle(img_copy3, (card_rect[0], card_rect[1]), (card_rect[2], card_rect[3]), (0, 0, 255), 2)
        cv2.imshow('frame_small_canny_hsv', img_copy3)
        cv2.waitKey(0)


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
