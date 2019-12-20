import cv2
import numpy as np
import argparse
import pdb # 调试器 https://www.cnblogs.com/xiaohai2003ly/p/8529472.html

# https://www.jianshu.com/p/13aeae6ed2b9
# https://blog.csdn.net/u012005313/article/details/69675803
# https://blog.csdn.net/wangleixian/article/details/78164475 滑动条做调色板，橡皮擦，图像过渡
# https://blog.csdn.net/u012005313/article/details/69675803 回调的方式调用膨胀操作
args = argparse.ArgumentParser()
args.add_argument("-i", "--imgName", help="image dir")
args.add_argument("-r", "--canny_ratio", default=3, type=int,
                  help="canny ratio")
args = args.parse_args()


class TrackerProp:
    def __init__(self, _name, _curr, _max):
        self.name = _name
        self.curr = _curr
        self.max = _max


def callbacks(x):
    pass


# 这个方法是获取阈值的方法
def get_trackbar_values(win_name, trackbars):
    values = []
    for bar in trackbars:
        values.append(cv2.getTrackbarPos(bar.name, win_name))
    return values


def setup_trackbars(win_name, trackbars):
    cv2.namedWindow(win_name)
    cv2.resizeWindow(win_name, 500, 500)
    for bar in trackbars:
        cv2.createTrackbar(bar.name, win_name, bar.curr, bar.max, callbacks)


if __name__ == "__main__":
    img = cv2.imread(args.imgName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bar_diameter = TrackerProp("bi_diameter", 15, 300)  # 设置滚动条名称，当前值，最大值
    bar_bisigma = TrackerProp("bi_sigma", 88, 200)
    bar_canny = TrackerProp("canny", 37, 300)
    # setup initial values
    bi_diameter, bi_sigma, canny_thresh = bar_diameter.curr, bar_bisigma.curr, bar_canny.curr
    trackbars = [bar_diameter, bar_bisigma, bar_canny]
    print('trackbars: ', trackbars)

    win_name = "test_trackbar"  # 15, 88, 37
    setup_trackbars(win_name, trackbars)

    while True:
        values = get_trackbar_values(win_name, trackbars)
        bi_diameter, bi_sigma, canny_thresh = values
        outline = np.zeros_like(gray, dtype='uint8')
        canny_ratio = args.canny_ratio
        gray_bifilt = cv2.bilateralFilter(gray, bi_diameter, bi_sigma, bi_sigma)
        edged = cv2.Canny(gray_bifilt, canny_thresh, canny_thresh * canny_ratio)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        cv2.drawContours(outline, [cnts], -1, 255, -1)
        cv2.imshow("img", outline)

        if cv2.waitKey(1) & 0xFF is ord('q'):
            print(values)
            break
    bg_thresh = cv2.bitwise_not(outline)
    img[bg_thresh > 0] = 0
    cv2.imwrite("result.jpg", img)

