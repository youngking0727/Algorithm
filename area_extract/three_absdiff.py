import cv2
import numpy as np
import time


def three_frame_differencing(videopath):
    cap = cv2.VideoCapture(videopath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('宽/高', width, height)
    one_frame = np.zeros((height, width), dtype=np.uint8),
    two_frame = np.zeros((height, width), dtype=np.uint8)
    three_frame = np.zeros((height, width), dtype=np.uint8)
    while cap.isOpened():
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame_gray = cv2.GaussianBlur(frame_gray, (3, 3), 0)
        if not ret:
            break
        one_frame, two_frame, three_frame = two_frame, three_frame, frame_gray
        abs1 = cv2.absdiff(one_frame, two_frame)  # 相减
        _, thresh1 = cv2.threshold(abs1, 5, 255, cv2.THRESH_BINARY)  # 二值，大于40的为255，小于0

        abs2 = cv2.absdiff(two_frame, three_frame)
        _, thresh2 = cv2.threshold(abs2, 5, 255, cv2.THRESH_BINARY)

        binary = cv2.bitwise_and(thresh1, thresh2)  # 与运算
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erode = cv2.erode(binary, kernel)  # 腐蚀
        dilate = cv2.dilate(erode, kernel)  # 膨胀
        dilate = cv2.dilate(dilate, kernel)  # 膨胀
        '''
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 1), np.uint8))
        img, contours, hei = cv2.findContours(binary, mode=cv2.RETR_EXTERNAL,
                                              method=cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        for contour in contours:
            if 100 < cv2.contourArea(contour) < 40000:
                x, y, w, h = cv2.boundingRect(contour)  # 找方框
                cv2.rectangle(binary, (x, y), (x + w, y + h), (255, 255, 255))
        # cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        # cv2.imshow("binary", binary)
        cv2.imshow("dilate", binary)
        #time.sleep(0.5)
        #cv2.imshow("frame", frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    three_frame_differencing('image/front_jeston_1.mp4')
    # three_frame_differencing('image/test4.mp4')
    # three_frame_differencing('image/jump_skip01.mov')

'''
————————————————
版权声明：本文为CSDN博主「人工弱智乔木」的原创文章，遵循
CC
4.0
BY - SA
版权协议，转载请附上原文出处链接及本声明。
原文链接：https: // blog.csdn.net / pengpengloveqiaoqiao / article / details / 89487049
'''