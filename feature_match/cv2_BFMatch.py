#opencv----特征匹配----BFMatching
import cv2
from matplotlib import pyplot as plt
#读取需要特征匹配的两张照片，格式为灰度图。
template=cv2.imread("image/template1.png",0)
target=cv2.imread("image/target_1032.png",0)
orb=cv2.ORB_create()#建立orb特征检测器
kp1,des1=orb.detectAndCompute(template,None)#计算template中的特征点和描述符
kp2,des2=orb.detectAndCompute(target,None) #计算target中的
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True) #建立匹配关系
mathces=bf.match(des1,des2) #匹配描述符
mathces=sorted(mathces,key=lambda x:x.distance) #据距离来排序
result= cv2.drawMatches(template,kp1,target,kp2,mathces[:40],None,flags=2) #画出匹配关系
plt.imshow(result),plt.show() #matplotlib描绘出来
'''
————————————————
版权声明：本文为CSDN博主「Demon_Hunter」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/zhuisui_woxin/article/details/84400439
https://www.jianshu.com/p/5083f8d75439 文章还对比了AKAZE，可以知道这个算法是偏向于相同的图之间的匹配
我换成都是直线的图，按理说都能匹配上，但是很难
https://www.cnblogs.com/alexme/p/11353137.html 这篇文章里有对这一算法的详细讲解
'''