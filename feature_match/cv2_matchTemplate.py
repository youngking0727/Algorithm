# https://blog.csdn.net/zhuisui_woxin/article/details/84400439
#opencv模板匹配----单目标匹配，这种匹配方式是必须图像一模一样
'''
模板匹配的原理其实很简单，就是不断地在原图中移动模板图像去比较，有6种不同的比较方法，详情可参考：TemplateMatchModes

平方差匹配CV_TM_SQDIFF：用两者的平方差来匹配，最好的匹配值为0
归一化平方差匹配CV_TM_SQDIFF_NORMED
相关匹配CV_TM_CCORR：用两者的乘积匹配，数值越大表明匹配程度越好
归一化相关匹配CV_TM_CCORR_NORMED
相关系数匹配CV_TM_CCOEFF：用两者的相关系数匹配，1表示完美的匹配，-1表示最差的匹配
归一化相关系数匹配CV_TM_CCOEFF_NORMED

作者：ex2tron
链接：https://www.jianshu.com/p/c20adfa72733
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''

import cv2
# 读取目标图片
target = cv2.imread("image/target.jpg",0)
# 读取模板图片
template = cv2.imread("image/template.jpg",0)

# 获得模板图片的高宽尺寸
theight, twidth = template.shape[:2]
template2 = cv2.resize(template, (int(0.5 * twidth), int(0.5 * theight)))
cv2.imshow('haha', template)
cv2.waitKey(0)
# 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
# 返回值是一个矩阵
result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
print(type(result),result.shape, target.shape, template.shape)
# 归一化处理
cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
# 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
'''
函数功能：假设有一个矩阵a,现在需要求这个矩阵的最小值，最大值，并得到最大值，最小值的索引。
咋一看感觉很复杂，但使用cv2.minMaxLoc()函数就可全部解决。函数返回的四个值就是上述所要得到的。
'''
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# 匹配值转换为字符串
# 对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
# 对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
strmin_val = str(min_val)
# 绘制矩形边框，将匹配区域标注出来
# min_loc：矩形定点
# (min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
# (0,0,225)：矩形的边框颜色；2：矩形边框宽度
cv2.rectangle(target, min_loc, (int(min_loc[0]+0.5 * twidth), int(min_loc[1]+0.5 * theight)),(0,0,225),2)
# 显示结果,并将匹配值显示在标题栏上
cv2.imshow("MatchResult----MatchingValue="+strmin_val,target)
cv2.waitKey()
cv2.destroyAllWindows()
'''
————————————————
版权声明：本文为CSDN博主「Demon_Hunter」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/zhuisui_woxin/article/details/84400439
'''