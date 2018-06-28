# -*- coding: UTF-8 -*-
import cv2
import numpy as np

'''
detect the region to recognition

'''



def morph_process(gray):   #elimate some region in order to detect the text
    ret, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    h,w = binary.shape

    #下半部分消除：
    ele = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_up = binary[0:h * 2 / 5, 0:w]
    binary_medium = binary[h * 2 / 5:h * 3 / 5, 0:w]
    binary_down = binary[h * 3 / 5:, 0:w]

    eroded_down = cv2.erode(binary_down, ele, iterations=1)
    binary = np.concatenate((binary_up,binary_medium,eroded_down))

    ele_up = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    ele_medium= cv2.getStructuringElement(cv2.MORPH_RECT, (40, 10))
    ele_down = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 10))
    binary_up = binary[0:h*2/5, 0:w]
    binary_medium = binary[h*2/5:h*3/5,0:w]
    binary_down = binary[h*3/5:,0:w]





    dilation_up = cv2.dilate(binary_up, ele_up, iterations=1)
    dilation_medium = cv2.dilate(binary_medium, ele_medium, iterations=1)
    dilation_down = cv2.dilate(binary_down, ele_down, iterations=1)
    dilation = np.concatenate((dilation_up, dilation_medium,dilation_down), axis = 0)

    return dilation


def findTextRegion(img):
    region = []
    # 1. 查找轮廓
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if (area < 5000):
            continue
        # 面积大的也不用
        if(area >40000):
            continue
        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])


        if (height *2 > width ): # 处理不扁的
            continue

        if(max(box[0][1] , box[2][1])> img.shape[0]*5/6): # 处理过于靠近底层的
            continue

        if(width < img.shape[1]/9): #处理过短的:
            continue

        if (width * 2 > img.shape[1]): # 处理过于宽的
            continue

        if(height<img.shape[0]/23): #处理过于细的
            continue

        if max(box[0][1], box[2][1]) > img.shape[0] * 4/5:
            continue


        if(max(box[2][0],box[0][0])  < img.shape[1]/4 and max(box[0][1] , box[2][1]) > img.shape[0]*1/2): #处理下左半部分
            continue

        if (max(box[2][0], box[0][0]) < img.shape[1] / 3 and min(box[0][1], box[2][1]) < img.shape[0] * 1/ 3): #处理上左
            continue
        if(min(box[0][1] , box[2][1]) < img.shape[0]/5):  #处理过于靠上的字符
            continue
        if (min(box[0][0], box[2][0]) > img.shape[1]/2):  # 中心往右的部分删除
            continue
        region.append(box)
    minwidth = 100000
    num = -1
    index = 10000
    for reg in region:  #将宽度最短的去掉
        num = num + 1
        width = abs(reg[0][0] - reg[2][0])
        if width < minwidth:
            minwidth = width
            index = num
    del region[index]
    return region


def grayImg(img):
    # 转化为灰度图

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 47,20)
    return gray


def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = morph_process(gray)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)

    # 4. 用绿线画出这些找到的轮廓
    ii = 0
    idImgs = []
    for box in region:
        h = abs(box[0][1] - box[2][1])
        w = abs(box[0][0] - box[2][0])
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        y1 = min(Ys)
        img2 = img.copy()
        if w > 0 and h > 0:
            idImg = grayImg(img2[y1:y1 + h, x1:x1 + w])
            ii += 1
            idImgs.append(idImg)
    return idImgs