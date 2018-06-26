# -*- coding: UTF-8 -*-
import cv2
#import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from PIL import Image
import argparse
#import string
debug = 1


parser = argparse.ArgumentParser(description='Image Path')
parser.add_argument('path', type=str)
args = parser.parse_args()





'''
prepare:detect the card

'''



img_file = args.path

#img_file = '/home/freedom/OCR/image-e4d5740b-74ef-4024-a0a4-26be6a12555d1790490136.jpg'
img_rgb = cv2.imread(img_file)
img_rgb = cv2.resize(img_rgb, (1000,1600))

#检测蓝色

def img_change(img_a):   #保留蓝色部分
    img_shape = img_a.shape
    img = img_a.copy()
    h = img_shape[0]
    w = img_shape[1]
    for i in range(h):
        for j in range(w):
            if(img[i,j,0] > img[i,j,1] and img[i,j,0] > img[i,j,2] and img[i,j,0]>100):
                img[i,j,1] = 0
                img[i,j,2] = 0
                img[i,j,0] = 255
            else:
                img[i, j, 1] = 0
                img[i, j, 2] = 0
                img[i, j, 0] = 0
    return img




img_save = img_change(img_rgb)
#cv2.imwrite('/home/freedom/OCR/testPic/img_change.jpg', img_save)
gray_b = cv2.split(img_save)[0]
ret, binary = cv2.threshold(gray_b, 180, 255, cv2.THRESH_BINARY)
binary_not = cv2.bitwise_not(binary)



#cv2.imwrite('/home/freedom/OCR/testPic/img_change.jpg', img_save)
ele = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
eroded = cv2.erode(binary, ele)
eroded_not = cv2.bitwise_not(eroded)
ele = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))

diated = cv2.dilate(eroded, ele)
# cv2.imwrite('/home/freedom/OCR/testPic/binary.jpg', binary)
# cv2.imwrite('/home/freedom/OCR/testPic/erode.jpg', eroded)
# cv2.imwrite('/home/freedom/OCR/testPic/diated.jpg', diated)
image, contours, hierarchy = cv2.findContours(diated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areas = []
region = []
def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]
    # cv2.imwrite('/home/freedom/OCR/testPic/crop.jpg', img_crop)
    return img_crop

for i in range(len(contours)):
    cnt = contours[i]
    area = cv2.contourArea(cnt)
   # areas.append(area)
    if area<400000:
        continue
    rect = cv2.minAreaRect(cnt)
    img_crop = crop_minAreaRect(img_rgb, rect)



img_crop = cv2.resize(img_crop, (1600,1000))
# cv2.imwrite('/home/freedom/OCR/testPic/del.jpg',img_crop)










#
# img = cv2.imread(img_file)
# gray_b = cv2.split(img)[0]
# ret, binary = cv2.threshold(gray_b, 180, 255, cv2.THRESH_BINARY)
# binary_not = cv2.bitwise_not(binary)
# ele = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
# eroded = cv2.erode(binary, ele)
# eroded_not = cv2.bitwise_not(eroded)
# #cv2.imwrite('/home/freedom/OCR/testPic/binary.jpg', binary)
# #cv2.imwrite('/home/freedom/OCR/testPic/erode.jpg', eroded)
# diated = cv2.dilate(eroded, ele)
# #cv2.imwrite('/home/freedom/OCR/testPic/diated.jpg', diated)
# image, contours, hierarchy = cv2.findContours(diated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# areas = []
# region = []
# def crop_minAreaRect(img, rect):
#
#     # rotate img
#     angle = rect[2]
#     rows,cols = img.shape[0], img.shape[1]
#     M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
#     img_rot = cv2.warpAffine(img,M,(cols,rows))
#
#     # rotate bounding box
#     rect0 = (rect[0], rect[1], 0.0)
#     box = cv2.boxPoints(rect)
#     pts = np.int0(cv2.transform(np.array([box]), M))[0]
#     pts[pts < 0] = 0
#
#     # crop
#     img_crop = img_rot[pts[1][1]:pts[0][1],
#                        pts[1][0]:pts[2][0]]
#     cv2.imwrite('/home/freedom/OCR/testPic/crop.jpg', img_crop)
#     return img_crop
#
# for i in range(len(contours)):
#     cnt = contours[i]
#     area = cv2.contourArea(cnt)
#    # areas.append(area)
#     if area<200000:
#         continue
#     rect = cv2.minAreaRect(cnt)
#     img_crop = crop_minAreaRect(img, rect)
# #img = cv2.resize(img_crop, (1600, 1000), interpolation=cv2.INTER_CUBIC)
# #cv2.imwrite('/home/freedom/OCR/testPic/resize.jpg', img)
# #     box = cv2.boxPoints(rect)
# #     box = np.int0(box)
# #     height = abs(box[0][1] - box[2][1])
# #     width = abs(box[0][0] - box[2][0])
# #     region.append(box)
# # for box in region:
# #     h = abs(box[0][1] - box[2][1])
# #     w = abs(box[0][0] - box[2][0])
# #     Xs = [i[0] for i in box]
# #     Ys = [i[1] for i in box]
# #     x1 = min(Xs)
# #     y1 = min(Ys)
# #     if w > 0 and h > 0:
# #         cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
# #     img_crop = img[y1:y1+h, x1:x1+w]
# # cv2.imwrite("contours.png", img_crop)


'''
detection and recognize
由于颜色变换容易出现遮挡的情况，所以不采用这种方法
'''

#
# def img_change(img):   #将黄色部分变成蓝色
#     img_shape = img.shape
#     h = img_shape[0]
#     w = img_shape[1]
#     for i in range(h):
#         for j in range(w):
#             if(img[i,j,0] < img[i,j,1] and img[i,j,0] < img[i,j,2]):
#                 img[i,j,1] = 0
#                 img[i,j,2] = 0
#                 img[i,j,0] = 255
#     return img
# img_crop = img_change(img_crop)


def preprocess(gray):
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

  #  dilation = cv2.dilate(binary, ele, iterations=1)

    # cv2.imwrite('/home/freedom/OCR/testPic/'+"binary.png", binary)
    # cv2.imwrite('/home/freedom/OCR/testPic/'+"dilation.png", dilation)

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
        if debug:
            print("rect is: ", rect)

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

     #   if (box[0][0] < img.shape[1]/5):
       #     continue

       # if (height * 18 < width):
       #     continue
        #if (width < img.shape[1] /4 and height < img.shape[0] / 50):
        #if (height)

        #if(width * height >30000 and width * height <100000):
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
    #gray = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = gray[:,:,0]
    # cv2.imwrite('/home/freedom/OCR/testPic/gray_test.png', gray)
    #retval, gray = cv2.threshold(gray, 60, 255,  cv2.THRESH_BINARY) #太小的话  颜色不深的字母就没轮廓了,太大的话，对于阴影情况识别差
    #gray = cv2.threshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, )
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 47,20)
    return gray


def detect(img):
    # fastNlMeansDenoisingColored(InputArray src, OutputArray dst, float h=3, float hColor=3, int templateWindowSize=7, int searchWindowSize=21 )
    #gray = cv2.fastNlMeansDenoisingColored(img, None, 10, 3, 3, 3)
   # coefficients = [0, 1, 1]
    #m = np.array(coefficients).reshape((1, 3))
   # gray = cv2.transform(gray, m)
    #gray = img[:,:,0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # if debug:
        # cv2.imwrite('/home/freedom/OCR/testPic/'+"gray.png", gray)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

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
        #if w > 0 and h > 0 and x1 < gray.shape[1] / 2:
        if w > 0 and h > 0:
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            #idImg = img2[y1:y1 + h, x1:x1 + w]
            idImg = grayImg(img2[y1:y1 + h, x1:x1 + w])
            ##idImg = cv2.resize(idImg, (2500,300))
            # cv2.imwrite('/home/freedom/OCR/testPic/'+str(ii) + ".png", idImg)
            #break
            ii += 1
            idImgs.append(idImg)

  #  if debug:
        # 带轮廓的图片
        # cv2.imwrite("/home/freedom/OCR/testPic/contours.png", img)
    return idImgs


def crop_image(img, tol=0):
    mask = img < tol
    return img[np.ix_(mask.any(1), mask.any(0))]




#f = open('/home/freedom/OCR/img_txt.txt', 'w')

def ocrIdCard(img, realId=""):

    #img = cv2.imread(img, cv2.IMREAD_COLOR)
    #img = cv2.resize(img, (800, 1600), interpolation=cv2.INTER_CUBIC)
    idImgs = detect(img)
    lens = len(idImgs)
    t= '0123456789-'

    for num, idImg in enumerate(idImgs):
        #idImg = cv2.resize(idImg,(1000,1000))
        image = Image.fromarray(idImg)
        tessdata_dir_config = '--psm 7'
        #print("checking")
        #print(realId)
        result = pytesseract.image_to_string(image,  lang='eng', config=tessdata_dir_config)
        # if num == 0 or num == 2:
        #     for item in str(result):
        #         if item not in t:
        #             item


        if lens == 3:
            result_3 = result
            if num == 0:
                result = str(result).replace(':','')
           # result = str(result).replace('')

        #f.write(str(result) + '\n')

        if lens ==4:
            result_4 = result
            if num == 0 or num ==1:
                for i ,item in enumerate(str(result_4)):
                    if item not in t:
                        result = str(result).replace(str(result_4)[i],'')




        print(result)



        # print(pytesseract.image_to_string(image, lang='eng', config=tessdata_dir_config))
        # if debug:
        #     f, axarr = plt.subplots(2, 3)
        #     axarr[0, 0].imshow(cv2.imread(imgPath))
        #     axarr[0, 1].imshow(cv2.imread("gray.png"))
        #     axarr[0, 2].imshow(cv2.imread("binary.png"))
        #     axarr[1, 0].imshow(cv2.imread("dilation.png"))
        #     axarr[1, 1].imshow(cv2.imread("contours.png"))
        #     axarr[1, 2].set_title("exp:" + realId + "\nocr:" + result)
        #     axarr[1, 2].imshow(cv2.imread("0.png"))
        #     plt.show()
ocrIdCard(img_crop)



