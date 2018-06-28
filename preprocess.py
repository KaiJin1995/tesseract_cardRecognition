# -*- coding: UTF-8 -*-
import cv2
import numpy as np

'''
preprocess:locate the card

'''


def img_change(img_a):  # 保留蓝色部分
    img_shape = img_a.shape
    img = img_a.copy()
    h = img_shape[0]
    w = img_shape[1]
    for i in range(h):
        for j in range(w):
            if (img[i, j, 0] > img[i, j, 1] and img[i, j, 0] > img[i, j, 2] and img[i, j, 0] > 100):
                img[i, j, 1] = 0
                img[i, j, 2] = 0
                img[i, j, 0] = 255
            else:
                img[i, j, 1] = 0
                img[i, j, 2] = 0
                img[i, j, 0] = 0
    return img



def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
               pts[1][0]:pts[2][0]]
    return img_crop

def img_preprocess(img):
    img_rgb = cv2.resize(img, (1000, 1600))



    img_save = img_change(img_rgb)
    gray_b = cv2.split(img_save)[0]
    ret, binary = cv2.threshold(gray_b, 180, 255, cv2.THRESH_BINARY)
    ele = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    eroded = cv2.erode(binary, ele)
    ele = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
    diated = cv2.dilate(eroded, ele)
    image, contours, hierarchy = cv2.findContours(diated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        # areas.append(area)
        if area < 400000:
            continue
        rect = cv2.minAreaRect(cnt)
        img_crop = crop_minAreaRect(img_rgb, rect)

    img_crop = cv2.resize(img_crop, (1600, 1000))
    return img_crop