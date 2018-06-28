# -*- coding: UTF-8 -*-
"""
tesseract:4.0
engine:LSTM
python:2.7
"""

import cv2
import argparse
from preprocess import img_preprocess
from detection import detect
from recognition import recognition_card


parser = argparse.ArgumentParser(description='Image Path')
parser.add_argument('path', type=str)
args = parser.parse_args()



img_file = args.path
img_rgb = cv2.imread(img_file) #图像读入




if __name__ == '__main__':
    img_card = img_preprocess(img_rgb)
    detect_imgs = detect(img_card)
    lens = len(detect_imgs)
    t = '0123456789-'
    for num, detect_img in enumerate(detect_imgs):
        result = recognition_card(detect_img)
        # post-preprocess (reject some impossible character)
        if lens == 3:
            result_3 = result
            if num == 0:
                result = str(result).replace(':','')

        if lens ==4:
            result_4 = result
            if num == 0 or num ==1:
                for i ,item in enumerate(str(result_4)):
                    if item not in t:
                        result = str(result).replace(str(result_4)[i],'')
        print result


