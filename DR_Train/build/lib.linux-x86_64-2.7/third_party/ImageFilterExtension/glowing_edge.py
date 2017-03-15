#!/usr/bin/env python
#coding=utf-8
'''
Created on 2011-7-8

@author: Chine
'''

import math
from PIL import Image
import numpy as np
import cv2


def glowing_edge(img):
    width, height, _ = img.shape
    bottom = img[1:, :-1, :]  # 下方像素点
    right = img[:-1, 1:, :]   # 右方像素点
    current = img[:-1, :-1, :] #当前像素点

    img_array = np.sqrt((current-bottom)**2 + (current-right)**2) * 2
    img_array = np.minimum(np.maximum(img_array, 0), 255).astype(np.uint8)
    if img_array.shape[-1] == 3:
        img_png = cv2.cvtColor(img_array, cv2.COLOR_RGB2RGBA)
    else:
        img_png = img_array
    return img_png


if __name__ == "__main__":
    import sys, os, time

    path = os.path.join(os.path.dirname(__file__), 'images', 'lam.jpg')
    
    if len(sys.argv) == 2:
        path = sys.argv[1]

    start = time.time()

    img = cv2.imread(path)[:, :, (2, 1, 0)].astype(np.uint32)
    img = glowing_edge(img)
    img_save = Image.fromarray(np.uint8(img))
    img_save.save(os.path.splitext(path)[0]+'.glowing_edge_2.jpg', 'JPEG')

    end = time.time()
    print 'It all spends %f seconds time' % (end-start)