#!/usr/bin/env python
#coding=utf-8
'''
Created on 2011-6-24

@author: Chine
'''
import math
from PIL import Image
from PIL import ImageFilter
import cv2
import numpy as np


def relief(img, angle=45):
    angle = np.random.randint(0, angle)
    img = Image.fromarray(img.astype(np.uint8))
    if angle < 0: angle = 0
    if angle > 360: angle = 360


    radian = angle * math.pi / 180
    pi4 = math.pi / 4

    matrix33 = (int(math.cos(radian + pi4) * 256),
                int(math.cos(radian + 2 * pi4) * 256),
                int(math.cos(radian + 3 * pi4) * 256),
                int(math.cos(radian) * 256),
                256,
                int(math.cos(radian + 4 * pi4) * 256),
                int(math.cos(radian - pi4) * 256),
                int(math.cos(radian - 2 * pi4) * 256),
                int(math.cos(radian - 3 * pi4) * 256))

    img = np.array(img.filter(ImageFilter.Kernel((3,3), matrix33, scale=256))).astype(np.uint8)

    return img

'''
def relief(img, angle):

    # @效果：彩色浮雕
    # @param img: instance of Image
    # @param angle: 进行卷积运算使用的其实偏移角度，大小范围[0, 360]
    # @return: instance of Image

    if angle < 0: angle = 0
    if angle > 360: angle = 360

    radian = angle * math.pi / 180
    pi4 = math.pi / 4

    # 进行卷积转换的3×3矩阵
    matrix33 = [
        [int(math.cos(radian + pi4) * 256),
         int(math.cos(radian + 2 * pi4) * 256),
         int(math.cos(radian + 3 * pi4) * 256)],
        [int(math.cos(radian) * 256),
         256,
         int(math.cos(radian + 4 * pi4) * 256)],
        [int(math.cos(radian - pi4) * 256),
         int(math.cos(radian - 2 * pi4) * 256),
         int(math.cos(radian - 3 * pi4) * 256)]
        ]

    m = Matrix33(matrix33, scale=256) # 缩放值256

    return m.convolute(img)
'''

if __name__ == "__main__":
    import sys, os, time

    path = os.path.join(os.path.dirname(__file__), 'images', 'lam.jpg')
    angle = 60
    
    if len(sys.argv) == 2:
        try:
            angle = int(sys.argv[1])
        except ValueError:
            path  = sys.argv[1]
    elif len(sys.argv) == 3:
        path = sys.argv[1]
        angle = int(sys.argv[2])

    start = time.time()
    
    #img = Image.open(path)
    img = cv2.imread(path)[:, :, (2, 1, 0)].astype(np.uint32)
    img = relief(img, angle)
    img_save = Image.fromarray(img)
    img_save.save(os.path.splitext(path)[0]+'.relief_2.jpg', 'JPEG')

    end = time.time()
    print 'It all spends %f seconds time' % (end-start)
