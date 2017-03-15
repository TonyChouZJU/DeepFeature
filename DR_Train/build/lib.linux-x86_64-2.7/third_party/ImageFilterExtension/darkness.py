#!/usr/bin/env python
#coding=utf-8
'''
Created on 2011-6-30

@author: Chine
'''

from PIL import Image
import numpy as np
import cv2


def darkness(img):
    # @效果：暗调
    # @param img: instance of Image
    # @return: instance of Image
    img_array = img**2 / 255
    img_array = np.minimum(np.maximum(img_array, 0), 255)
    return img_array

if __name__ == "__main__":
    import sys, os, time

    path = os.path.join(os.path.dirname(__file__), 'images', 'lam.jpg')
    
    if len(sys.argv) == 2:
        path = sys.argv[1]

    start = time.time()

    img = cv2.imread(path)[:, :, (2, 1, 0)].astype(np.uint32)
    img = darkness(img)

    img = Image.fromarray(np.uint8(img))
    img.save(os.path.splitext(path)[0]+'.darkness_2.jpg', 'JPEG')

    end = time.time()
    print 'It all spends %f seconds time' % (end-start)
