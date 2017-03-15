#!/usr/bin/env python
#coding=utf-8
'''
Created on 2011-7-6

@author: Chine
'''

import random
from PIL import Image
import numpy as np
import cv2


def diffuse(img, degree=1):
    height, width = img.shape[0:2]
    r_idx_array, c_idx_array = np.where(img[:, :, 0] < 256)
    r_idx_array = r_idx_array.reshape(height, width)
    c_idx_array = c_idx_array.reshape(height, width)
    rdm_array = np.random.randint(-degree, degree, size=(2, height, width))
    r_idx_array += rdm_array[0]
    c_idx_array += rdm_array[1]
    new_r_idx_array = np.minimum(np.maximum(r_idx_array, 0), height - 1)
    new_c_idx_array = np.minimum(np.maximum(c_idx_array, 0), width - 1)
    #new_r_idx_array = new_r_idx_array.flatten()
    #new_c_idx_array = new_c_idx_array.flatten()
    #img = img[new_r_idx_array, new_c_idx_array].reshape(height, width, 3)
    img = img[new_r_idx_array, new_c_idx_array]
    return img


if __name__ == "__main__":
    import sys, os, time

    path = os.path.join( os.path.dirname(__file__), 'images', 'lam.jpg')
    #degree = 16
    degree = 2 
    
    if len(sys.argv) == 2:
        try:
            degree = int(sys.argv[1])
        except ValueError:
            path  = sys.argv[1]
    elif len(sys.argv) == 3:
        path = sys.argv[1]
        degree = sys.argv[2]

    start = time.time()

    img = cv2.imread(path)[:, :, (2, 1, 0)].astype(np.uint32)
    img = diffuse(img, degree)
    img = Image.fromarray(np.uint8(img))
    img.save(os.path.splitext(path)[0]+'.diffuse_3.jpg', 'JPEG')

    end = time.time()
    print 'It all spends %f seconds time' % (end-start)
