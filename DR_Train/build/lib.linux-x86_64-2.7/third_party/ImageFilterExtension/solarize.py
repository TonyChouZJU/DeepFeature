#!/usr/bin/env python
#coding=utf-8
'''
Created on 2011-6-27

@author: Chine
'''

from PIL import Image
import numpy as np
import cv2

def solarize(img):
    img[img < 128] ^= 0xFF
    return img

'''
def solarize(img):
    # @效果：曝光
    # @param img: instance of Image
    # @return: instance of Image
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img.point(lambda i: i ^ 0xFF if i < 128 else i)
'''

if __name__ == "__main__":
    import sys, os, time

    path = os.path.dirname(__file__) + os.sep.join(['./', 'images', 'lam.jpg'])
    path = os.path.join(os.path.dirname(__file__), 'images', 'lam.jpg')
    
    if len(sys.argv) == 2:
        path = sys.argv[1]

    start = time.time()
    
    img = Image.open(path)
    img = cv2.imread(path)[:, :, (2, 1, 0)].astype(np.uint32)
    img = solarize(img)
    img_save = Image.fromarray(np.uint8(img))
    img_save.save(os.path.splitext(path)[0]+'.solarize_2.jpg', 'JPEG')

    end = time.time()
    print 'It all spends %f seconds time' % (end-start) 
