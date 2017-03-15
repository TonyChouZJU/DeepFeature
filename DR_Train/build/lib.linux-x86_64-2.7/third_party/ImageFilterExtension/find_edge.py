#!/usr/bin/env python
#coding=utf-8
'''
Created on 2011-7-7

@author: Chine
'''

import math
from PIL import Image
from PIL import ImageFilter
import cv2
import math
import numpy as np

from adjustment import invert

def find_edge(img, angle=60):
    radian = angle * 2.0 * math.pi / 360.0

    pi4 = math.pi / 4.0

    img = Image.fromarray(np.uint8(img))
    #注意顺序
    matrix33 = (
        int(math.cos(radian + pi4) * 256),
        int(math.cos(radian + 2.0 * pi4) * 256),
        int(math.cos(radian + 3.0 * pi4) * 256),
        int(math.cos(radian) * 256),
        0,
        int(math.cos(radian + 4.0 * pi4) * 256),
        int(math.cos(radian - pi4) * 256),
        int(math.cos(radian - 2.0 * pi4) * 256),
        int(math.cos(radian - 3.0 * pi4) * 256)

    )

    img = img.filter(ImageFilter.Kernel((3, 3), matrix33, scale=256))

    return invert(np.array(img))


if __name__ == "__main__":
    import sys, os, time

    path = os.path.join(os.path.dirname(__file__), 'images', 'lam.jpg')

    angle = 60
    
    if len(sys.argv) == 2:
        try:
            angle = int(sys.argv[1])
        except ValueError:
            path = sys.argv[1]
    elif len(sys.argv) == 3:
        path = sys.argv[1]
        angle = int(sys.argv[2])

    start = time.time()
    
    img = cv2.imread(path)[:, :, (2, 1, 0)].astype(np.uint32)
    img = find_edge(img, angle)
    img_save = Image.fromarray(np.uint8(img))
    img_save.save(os.path.splitext(path)[0]+'.find_edge_2.jpg', 'JPEG')

    end = time.time()
    print 'It all spends %f seconds time' % (end-start)
