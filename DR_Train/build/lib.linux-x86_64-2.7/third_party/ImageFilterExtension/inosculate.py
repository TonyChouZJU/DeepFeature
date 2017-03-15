#!/usr/bin/env python
#coding=utf-8
'''
Created on 2011-7-3

@author: Chine
'''

from PIL import Image
import numpy as np
import cv2


def inosculate(bg_img, fg_img, transparency=128):
    if fg_img.shape[-1] == 3:
        fg_img_png = cv2.cvtColor(fg_img.astype(np.uint8), cv2.COLOR_RGB2RGBA).astype(np.uint32)
    else:
        fg_img_png = fg_img

    if bg_img.shape[-1] == 3:
        bg_img_png = cv2.cvtColor(bg_img.astype(np.uint8), cv2.COLOR_RGB2RGBA).astype(np.uint32)
    else:
        bg_img_png = bg_img

    bg_height, bg_width, _ = bg_img.shape
    fg_height, fg_width, _ = fg_img.shape
    height = min(bg_height, fg_height)
    width = min(bg_width, fg_width)

    dst_img = (fg_img_png[:height, :width, :] - bg_img_png[:height, :width, :]) * transparency / 255 + \
              bg_img_png[:height, :width, :]

    return dst_img


if __name__ == "__main__":
    import sys, os, time

    bg_img_path = os.path.join(os.path.dirname(__file__), 'images', 'guanlangaoshou.jpg')
    fg_img_path = os.path.join(os.path.dirname(__file__), 'images', 'lam.jpg')


    transparency = 128
    
    if len(sys.argv) == 2:
        transparency = int(sys.argv[1])
    elif len(sys.argv) == 3:
        bg_img_path = sys.argv[1]
        fg_img_path = sys.argv[2]
    elif len(sys.argv) == 4:
        bg_img_path = sys.argv[1]
        fg_img_path = sys.argv[2]
        transparency = int(sys.argv[3])

    start = time.time()

    bg_img = cv2.imread(bg_img_path)[:, :, (2, 1, 0)].astype(np.uint32)
    fg_img = cv2.imread(fg_img_path)[:, :, (2, 1, 0)].astype(np.uint32)
    img = inosculate(bg_img, fg_img, transparency)

    img_save = Image.fromarray(np.uint8(img))
    img_save.save(os.path.splitext(fg_img_path)[0]+'.inosculate_2.png', 'PNG')

    end = time.time()
    print 'It all spends %f seconds time' % (end-start)
