#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 22:40:25 2018

@author: ramesh
"""
import numpy as np
import csv
#import tensorflow as tf
#import cv2
from PIL import Image
img=Image.open("result3.png")
img.show()


width, height=img.size
format=img.format
mode=img.mode

img_grey=img.convert('L')
print(width,"   ", height)
img_grey.save('result.jpg')
img_grey.show()

imgarray=np.array(img_grey, dtype=int)
imgarray=list(list(imgarray))
print (imgarray)
with open("img_pixels.csv", 'a') as f:
    writer = csv.writer(f)
    writer.writerow(imgarray)
