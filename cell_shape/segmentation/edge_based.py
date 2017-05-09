# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:39:04 2017

@author: univ4208
"""

##########################################################################
#  This Script is designed to prototye of segmentation algorithm
#  for yoshi cells. The idea here is to use edge detection.
#########################################################################
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.feature
import skimage.filters
from sklearn.preprocessing import normalize

plt.close("all")
def openFrame(number):
    name = os.path.join("..","data","yoshi_mov_2",str(number)+".tif")
    img = Image.open(name)
    img = np.asarray(img)
    return img

def si(img):
    plt.figure()
    plt.imshow(img)
    
im = openFrame(11)
max_val = np.iinfo(im.dtype).max

sigm = 2
#/!\ The thresholds in canny have to be defined with respect to the max value of dtype
m=np.max(im)
lt = 0.01*m
ht = 0.15*m

im2 = skimage.feature.canny(im,sigma=sigm,low_threshold= lt,high_threshold=ht)


sigm2=2
#im2 = skimage.filters.gaussian(im2,sigm2)


im3 = skimage.filters.gaussian(im,sigm)
si(im3)
im3 = skimage.filters.sobel(im3)
im3 =  skimage.filters.gaussian(im3,sigm2)
im3=im3/np.max(im3)
si(im3>skimage.filters.threshold_otsu(im3))
plt.colorbar()

plt.figure()
plt.subplot(121)
plt.imshow(im2)
plt.subplot(122)
plt.imshow(im)

"""from skimage.filters import try_all_threshold
try_all_threshold(im3)"""