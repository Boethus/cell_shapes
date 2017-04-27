# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:01:34 2017

@author: univ4208
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.feature
import skimage.filters
from sklearn.preprocessing import normalize

import pywt
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

cA, (cH, cV, cD) = pywt.dwt2(im,"haar")
for i in range(0):
    cA, (cH, cV, cD) = pywt.dwt2(cA,"haar")
    
im3 = skimage.filters.sobel(cA)

im3, (cHs, cVs, cDs) = pywt.dwt2(im3,"haar")
im3 = skimage.filters.gaussian_filter(im3,2)
si(im3)
cA=im3
plt.figure()
plt.subplot(122)
plt.imshow(cA>skimage.filters.threshold_otsu(cA))
plt.subplot(121)
plt.imshow(cA)