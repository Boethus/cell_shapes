# -*- coding: utf-8 -*-
"""
Created on Wed May 03 11:11:06 2017

@author: univ4208
"""

import methods as m

import numpy as np
import matplotlib.pyplot as plt
import skimage.feature
import skimage.filters
import cv2
import scipy.ndimage as ndi
import pywt

plt.close('all')
    


#wlt = pywt.Wavelet("rbio3.1")
wlt = 'sym2'

#A trous method
im = m.openFrame(67)
im = im[:-2,:]  #To change max level
im = skimage.filters.sobel(im)

total = m.wavelet_denoising(im)
img_f = m.segmentation(skimage.filters.gaussian(total,0.01))
img_g = m.segmentation(skimage.filters.gaussian(total,0.5))

plt.figure()
plt.subplot(121)
plt.imshow(img_f)
plt.subplot(122)
plt.imshow(img_g)
