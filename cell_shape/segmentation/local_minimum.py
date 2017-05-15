#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:39:45 2017

@author: aurelien
"""

import numpy as np
import scipy.ndimage.filters as filters
import skimage
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import os
import methods as m
from PIL import Image
import skimage.filters
plt.close('all')
def openFrame(number):
    name = os.path.join("..","data","microglia","RFP1_denoised","filtered_Scene1Interval"+str(number)+"_RFP.png")
    img = Image.open(name)
    img = np.asarray(img)
    return img

img = openFrame(107)
im_gaus=skimage.filters.gaussian(img,2)
from skimage.morphology import disk

minima = filters.minimum_filter(im_gaus, footprint = disk(4))
minima = (minima ==im_gaus)
threshold = skimage.filters.threshold_li(im_gaus)
threshold = im_gaus>threshold
m.si(threshold)
m.si(np.logical_and(minima,threshold))

minima = np.logical_and(minima,threshold)
    
m.show_points_on_img(minima,img)

labels, nr = ndi.label(threshold)

m.overlay_mask2image(img,labels)