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
import cv2

plt.close('all')
def openFrame(number):
    name = os.path.join("..","data","microglia","RFP1_denoised","filtered_Scene1Interval"+str(number)+"_RFP.png")
    img = Image.open(name)
    img = np.asarray(img)
    return img

#Use the gaussianed filtered image to detect the centers and the 
#original image for the most accurate segmentation available
img = openFrame(107)
im_gaus=skimage.filters.gaussian(img,2)
from skimage.morphology import disk

minima = filters.minimum_filter(im_gaus, footprint = disk(4))
minima = (minima ==im_gaus)
threshold = skimage.filters.threshold_li(im_gaus)
threshold = im_gaus>threshold

minima = np.logical_and(minima,threshold)
    
m.show_points_on_img(minima,img)

labels, nr = ndi.label(threshold)

#m.overlay_mask2image(img,labels)
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(50,50))
img = clahe.apply(img)
nibl = skimage.filters.threshold_niblack(img,window_size=51,k=-0.1)
m.si(img>nibl,"niblack thresholding")
#skimage.filters.try_all_threshold(img)
threshold = img>skimage.filters.threshold_li(img)
m.si2(img,threshold,"original","thresholded")
threshold=threshold.astype(np.uint8)
kernel = np.ones((5,5),dtype = np.uint8)
threshold_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel,iterations=3)
m.si2(threshold,threshold_open,"threshold","open")
m.si2(threshold,threshold_open-threshold,"threshold","difference")

img_nibl = img>nibl
img_nibl = img_nibl.astype(np.uint8)
img_nibl = cv2.morphologyEx(img_nibl, cv2.MORPH_OPEN, kernel,iterations=1)   #Noise removal
m.si(img_nibl,"niblack after some opening")
nibl_open = cv2.morphologyEx(img_nibl, cv2.MORPH_OPEN, kernel,iterations=3)
m.si2(nibl_open,img_nibl-nibl_open,"niblack opened","difference")