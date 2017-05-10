#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:59:25 2017

@author: aurelien
"""

import matplotlib.pyplot as plt
import methods as m
import os
from PIL import Image
import numpy as np
import cv2
import skimage.filters
import scipy.ndimage as ndi
import time
plt.close('all')
frameNum = 130
path = os.path.join("..",'data','microglia','RFP1_denoised','filtered_Scene1Interval'+str(frameNum)+'_RFP.png')
img = Image.open(path)
im = np.asarray(img)

def proba_map_gaussian(img):
    """Optimal template matching to find a gaussian with std sigma in img"""
    list_of_sigmas = [40,30,20,10]
    list_of_sigmas = range(9,40)
    list_of_results = []
    max_map = np.zeros(img.shape)
    coeffs_map = np.zeros(img.shape)
    method = 'cv2.TM_CCOEFF_NORMED'
    for sigma in list_of_sigmas:
        size=3*sigma
        if size%2==0:
            size+=1
        template = m.gaussian(size,sigma)
        template/=template.max()
        template*=255
        template = template.astype(np.uint8)
        
        w, h = template.shape[::-1]
        
        img2 = img.copy()
        meth = eval(method)
        # Apply template Matching
        res = cv2.matchTemplate(img2,template,meth)
        res = np.pad(res,size/2,mode="constant")
        indices = res>max_map
        max_map[indices] = res[indices]
        coeffs_map[indices] = sigma
        list_of_results.append(np.max(res))
    plt.figure()
    plt.plot(list_of_sigmas,np.asarray(list_of_results))
    plt.xlabel("sigma")
    plt.ylabel("maximum proba map value")
    plt.title("Finding a gaussian mask of size sigma in frame 130")
    m.si2(img,max_map,'original image','Map of maximum probabilities for template matching')
    return max_map,coeffs_map
max_map,coeffs_map=proba_map_gaussian(im)
plt.figure()
plt.subplot(131)
plt.imshow(im)
plt.subplot(132)
plt.imshow(max_map)
plt.subplot(133)
plt.imshow(coeffs_map)


########Superpixels##################################

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
 
image = im.copy()
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR);

# loop over the number of segments
for numSegments in (400, 500, 600):
	# apply SLIC and extract (approximately) the supplied number
	# of segments
	segments = slic(image, n_segments = numSegments, sigma = 5)
 
	# show the output of SLIC
	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(image, segments))
	plt.axis("off")
 
# show the plots
plt.show()