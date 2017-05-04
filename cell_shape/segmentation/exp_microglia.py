# -*- coding: utf-8 -*-
"""
Created on Thu May 04 12:29:57 2017

@author: univ4208
"""

import methods as m

import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import skimage.feature
import skimage.filters
import cv2
import scipy.ndimage as ndi
import os
from PIL import Image
from skimage.filters import try_all_threshold

def gf(img,sigma):
    img2 = skimage.filters.gaussian(img,sigma)
    return img2

def removeGaussians(img):
    """Convolves the original image with different scales of gaussians
    If an object is gaussian, the values of the object convolved should be high."""
    #filtered = m.wavelet_denoising(img,lvl=2)
    filtered = np.copy(img)
    filtered = filtered/np.max(filtered)
    #Tries different thresholds
    t=skimage.filters.threshold_adaptive(filtered,301)
    t2=skimage.filters.threshold_otsu(filtered)
    otsu_bin = filtered>t2
    m.si2(t,otsu_bin,"adaptive thresh","otsu")
    #adapt the intensity of each cluster
    labels,nb = ndi.label(otsu_bin)
    for i in range(nb):
        filtered[labels==i+1] /= (np.max(filtered[labels==i+1]))
        
    m.si2(labels,filtered,'labels','filtered image')
    #filtered[~otsu_bin]=0
    
    total = np.ones(img.shape)
    total_maxwise = np.zeros(img.shape)
    for i in range(10,20):
        out = gf(filtered,i)
        total*=out
        total_maxwise[out>total_maxwise] = out[out>total_maxwise]
        
    m.si(total,"product of stuff")
    m.si(total_maxwise,"Maxima")
    print np.count_nonzero(total)
    ###################################################################
    ## Find the elements which have a high 'total' value corresponding to the out of focus stuff
    ## Inch Allah
    #######################################################################
    mask = total>skimage.filters.threshold_li(total)
    try_all_threshold(total)
    m.si(mask,"mask of the elements to remove")
    labels_to_remove = labels[mask]
    labels_to_remove = np.unique(labels_to_remove)
    
    filtered_array = np.copy(filtered)
    for lab in labels_to_remove:
        filtered_array[labels==lab]=0
    
    print "Nb elements supprimes: ", labels_to_remove.size
    m.si2(img,filtered_array,"Original image","Degaussianed image")

plt.close('all')

frame_number=230
#Opening desired image
name = os.path.join("..",'data','microglia','RFP1','Scene1Interval'+str(frame_number)+'_RFP.png')
img = Image.open(name)
img = np.asarray(img)

"""Use Watershed!!!"""

equilibrium = cv2.equalizeHist(img.astype(np.uint8))
m.si(equilibrium,'Histogram equilibrated image')
filtered = m.wavelet_denoising2(equilibrium,wlt='sym2',lvl=6,fraction=0.8)
m.si2(equilibrium,filtered,'Histogram equilibrated image','image filtered')
#cA= np.arctan(img/np.max(img)*np.pi*5)
for i in range(1,7):
    m.si(m.wavelet_denoising2(equilibrium,wlt='sym2',lvl=i,fraction=0.8),'Denoising lvl '+str(i))
#removeGaussians(filtered)
