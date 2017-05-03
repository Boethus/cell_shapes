# -*- coding: utf-8 -*-
"""
Created on Wed May 03 11:20:33 2017

@author: univ4208
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.feature
import skimage.filters
from sklearn.preprocessing import normalize
import cv2
import isingmrf_spmat as mrf
import scipy.ndimage as ndi
import pywt

plt.close("all")

def openFrame(number):
    name = os.path.join("..","data","yoshi_mov_2",str(number)+".tif")
    img = Image.open(name)
    img = np.asarray(img)
    return img

def si(img,title=None):
    plt.figure()
    plt.imshow(img,cmap='gray')
    if title:
        plt.title(title)

def displayWlt(wlt):
    cA, (cH, cV, cD)=wlt
    shapes = (cA.shape[0]+cV.shape[0],cA.shape[1]+cV.shape[1])
    out = np.zeros(shapes)
    out[:cA.shape[0],:cA.shape[1]]=cA
    out[cA.shape[0]:,cA.shape[1]:]=cD
       
    out[:cH.shape[0],cA.shape[1]:]=cH
    out[cA.shape[0]:,:cV.shape[1]]=cV
    return out

def displayMulti(wlt_list):
    cA = wlt_list[0]
    
    for i in range(1,len(wlt_list)):
        print i, cA.shape,wlt_list[i][0].shape
        cA = displayWlt((cA,wlt_list[i]))
    plt.figure()
    plt.imshow(cA,cmap="gray")
    plt.title("Wavelet decomposition")
    
def abe(img,variance):
    """proceeds to the Amplitude-scale invariant Bayes Estimation (ABE)"""
    nominator = img**2-3*variance
    nominator[nominator<0] = 0
    out = np.divide(nominator,img)
    out[img==0]=0
    return out

def getNoiseVar(img):
    """Gets the nth% of lower intensity pixels in an image correspondin to the noise
    n is determined empirically."""
    last_val = np.percentile(img,95)
    #si(img<last_val,title="Pixel values considered as noise")
    return np.var(img[img<last_val])

def filter_by_size(img_segm,mini_nb_pix):
    """filters a segmented image by getting rid of the components with too few pixels"""
    numbers = np.zeros(np.max(img_segm)-1)
    for i in range(1,np.max(img_segm)):
        numbers[i-1] = np.sum(img_segm==i)
        
    indexes = np.arange(1,np.max(img_segm))
    #indexes = indexes[numbers>np.mean(numbers)] #Deletes the 1-pixel elements
    indexes = indexes[numbers>mini_nb_pix] #Deletes the 1-pixel elements
    
    segm_filtered = np.zeros(img_segm.shape)
    j=1
    for i in (indexes):
        segm_filtered[img_segm==i] = j
        j+=1
    return segm_filtered