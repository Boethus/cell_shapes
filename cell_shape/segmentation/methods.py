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
import cv2
import scipy.ndimage as ndi
import pywt

#plt.close("all")

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
        
def si2(im1,im2,title1=None,title2=None):
    plt.figure()
    plt.subplot(121)
    plt.imshow(im1)
    if title1:
        plt.title(title1)
    plt.subplot(122)
    plt.imshow(im2)
    if title2:
        plt.title(title2)

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

def getNoiseVar(img,fraction=0.95):
    """Gets the nth% of lower intensity pixels in an image correspondin to the noise
    n is determined empirically."""
    last_val = np.percentile(img,fraction)
    #si(img<last_val,title="Pixel values considered as noise")
    return np.var(img[img<last_val])

def filter_by_size(img_segm,mini_nb_pix):
    """filters a segmented image by getting rid of the components with too few pixels"""
    numbers = np.zeros(np.max(img_segm))
    for i in range(1,np.max(img_segm)+1):
        numbers[i-1] = np.sum(img_segm==i)
        
    indexes = np.arange(1,np.max(img_segm)+1)
    #indexes = indexes[numbers>np.mean(numbers)] #Deletes the 1-pixel elements
    indexes = indexes[numbers>mini_nb_pix] #Deletes the 1-pixel elements
    
    segm_filtered = np.zeros(img_segm.shape)
    j=1
    for i in (indexes):
        segm_filtered[img_segm==i] = j
        j+=1
    return segm_filtered

def fillHoles(img):
    out,contour,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    i=0
    for cnt in contour:
            cv2.drawContours(img,contour,i,255,-1)
            i+=1
    return img

def modulusMaximum(amplitude,angle):
    """Unused method which computes for each pixel if it is the maximum of its neighbors
    along the axis defined by angle"""
    angle = np.mod(angle+np.pi,np.pi)/np.pi  #The sign does not matter anyway, goes between 0 and 1
    
    #Do 4 cases: angle <0.25,0.5,0.75,>0.75
    results_plus = np.zeros(amplitude.shape,dtype = np.uint8)
    results_minus = np.zeros(amplitude.shape,dtype = np.uint8)
    
    ampl_pad = np.pad(amplitude,1,'constant')
    
    #Case angle is between 0 and 45deg, ie angle is between 0 and 0.25
    print "max angle",np.max(angle)
    tmp = amplitude>ampl_pad[1:-1,2:]
    results_plus[angle<0.25] = tmp[angle<0.25]
    tmp = amplitude>ampl_pad[1:-1,:-2]
    results_minus[angle<0.25] = tmp[angle<0.25]
    
    #Case angle between 0.25 and 0.5
    tmp = amplitude>ampl_pad[:-2,2:]
    results_plus[np.logical_and(angle>=0.25,angle<0.5)] = tmp[np.logical_and(angle>=0.25,angle<0.5)]
    tmp = amplitude>ampl_pad[2:,:-2]
    results_minus[np.logical_and(angle>=0.25,angle<0.5)] = tmp[np.logical_and(angle>=0.25,angle<0.5)]
       
    #Case angle between 0.5 and 0.75
    tmp = amplitude>ampl_pad[2:,1:-1]
    results_plus[np.logical_and(angle>=0.5,angle<0.75)] = tmp[np.logical_and(angle>=0.5,angle<0.75)]
    tmp = amplitude>ampl_pad[:-2,1:-1]
    results_minus[np.logical_and(angle>=0.5,angle<0.75)] = tmp[np.logical_and(angle>=0.5,angle<0.75)]
    
    #Case angle >0.75
    tmp = amplitude>ampl_pad[:-2,:-2]
    results_plus[angle>=0.75] = tmp[angle>=0.75]
    tmp = amplitude>ampl_pad[2:,2:]
    results_minus[angle>=0.75] = tmp[angle>=0.75]
    
    return np.logical_and(results_plus,results_minus)

def saveFrame(name,total):
    if total.dtype!='uint8':
        total = total/np.max(total)*255
        total = total.astype(np.uint8)
    cv2.imwrite(name,total)

def wavelet_denoising(im,wlt='sym2',lvl=5):
    coeffs_trous = pywt.swt2(im,wlt,lvl,start_level=0)
    
    total = np.ones(im.shape)
    #Add Gaussian blur
    for elts in coeffs_trous:
        cA,(cH,cV,cD) = elts
        var = getNoiseVar(cA)        
        cA = abe(cA,var)
        cA= np.arctan(cA/np.max(cA)*np.pi*3)
        #m.si(tata)
        total*=cA
    return total

def segmentation(total):
    t = skimage.filters.threshold_li(total)
    mask = (total>t).astype(np.uint8)
    #mask = cv2.dilate(mask,np.ones((4,4)),iterations = 1)
    kernel = np.ones((5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #m.fillHoles(mask)
    si(mask)
    mask,nr = ndi.label(mask)
    mask = filter_by_size(mask,500)
    return mask.astype(np.uint8)

def wavelet_denoising2(im,wlt='sym2',lvl=5,fraction=0.76):
    coeffs_trous = pywt.swt2(im,wlt,lvl,start_level=0)
    total = np.ones(im.shape)
    #Add Gaussian blur
    for elts in coeffs_trous:
        cA,(cH,cV,cD) = elts
        var = getNoiseVar(cA,fraction)        
        cA = abe(cA,var)
        #m.si(tata)
        total*=cA
    return total

def gaussian(size,sigma):
    """Generates a square gaussian mask with size*size pixels and std sigma"""
    a,b=np.ogrid[-size/2:size/2,-size/2:size/2]
    mask = a**2+b**2
    mask = np.exp(-mask.astype('float')/(2*float(sigma**2)))
    return mask