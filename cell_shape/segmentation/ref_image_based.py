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


def modulusMaximum(amplitude,angle):
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

def wavelet_edge(im,wlt,lvl=2):

    sigma =0.5*lvl
    decomposition = pywt.wavedec2(im, wlt,level = lvl)
    (cH,cV,cD) = decomposition[1]
    cH= skimage.filters.gaussian(cH,sigma)
    cV = skimage.filters.gaussian(cV,sigma)
    
    amplitude = np.sqrt(cH**2+cV**2)
    angle = np.arctan2(cH,cV)
    maxMod = modulusMaximum(amplitude,angle)
    amplitude[maxMod] = 0
    return amplitude

def saveFrame(name,total):
    total = total/np.max(total)*255
    total = total.astype(np.uint8)
    cv2.imwrite(name,total)
    
im = m.openFrame(67)

wlt = pywt.Wavelet("rbio3.1")
#wlt = pywt.Wavelet("haar")
"""
for i in range(1,6):
    filtered = wavelet_edge(im,wlt,lvl=i)
    m.si(filtered,"Decomposition lvl = "+str(i))"""

#A trous method
im = im[:-2,:]  #To change max level
im = skimage.filters.sobel(im)

print pywt.swt_max_level(im.shape[1]),pywt.swt_max_level(im.shape[0])
coeffs_trous = pywt.swt2(im,wlt,2,start_level=0)

total = np.ones(im.shape)
#Add Gaussian blur
for elts in coeffs_trous:
    cA,(cH,cV,cD) = elts
    tata = np.sqrt(cH**2+cV**2)
    var = m.getNoiseVar(cA)
    #m.si(cA,"unfiltered")
    #plt.figure()
    #plt.hist(cA.reshape(-1),bins=200)
    #cA=np.power(m.abe(cA,var),0.5)
    #m.si(cA,"filtered")
    for i in range(1):
        cA = m.abe(cA,var)
    cA= np.arctan(cA/np.max(cA)*np.pi*1.5)
    #m.si(tata)
    total*=cA
m.si(total)
from skimage.filters import try_all_threshold
try_all_threshold(total)

def segmentation(total):
    t = skimage.filters.threshold_li(total)
    mask = (total>t).astype(np.uint8)
    #mask = cv2.dilate(mask,np.ones((4,4)),iterations = 1)
    kernel = np.ones((7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #m.fillHoles(mask)
    m.si(mask)
    mask,nr = ndi.label(mask)
    mask = m.filter_by_size(mask,500)
    return mask.astype(np.uint8)

img_f = segmentation(skimage.filters.gaussian(total,0.01))
img_g = segmentation(skimage.filters.gaussian(total,0.5))

plt.figure()
plt.subplot(121)
plt.imshow(img_f)
plt.subplot(122)
plt.imshow(img_g)


save = False
if save:
    total = total/np.max(total)*255
    total = total.astype(np.uint8)
    cv2.imwrite("stack.tif",total)