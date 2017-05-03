# -*- coding: utf-8 -*-
"""
Created on Wed May 03 11:11:06 2017

@author: univ4208
"""

import methods as m

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

im = m.openFrame(10)

wlt = pywt.Wavelet("rbio3.1")
#wlt = pywt.Wavelet("haar")
"""
for i in range(1,6):
    filtered = wavelet_edge(im,wlt,lvl=i)
    m.si(filtered,"Decomposition lvl = "+str(i))"""

#A trous method
im = im[:-2,:]  #To change max level
#im = skimage.filters.sobel(im)

print pywt.swt_max_level(im.shape[1]),pywt.swt_max_level(im.shape[0])
coeffs_trous = pywt.swt2(im,wlt,5,start_level=0)

total = np.ones(im.shape)

for elts in coeffs_trous:
    cA,(cH,cV,cD) = elts
    tata = np.sqrt(cH**2+cV**2)
    var = m.getNoiseVar(cA)
    #cA=m.abe(cA,var)
    m.si(cA)
    total*=cA
m.si(total)