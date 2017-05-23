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
import skimage.filters as f
import scipy.ndimage as ndi
import time
from skimage.filters import try_all_threshold
import ftracker

plt.close('all')

def proba_map_gaussian(img):
    """Optimal template matching to find a gaussian with std sigma in img"""
    list_of_sigmas = [40,30,20]
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

def custom_threshold(im):
    th1=im>skimage.filters.threshold_niblack(im,55,-0.5)
    th1=np.logical_and(th1,im>skimage.filters.threshold_otsu(im))
    #m.si(th1,"1st threshold before morphology")
    kernel = np.ones((5,5),np.uint8)
    opening1 = cv2.morphologyEx(th1.astype(np.uint8), cv2.MORPH_OPEN, kernel,iterations=2)
    #m.si(opening1,"After opening")
    opening1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel,iterations=2)
    return opening1

def find_cells(frameNum,sizeFilter=150,cinema=False):
    frameNum=str(frameNum)
    nb = '0000'
    nb=nb[0:4-len(frameNum)]+frameNum
    path = os.path.join("..",'data','microglia','RFP1_cropped','RFP1_denoised'+nb+'.png')
    img = Image.open(path)
    im = np.asarray(img)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(30,30))
    im = clahe.apply(im)
    #thresh_hard = custom_threshold(im)
    thresh_hard = (im>f.threshold_li(im)).astype(np.uint8)
    if sizeFilter>0:
        kernel = np.ones((5,5),np.uint8)
        thresh_hard = cv2.morphologyEx(thresh_hard.astype(np.uint8), cv2.MORPH_OPEN, kernel,iterations=1)
        
    labels,nr = ndi.label(thresh_hard)
    if sizeFilter>0:
        labels = m.filter_by_size(labels,sizeFilter)
        
    if cinema:
        im.setflags(write=1)
        im[labels==0]=0
        cv2.imshow("Frame",im)
        cv2.waitKey(60)
    return labels

def openImage(frameNum):
    frameNum=str(frameNum)
    nb = '0000'
    nb=nb[0:4-len(frameNum)]+frameNum
    path = os.path.join("..",'data','microglia','RFP1_cropped','RFP1_denoised'+nb+'.png')
    img = Image.open(path)
    im = np.asarray(img)
    return im

class Cell(object):
    def __init__(self,nb,n_frames):
        self.number=nb
        self.n_frames = n_frames
        self.list_of_ids = []
        self.gaussian_coeffs=[]
        
def movie():
    points=[]
    for i in range(150):
        n=find_cells(i,True)
        points.append(np.max(n))
    cv2.destroyAllWindows()
    plt.figure()
    plt.plot(points)
    plt.title("Number of elements detected")

plt.close('all')
frameNum = 32
frameNum=str(frameNum)
nb = '0000'
nb=nb[0:4-len(frameNum)]+frameNum
path = os.path.join("..",'data','microglia','RFP1_cropped','RFP1_denoised'+nb+'.png')
img = Image.open(path)
im = np.asarray(img)


clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(30,30))
im = clahe.apply(im)
m.si(im,"Local histo equalization on image")
max_map,coeffs_map=proba_map_gaussian(im)
print np.max(max_map)
plt.figure()
plt.subplot(131)
plt.imshow(im)
plt.subplot(132)
plt.imshow(max_map)
plt.subplot(133)
plt.imshow(coeffs_map)

elt = find_cells(frameNum)
print np.max(elt)
#movie()

img_underlined = im.copy()
img_underlined[elt>0]=255
m.si(img_underlined,"image underlined")

def overlay(img,thresh):
    out = img.copy()
    if thresh.dtype==bool:
        out[~thresh]=0
    else:
        out[thresh==0]=0
    return out

img_underlined2 = im.copy()
th_li = f.threshold_li(im)
img_underlined2[im<th_li]=0
tli = im>th_li
lab,nr = ndi.label(tli)
lab=m.filter_by_size(lab,20)
nr=np.max(lab)
m.si2(img_underlined2,lab,"Threshold li","labeled "+str(nr)+" components")
opli = cv2.morphologyEx(tli.astype(np.uint8), cv2.MORPH_OPEN, kernel,iterations=2)
m.si2(im,overlay(im,opli),"original","pverlaid")

