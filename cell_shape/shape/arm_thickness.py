#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:31:44 2017

@author: aurelien
"""
import methods as m
import numpy as np
import cv2
from skimage.morphology import disk

def arms_thickness(image):
    """same as distance transform"""
    white = (image>0).astype(np.uint8)   #A white image is just a binary image
    kernel = np.ones((3,3),dtype = np.uint8)
    kernel = disk(2)
    thickness = white.copy()
    while(np.count_nonzero(white)>0):
        white = cv2.erode(white, kernel, iterations = 1)
        thickness+=white
    return thickness

def erosion_stability(image):
    kernel = disk(3)
    stability = image.copy()
    i=1
    while(np.count_nonzero(image)>0):
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=i)
        stability+=image
        i+=1
        print i
    print "stability done in",i,"iterations"
    return stability

def get_thickness_list(path,nr):
    arms = m.open_frame(path,nr)
    thick = cv2.distanceTransform((arms>0).astype(np.uint8),cv2.DIST_L2,3)
    thickness_list=[]
    for i in range(np.max(arms)):
        thickness = np.max(thick[arms==i+1])
        thickness_list.append(thickness)
    return thickness_list

