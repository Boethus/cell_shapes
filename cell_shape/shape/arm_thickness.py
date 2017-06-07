#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:31:44 2017

@author: aurelien
"""
import methods as m
import os
import numpy as np
import cv2
from skimage.morphology import disk
import matplotlib.pyplot as plt

plt.close('all')

path = os.path.join("..",'data','microglia','8_denoised')
path_centers = os.path.join("..",'data','microglia','8_centers') 
path_arms = os.path.join("..",'data','microglia','8_arms')    

nr=  5 
image = m.open_frame(path_centers,nr)
arms = m.open_frame(path_arms,nr)
image_denoised = m.open_frame(path,nr)
total_image = (m.open_frame(path_arms,nr)>0).astype(np.uint8)+(m.open_frame(path_centers,nr)>0).astype(np.uint8)

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

thick = arms_thickness(arms)
m.si2(arms,thick,"white image","corresponding thickness")

stability = erosion_stability(total_image)
m.si2(stability,image_denoised,"stability to erosion")
kernel = disk(3)
opening = cv2.morphologyEx(total_image, cv2.MORPH_OPEN, kernel,iterations=2)*255

overlay_opening = m.cv_overlay_mask2image(opening,image_denoised,color="green")
overlay_current = m.cv_overlay_mask2image((image>0).astype(np.uint8)*255,image_denoised,color="green")
m.si2(overlay_opening,overlay_current)

thick = thick.astype(np.float)
thick = cv2.distanceTransform((arms>0).astype(np.uint8),cv2.DIST_L2,3)
thick2 = thick.copy()
#Measure actual thickness:
for i in range(np.max(arms)):
    thickness = np.max(thick[arms==i+1])
    thick2[arms==i+1] = np.mean(thick[arms==i+1])
    thick[arms==i+1] = thickness
m.si2(thick,thick2)

def get_thickness_list(path,nr):
    arms = m.open_frame(path,nr)
    thick = cv2.distanceTransform((arms>0).astype(np.uint8),cv2.DIST_L2,3)
    thickness_list=[]
    for i in range(np.max(arms)):
        thickness = np.max(thick[arms==i+1])
        thickness_list.append(thickness)
    return thickness_list

l=get_thickness_list(path_arms,5)
plt.figure()
plt.hist(l,bins=20)
