#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:56:25 2017

@author: aurelien
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy import ndimage as ndi

from skimage import color
import scipy.ndimage as ndimage

from skimage.filters import threshold_otsu
import skimage.filters

import scipy.misc
import time

import segmentationFunctions as sf

#Manually written scripts
import ftracker
import displayFunctions as df
                
t = time.clock()
"""1st pass"""

print "1st pass"
filename = os.path.join("data",'itoh-cell-migration-02.mov')

cap = cv2.VideoCapture(filename)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
save=True
filename='video1/'
saveLabels = 0
labels_list=[[0],[0]]
total_buffer= np.zeros((height-30,width,length),dtype=np.uint8)
for i in range(length):
    #Get frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=gray[30:,:]
    #Definition of the different quantities used for temporal analysis
    if i==0:
        images_buffer = np.zeros((gray.shape[0],gray.shape[1],2))
        gray_buffer = np.zeros((gray.shape[0],gray.shape[1],2),dtype=np.uint8) # remembers i-1 and i-2
        centroids_buffer = [0,0]   #List with two entries
        #total_buffer = np.zeros((gray.shape[0],gray.shape[1],length),dtype=np.uint8)
        index_disappeared = [] 
        #List3d tuples (i,size_diff,indexes) with the iteration number, the difference between number of cells found
        #between two frames and the index of the cell which appeared or disappeared.  
    #thresholding
    thresh = ftracker.w_thresh(gray)
    gray_buffer[:,:,i%2] = gray
    #Segmentation
    labels_new, nlabs = ndi.label(thresh)
    labels_new_filtered = ftracker.filter_by_size(labels_new)
    
    #Misc. assigmnments
    images_buffer[:,:,i%2] = labels_new_filtered  
    total_buffer[:,:,i] = images_buffer[:,:,i%2]
    labels_list[i%2] = range(1,int(np.amax(labels_new_filtered))+1)
    centroids_buffer[i%2] = ftracker.centroids2(labels_new_filtered,labels_list[i%2])
    
    if i==15:
        saveLabels = np.copy(images_buffer)
    #Hungarian algorithm
    if i>0:
        images_buffer[:,:,i%2] = ftracker.w_hungarian(centroids_buffer,i,index_disappeared,images_buffer,labels_list)
        total_buffer[:,:,i] = images_buffer[:,:,i%2]
        
    gray_contours = df.drawContours(gray_buffer[:,:,i%2],images_buffer[:,:,i%2]==3)
    xc,yc = centroids_buffer[i%2]
    df.drawCentroids(gray_contours,xc,yc)
        
cap.release()
filename = 'itoh-cell-migration-02.mov'
df.showSegmentation_color(filename,total_buffer,save=True,savename='video1/')

"""2nd pass"""
print "2nd pass"
tota_buf = np.copy(total_buffer)
to_d,list_of_is = ftracker.filter_out_flickers(tota_buf,index_disappeared)
buff = ftracker.remove_to_destroy(tota_buf,to_d)
filename = 'itoh-cell-migration-02.mov'

rem_events = ftracker.get_remaining_events(index_disappeared,to_d)

"""Defuse the fused people"""
fusionEvents=[]
divisionEvents = []
for elts in rem_events:
    boole,label = ftracker.isFusion(elts,total_buffer)
    boole2,label2 = ftracker.isDivision(elts,total_buffer)
    if boole:
        fusionEvents.append((elts,label))
    if boole2:
        divisionEvents.append((elts,label2))

buffe = np.copy(buff)
ftracker.bindEvents(fusionEvents,divisionEvents,buffe)

#showSegmentation_color(filename,buffe,save=True,savename='video_color/')

fusionEvents=[]
divisionEvents = []
for elts in rem_events:
    boole,label = ftracker.isFusion(elts,total_buffer)
    boole2,label2 = ftracker.isDivision(elts,total_buffer)
    if boole:
        fusionEvents.append((elts,label))
    if boole2:
        divisionEvents.append((elts,label2))

buffe = np.copy(buff)
ftracker.bindEvents(fusionEvents,divisionEvents,buffe)

df.showSegmentation_color(filename,buffe,save=True,savename='video2')

print "total time", time.clock()-t