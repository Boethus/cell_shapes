# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:49:02 2017

@author: univ4208
"""
import os
import cv2
import numpy as np
import ftracker
import scipy.ndimage as ndi
import displayFunctions as df
import matplotlib.pyplot as plt

filename = os.path.join("data",'itoh-cell-migration-02.mov')

cap = cv2.VideoCapture(filename)

nframe = 15

save=True
filename='video1/'
saveLabels = 0
for i in range(90):
    #Get frame
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame[30:,:,0]
    cv2.imwrite(os.path.join("data","yoshi_mov_2",str(i)+".tif"),gray)
    """With only blue channel:"""
    """ gray = frame[:,:,0]
    gray=gray[30:,:]
    thresh = ftracker.w_thresh(gray)
    #Segmentation
    labels_new, nlabs = ndi.label(thresh)
    labels_new_filtered = ftracker.filter_by_size(labels_new)
    
    f1 = np.copy(frame)
    f1=f1[30:,:]
    f1=df.drawContours(f1,labels_new_filtered)
    
    #With all channels
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=gray[30:,:]
    thresh = ftracker.w_thresh(gray)
    #Segmentation
    labels_new, nlabs = ndi.label(thresh)
    labels_new_filtered = ftracker.filter_by_size(labels_new)
    f2 = np.copy(frame)
    f2 = f2[30:,:]
    f2 = df.drawContours(f2,labels_new_filtered,color=1)
    
    plt.figure()
    plt.title("frame number"+str(i))
    plt.subplot(121)
    plt.imshow(f1)
    plt.title("segmentation with all channels")
    
    plt.subplot(122)
    plt.imshow(f2)
    plt.title("segmentation only blue")"""
    
cap.release()