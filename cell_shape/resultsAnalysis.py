#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:08:39 2017

@author: aurelien
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def centroid(img,label):
    out = np.zeros(2)
    pos_list = np.where(img==label)
    out[0] = np.mean(pos_list[0])
    out[1] = np.mean(pos_list[1])
    return out

filename = "labeledSegments/"

img = cv2.imread(filename+"frame"+str(89)+".png")
n = np.amax(img)
images = np.zeros((img.shape[0],img.shape[1],3))

speeds = np.zeros((90,n))
ratios = np.zeros((90,n))

for i in range(90):
    img = cv2.imread(filename+"frame"+str(i)+".png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images[:,:,i%3] = img
    for label in range(np.amax(img)):
        if np.count_nonzero(images[:,:,(i-1)%3]==label+1)>0 and i>1:
            speed = np.sqrt(np.sum((centroid(images[:,:,(i-1)%3],label)-centroid(images[:,:,(i-2)%3],label))**2))
            speed += np.sqrt(np.sum((centroid(images[:,:,(i-1)%3],label)-centroid(images[:,:,(i)%3],label))**2))    
            speeds[i,label] = speed/2
            
            p=PCA()
            tmp = np.where(images[:,:,(i-1)%3]==label+1)
            to_pcaer = np.zeros((len(tmp[0]),2))
            to_pcaer[:,0] = tmp[0]
            to_pcaer[:,1] = tmp[1]
            p.fit(to_pcaer)
            var = p.explained_variance_ratio_
            
            ratios[i,label] = var[1]/var[0]


plt.figure()
plt.scatter(speeds.reshape(-1),ratios.reshape(-1))
plt.xlabel("speeds")
plt.ylabel("ratios")