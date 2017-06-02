# -*- coding: utf-8 -*-
"""
Created on Fri Jun 02 15:00:32 2017

@author: univ4208 aka bobotte ma gueule eh ouais tu vas faire quoi
"""

#This script intends to compute the portion of microglia "overlapping" chunks of neuron
#in an experiment

import methods as m
import os
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import skimage.filters
import matplotlib.pyplot as plt
import cv2
plt.close('all')

def open_phase(path,fr_nb):
    name = str(fr_nb).zfill(3)
    phase_path = os.path.join(path,"Scene1Interval"+name+"_PHASE.png")
    
    phase= Image.open(phase_path)
    phase = np.asarray(phase)
    return phase

#Experiment number
exp = 1
path = os.path.join("..",'data','microglia','Beacon-'+str(exp)+' unst')
path_fluo = os.path.join("..",'data','microglia','RFP'+str(exp)+'_denoised')

frame=2
ratios = np.zeros(241)

phase = open_phase(path,100)
inv = np.max(phase)-phase
img = skimage.filters.gaussian(inv,4)
t= skimage.filters.threshold_otsu(img)
chunks = img>t
m.si2(phase,chunks)
for frame in range(1,242):
    print "processing frame",frame

    fluo_frame = m.open_frame(path_fluo,frame)
    thresh_fluo = m.hysteresis_thresholding(fluo_frame,6,10)
    
    projection = chunks*thresh_fluo
    ratio = float(np.count_nonzero(projection))/float(np.count_nonzero(thresh_fluo))
    ratios[frame-1] = ratio
plt.figure()
plt.plot(ratios)
plt.title("Ratio of microglia above big chunks of neurons")
plt.xlabel("frame")
plt.ylabel("ratio")
np.save("ref100_ratios_exp"+str(exp),ratios)