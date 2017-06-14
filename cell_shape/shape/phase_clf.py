# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:14:24 2017

@author: univ4208
"""

import sys
import os
sys.path.append(os.path.join(".","..","segmentation"))
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.filters
import methods as m
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
phase = open_phase(path,5)

sob_h = skimage.filters.sobel_h(phase)
sob_v = skimage.filters.sobel_v(phase)
angle = np.arctan2(sob_h,sob_v)

m.si2(phase,angle)