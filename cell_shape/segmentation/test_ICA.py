# -*- coding: utf-8 -*-
"""
Created on Mon May 08 14:11:09 2017

@author: univ4208
"""

print(__doc__)

# Authors: Vlad Niculae, Alexandre Gramfort
# License: BSD 3 clause

import logging
from time import time

from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition
import numpy as np
import os
from PIL import Image
import cv2

frameNum = 130
path = os.path.join("..",'data','microglia','RFP1_denoised','filtered_Scene1Interval'+str(frameNum)+'_RFP.png')
path = os.path.join("..",'data','microglia','RFP1','Scene1Interval'+str(frameNum)+'_RFP.png')

img = Image.open(path)

im = np.asarray(img)
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(30,30))
cl1 = clahe.apply(im)

cl1_centered = cl1-np.mean(cl1)

cl1_centered = cl1_centered.reshape(-1,1)

clf = decomposition.PCA()

clf.fit(cl1_centered)