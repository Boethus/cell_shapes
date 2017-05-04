# -*- coding: utf-8 -*-
"""
Created on Thu May 04 12:29:57 2017

@author: univ4208
"""

import methods as m

import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import skimage.feature
import skimage.filters
import cv2
import scipy.ndimage as ndi
import os
from PIL import Image
from skimage.filters import try_all_threshold

def gf(img,sigma):
    img2 = skimage.filters.gaussian(img,sigma)
    return img2

plt.close('all')

frame_number=230
#Opening desired image
name = os.path.join("..",'data','microglia','RFP1','Scene1Interval'+str(frame_number)+'_RFP.png')
img = Image.open(name)
img = np.asarray(img)
filtered = m.wavelet_denoising(img,lvl=2)
#filtered = filtered/np.max(filtered)
print "max filtered image",np.max(filtered)

"""
try_all_threshold(filtered)"""

t=skimage.filters.threshold_adaptive(filtered,301)

t2=skimage.filters.threshold_otsu(filtered)
otsu_bin = filtered>t2
m.si2(t,otsu_bin,"adaptive thresh","otsu")

#adapt the intensity of each cluster
labels,nb = ndi.label(otsu_bin)
for i in range(nb):
    filtered[labels==i+1] /= (np.max(filtered[labels==i+1]))
    
m.si2(labels,filtered,'labels','filtered image')
#filtered[~otsu_bin]=0

total = np.ones(img.shape)
total_sum = np.zeros(img.shape)
total_maxwise = np.zeros(img.shape)
for i in range(10,20):
    out = gf(filtered,i)
    total*=out
    total_sum=+out
    total_maxwise[out>total_maxwise] = out[out>total_maxwise]
    
m.si(total,"product of stuff")
m.si(total_maxwise,"Maxima")
print np.count_nonzero(total)

#Displays data differently
values = total[total>0]
niou_array = np.zeros(total.shape)

for i in range(255):
    per = np.percentile(values,float(i+1)/2.55)
    niou_array[total>per]+=1
m.si(niou_array)

def redistribute_per_percentile(total):
    values = total[total>0]
    niou_array = np.zeros(total.shape)

    for i in range(255):
        per = np.percentile(values,float(i+1)/2.55)
        niou_array[total>per]+=1

    return niou_array

maxwise = redistribute_per_percentile(total_maxwise)
m.si(maxwise,'Maxima elemt-wise')

###################################################################
## Find the elements which have a high 'total' value corresponding to the out of focus stuff
## Inch Allah
#######################################################################

mask = total>skimage.filters.threshold_li(total)
try_all_threshold(total)
m.si(mask,"mask of the elements to remove")
labels_to_remove = labels[mask]
labels_to_remove = np.unique(labels_to_remove)

filtered_array = np.copy(filtered)
for lab in labels_to_remove:
    filtered_array[labels==lab]=0

print "Nb elements supprimes: ", labels_to_remove.size
m.si2(img,filtered_array,"Original image","Degaussianed image")