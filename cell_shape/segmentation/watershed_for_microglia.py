# -*- coding: utf-8 -*-
"""
Created on Thu May 04 16:48:00 2017

@author: univ4208
"""

"""Aims at improving the segmentation of microglia images using watershed algorithm"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os 
import methods as m
from PIL import Image
import skimage.filters
import scipy.ndimage as ndi
import skimage.morphology
from skimage.morphology import watershed, disk
from skimage.filters import rank

plt.close('all')

frameNum = 130
path = os.path.join("..",'data','microglia','RFP1_denoised','filtered_Scene1Interval'+str(frameNum)+'_RFP.png')
path = os.path.join("..",'data','microglia','RFP1','Scene1Interval'+str(frameNum)+'_RFP.png')

img = Image.open(path)

#Filtering
gray = np.asarray(img)

gray = gray.astype(np.float)
gray=np.arctan(gray/np.max(gray)*np.pi*3)
gray = gray/np.max(gray)
gray = rank.median(gray, disk(2))

m.si(gray,'original image')

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

t = skimage.filters.threshold_local(gray,501)
t = gray>t
#t = (rank.median(t, disk(2))).astype(np.bool)

m.si(t)
test=np.copy(gray)
test[~t]=0
test2=np.copy(gray)
test2[thresh==0]=0
m.si2(test,test2,"Image thresholded with adaptive","image thresholded with otsu")

kernel = np.ones((3,3),np.uint8)

#Carry on with adaptive threshold
thresh = t.astype(np.uint8)*255

opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
m.si2(thresh,opening,"otsu","opening otsu")



sure_bg = cv2.dilate(thresh,kernel,iterations=4)
tmp = np.copy(gray)
tmp[sure_bg==255] = 0
m.si2(sure_bg,tmp,"sure background","Only background normally")

dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,5)
m.si2(gray,dist_transform,"original image","distance transform")

ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)

m.si(sure_fg)

lab,nr = ndi.label(sure_fg)
print nr

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
#m.si(unknown,'unknown')

markers,nr = ndi.label(sure_fg)
markers+=1

markers[unknown==255]=0

markers = markers.astype(np.int32)
#gray = 255-gray

gradient = skimage.filters.sobel(gray)
gradient = gradient/np.max(gradient)*255
gradient = gradient.astype(np.uint8)
gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
gray_converted = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
labels = cv2.watershed(gradient,markers)
m.si(labels,"watershed with gradient")
markers = cv2.watershed(gray_converted,markers)

m.si(markers)
gray_converted[markers==-1]=[255,0,0]
m.si(gray_converted,"Image segmented with contours maggle")
######################################################



image=255-gray

denoised = rank.median(image, disk(2))

# find continuous region (low gradient -
# where less than 10 for this image) --> markers
# disk(5) is used here to get a more smooth image
markers = rank.gradient(denoised, disk(5)) < 10
markers = ndi.label(markers)[0]

# local gradient (disk(2) is used to keep edges thin)
gradient = rank.gradient(denoised, disk(2))
gradient = skimage.filters.sobel(denoised)
# process the watershed
labels = watershed(gradient, markers)

# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title("Original")

ax[1].imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
ax[1].set_title("Local Gradient")

ax[2].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
ax[2].set_title("Markers")

ax[3].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[3].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
ax[3].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()