# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:11:33 2017

@author: univ4208
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:01:34 2017

@author: univ4208
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.feature
import skimage.filters
from sklearn.preprocessing import normalize
import cv2

import pywt
plt.close("all")
def openFrame(number):
    name = os.path.join("..","data","yoshi_mov_2",str(number)+".tif")
    img = Image.open(name)
    img = np.asarray(img)
    return img

def si(img):
    plt.figure()
    plt.imshow(img)
    
def threshold(data,th):
    cA, (cH, cV, cD) = data
    cA[np.abs(cA)<th] = 0
    cH[np.abs(cH)<th] = 0
    cV[np.abs(cV)<th] = 0
    cD[np.abs(cD)<th] = 0
    return (cA, (cH, cV, cD))

def varInBg(patch):
    bp = patch[:50,:50]
    sigm = np.var(bp)
    return sigm

im = openFrame(10)
print im.dtype
im = skimage.filters.sobel(im)
"""im*=255.0/np.amax(im)
im = im.astype(np.uint8)
print im.dtype"""
#im = cv2.equalizeHist(im)   #Try histogram equalization?
si(im)


"""
bp = im[100:,100:]
sigm = np.var(bp)"""

wlt = "db1"
cA, (cH, cV, cD) = pywt.dwt2(im,wlt)
"""
sigmSqu = varInBg(cH)

l = map(varInBg,[cH,cV,cD])
print "Variances for background patches: ",l

bayesian_mask = np.abs(np.divide(cA**2-3*sigmSqu,cA))
cA[np.abs(cA)<bayesian_mask]=0
"""

t = 0.05
print "Current threshold:",t,"Otsu:",skimage.filters.threshold_otsu(cA)
cA[np.abs(cA)<t]=0


cH[:,:]=0
cV[:,:]=0
cD[:,:]=0

out = pywt.idwt2((cA, (cH, cV, cD)),wlt)



plt.figure()
plt.subplot(121)
plt.imshow(im)
plt.title("original image")

plt.subplot(122)
plt.imshow(out)
plt.title("Modified image")

diff = np.sum(np.abs(im-out))/np.sum(np.abs(im))
print "Difference:", diff
"""
plt.figure()
plt.hist(cV.reshape(-1),bins=300)
"""

#*----------------------#
#    Trying contour detection and filling
#------------------------------------------#

th = skimage.filters.threshold_otsu(out)
img = (out>th).astype(np.uint8)


out,contour,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#Step 3
i=0
im = np.zeros(im.shape)
for cnt in contour:
        cv2.drawContours(img,contour,i,255,-1)
        i+=1
plt.figure()
plt.imshow(out)