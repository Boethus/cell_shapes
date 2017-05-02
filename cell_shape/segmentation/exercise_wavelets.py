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

def si(img,title=None):
    plt.figure()
    plt.imshow(img)
    if title:
        plt.title(title)
    
def threshold(data,th):
    cA, (cH, cV, cD) = data
    cA[np.abs(cA)<th] = 0
    cH[np.abs(cH)<th] = 0
    cV[np.abs(cV)<th] = 0
    cD[np.abs(cD)<th] = 0
    return (cA, (cH, cV, cD))

def otsu_threshold(img):
    th = skimage.filters.threshold_otsu(img)
    out = (img>th).astype(np.uint8)
    return out

def diff(im,out):
    return np.sum(np.abs(im-out))/np.sum(np.abs(im))
    
def varInBg(patch):
    bp = patch[:50,:50]
    sigm = np.var(bp)
    return sigm

def positivePart(arr):
    arr[arr<0] = 0
    return arr

def getNoiseVar(img):
    """Gets the nth% of lower intensity pixels in an image correspondin to the noise
    n is determined empirically."""
    last_val = np.percentile(img,90)
    #si(img<last_val,title="Pixel values considered as noise")
    return np.var(img[img<last_val])

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

wlt = "db7"
cA, (cH, cV, cD) = pywt.dwt2(im,wlt)

sigmSqu = varInBg(cH)

l = map(varInBg,[cH,cV,cD,cA])
print "Variances for background patches: ",l
"""
bayesian_mask = np.abs(np.divide(cA**2-3*sigmSqu,cA))
cA[np.abs(cA)<bayesian_mask]=0
"""

t = 0.05
print "Current threshold:",t,"Otsu:",skimage.filters.threshold_otsu(cA)
ots = skimage.filters.threshold_otsu(cA)
cA[np.abs(cA)<ots]=0
   
otsH = skimage.filters.threshold_otsu(cH)
cH[np.abs(cH)<otsH]=0
print np.count_nonzero(cH)

otsV = skimage.filters.threshold_otsu(cV)
cV[np.abs(cV)<otsV]=0
print np.count_nonzero(cV)

otsD = skimage.filters.threshold_otsu(cD)
cD[np.abs(cD)<otsD]=0
print np.count_nonzero(cD)

print "thresholds: ", otsH,otsV,otsD,ots
"""
cH[:,:]=0
cV[:,:]=0
cD[:,:]=0"""

out = pywt.idwt2((cA, (cH, cV, cD)),wlt)

plt.figure()
plt.subplot(221)
plt.imshow(im)
plt.title("original image")

plt.subplot(222)
plt.imshow(out)
plt.title("Modified image")

plt.subplot(223)
plt.imshow(openFrame(10))
plt.title("Non processed image")

plt.subplot(224)
plt.imshow(otsu_threshold(out))
plt.title("thresholded version of the modified image")

diffe = np.sum(np.abs(im-out))/np.sum(np.abs(im))
print "Difference:", diffe
"""
plt.figure()
plt.hist(cV.reshape(-1),bins=300)
"""
  
#--------------------------------------------------------------#
#   Multi level decomposition filtering                        #
#--------------------------------------------------------------#

im = openFrame(10)
print im.dtype
im = skimage.filters.sobel(im)

def wavelet_filter(im):
    """Using a multi scale wavelet decomposition, filters the noise out of an image"""
    w = pywt.Wavelet(wlt)
    lvl = 2   #Decomposition level
    decomp = pywt.wavedec2(im, w,level = lvl)
    patch_size = 128
    cAn = decomp[0]
    (cHn, cVn, cDn) = decomp[1]
    image_stack=[]
    
    for i in range(len(decomp)-1):
        bg_img = decomp[i+1][1]
        cH,cV,cD = decomp[i+1]
        bp = bg_img[:patch_size/2**(lvl-i),:patch_size/2**(lvl-i)] #/!\bp should be calculated not on cA
        sigmaSq = np.var(bp)
        print sigmaSq
        bayesian_mask = np.divide(positivePart(cAn**2-3*sigmaSq),cAn)
        bayesian_mask[cAn==0]=0
        #cAn[cAn<bayesian_mask]=0
        cAn = bayesian_mask   
        #Thresholding the details as well
        bayesian_mask = np.abs(np.divide(positivePart(cH**2-3*sigmaSq),cH))
        cH[cH<bayesian_mask]=0
        bayesian_mask[cH==0]=0
        cH = bayesian_mask 
                     
        bayesian_mask = np.abs(np.divide(positivePart(cV**2-3*sigmaSq),cV))
        cV[cV<bayesian_mask]=0
        bayesian_mask[cV==0]=0
        cV = bayesian_mask 
        
        bayesian_mask = np.abs(np.divide(positivePart(cD**2-3*sigmaSq),cD))
        cD[cD<bayesian_mask]=0
        bayesian_mask[cD==0]=0
        cD = bayesian_mask 
        
        
        decomp[i+1] = (cH,cV,cD)
        
          
        if cAn.shape[0]>decomp[i+1][0].shape[0]:
            cAn = cAn[:decomp[i+1][0].shape[0],:]
        #cAn = pywt.idwt2((cAn,(np.zeros(cAn.shape),np.zeros(cAn.shape),np.zeros(cAn.shape))),wlt)
        cAn = pywt.idwt2((cAn,decomp[i+1]),wlt)
        image_stack.append(cAn)
    return cAn,image_stack

filtered ,stack = wavelet_filter(im)
si(filtered)

def correlationStack(stack):
    w = pywt.Wavelet(wlt)
    l = len(stack)
    for i in range(l-1):
        for j in range(i+1):
            details = np.zeros(stack[j].shape)
            img = pywt.idwt2((stack[j],(details,details,details)),w)
            stack[j]=img
    m0=stack[0].shape[0]
    m1=stack[0].shape[1]
    for elts in stack:
        m0 = min(elts.shape[0],m0)
        m1 = min(elts.shape[1],m1)
    total = np.zeros((m0,m1))
    for i in range(l):
        stack[i] = stack[i][:m0,:m1]
        total += stack[i]
    return stack,total

cs,total = correlationStack(stack)
si(total)
for i in cs:
    si(i)
    print i.shape
total = cs[-1]
t = otsu_threshold(total)
si(otsu_threshold(total),title="otsu threshold version of the summed stack")
#*-----------------------------------------#
#    Trying contour detection and filling
#------------------------------------------#
def postProcess(img):   
    out,contour,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    i=0
    for cnt in contour:
            cv2.drawContours(img,contour,i,255,-1)
            i+=1
    plt.figure()
    plt.imshow(img)
postProcess(t)

print "Variance calculated from background",getNoiseVar(im)
plt.colorbar()