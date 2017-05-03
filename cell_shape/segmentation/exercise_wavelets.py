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
import isingmrf_spmat as mrf
import scipy.ndimage as ndi
import pywt
import methods as m
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

def compareImages(im,out):
    """Plots 4 images : im being the edge detection version, out the denoised version, 
    and also plots the thresholded version of out an the original image"""
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

im = openFrame(10)
im = skimage.filters.sobel(im)


wlt = 'coif1'
#db11 looks ok
cA, (cH, cV, cD) = pywt.dwt2(im,wlt)

sigmSqu = varInBg(cH)

l = map(varInBg,[cH,cV,cD,cA])
print "Variances for background patches: ",l


t = 0.05
print "Current threshold:",t,"Otsu:",skimage.filters.threshold_otsu(cA)
ots = skimage.filters.threshold_otsu(cA)
cA[np.abs(cA)<ots]=0
print np.count_nonzero(cA)
   
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
si(out,"Filtering with OTSU")
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
im = skimage.filters.sobel(im)


def getNoiseVar(img):
    """Gets the nth% of lower intensity pixels in an image correspondin to the noise
    n is determined empirically."""
    last_val = np.percentile(img,95)
    #si(img<last_val,title="Pixel values considered as noise")
    return np.var(img[img<last_val])

def abe(img,variance):
    """proceeds to the Amplitude-scale invariant Bayes Estimation (ABE)"""
    nominator = img**2-3*variance
    nominator[nominator<0] = 0
    out = np.divide(nominator,img)
    out[img==0]=0
    return out

def wavelet_filter(im):
    """Using a multi scale wavelet decomposition, filters the noise out of an image"""
    w = pywt.Wavelet(wlt)
    lvl = 2   #Decomposition level
    decomp = pywt.wavedec2(im, w,level = lvl)
    variance = getNoiseVar(im)
    print "variance:",variance
    ABE = lambda image: abe(image,variance)
    
    decomp[0] = ABE(decomp[0])
    
    for i in range(1,len(decomp)):
        decomp[i] = map(ABE,decomp[i])
    out = pywt.waverec2(decomp,w)
    return out

def hardTh(img):
    otsu = skimage.filters.threshold_otsu(img)
    out = pywt.threshold(img,otsu,mode='hard')
    return out

def wavelet_filter_ht(im):
    """Multiscale hard thresholding with OTSU"""
    w = pywt.Wavelet(wlt)
    lvl = 2   #Decomposition level
    decomp = pywt.wavedec2(im, w,level = lvl)
    decomp[0] = hardTh(decomp[0])
    for i in range(1,len(decomp)):
        decomp[i] = map(hardTh,decomp[i])
    out = pywt.waverec2(decomp,w)
    return out

filtered= wavelet_filter(im)

si(filtered,"After level n wavelet filter")
n_iter = 30
for i in range(n_iter):
    filtered = wavelet_filter(filtered)
si(filtered,"image filtered "+str(n_iter)+ " times")
compareImages(im,filtered)
otsu_filtered = otsu_threshold(filtered)
otsu_multi = wavelet_filter_ht(im)
si(otsu_multi,"Otsu thresholding with multipass")
compareImages(im,otsu_multi)

#*-----------------------------------------#
#    Trying contour detection and filling
#------------------------------------------#
def postProcess(img):   
    #img = otsu_threshold(img)
    out,contour,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    i=0
    for cnt in contour:
            cv2.drawContours(img,contour,i,255,-1)
            i+=1
    plt.figure()
    plt.imshow(img)
    return img

cA, (cH, cV, cD) = pywt.dwt2(im,wlt)
cA2, (cH2, cV2, cD2) = pywt.dwt2(cA,wlt)
v = getNoiseVar(cA2)

oo = abe(cA2,v)
for i in range(5):
    oo=abe(oo,v)

def filter_n_chill(img,sigma):
    from skimage.filters import try_all_threshold
    
    exp = ndi.gaussian_filter(img,sigma)
    try_all_threshold(exp)

#----------Final thresholding-----------#
filter_n_chill(filtered,2)
exp = ndi.gaussian_filter(filtered,1)
th_li = skimage.filters.thresholding.threshold_li(exp)
exp = (exp>th_li).astype(np.uint8)
out = postProcess(exp)

labels,nr = ndi.label(out)
numbers = np.zeros(nr)
for i in range(1,nr+1):
    numbers[i-1] = np.count_nonzero(labels==i)
plt.figure()
plt.plot(numbers)
plt.title("number of pixels per label")

final_image = m.filter_by_size(labels,500)
compareImages(im,final_image)