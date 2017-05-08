# -*- coding: utf-8 -*-
"""
Created on Fri May 05 13:15:08 2017

@author: univ4208
"""
import pywt
import matplotlib.pyplot as plt
import methods as m
import os
from PIL import Image
import numpy as np
import cv2
import skimage.filters
from skimage.morphology import watershed, disk
from skimage.filters import rank
import scipy.signal
"""Attempt to remove gaussian spots using wavelets"""
plt.close('all')

def display6Wlts(coeff_list,coeff=0):
    plt.figure()
    i=1
    for elt in coeff_list:
        cA,tup = elt
        if coeff:
            cA = tup[coeff-1]
            cA = np.sqrt(tup[0]**2+tup[1]**2)
        plt.subplot(3,2,i)
        i+=1
        plt.imshow(cA)
        plt.title("Coeff "+str(i))

#Technique: Laplaian of the gaussian
def blobDet(img,sigma=10):
    blob = np.asarray(img)
    blob = skimage.filters.gaussian(blob,sigma)
    blob = skimage.filters.laplace(blob,ksize=3)
    return blob

#Filtering
def process(im):
    """Combines Laplace and wavelet filtering to set alight the signal and 
    remove the annoying gaussian spots"""
    im=skimage.filters.laplace(im,ksize=3)
    #im = blobDet(im,3)
    #im = cv2.equalizeHist(im)
    wlt = 'dmey'
    lvl=6
    coeffs_trous = pywt.swt2(im,wlt,lvl,start_level=0)
    total = np.ones(im.shape)
    for i in coeffs_trous:
        coeff,osef=i
        total+=(coeff)
    return total

frameNum = 130
path = os.path.join("..",'data','microglia','RFP1_denoised','filtered_Scene1Interval'+str(frameNum)+'_RFP.png')
path = os.path.join("..",'data','microglia','RFP1','Scene1Interval'+str(frameNum)+'_RFP.png')

img = Image.open(path)

im = np.asarray(img)

def wavelet_denoising2(im,wlt='sym2',lvl=5,fraction=0.76):
    coeffs_trous = pywt.swt2(im,wlt,lvl,start_level=0)
    total = np.ones(im.shape)
    #Add Gaussian blur
    for elts in coeffs_trous:
        cA,(cH,cV,cD) = elts
        var = m.getNoiseVar(cA,fraction)        
        cA = m.abe(cA,var)
        #m.si(tata)
        total*=cA
    return total


clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(30,30))
cl1 = clahe.apply(im)

cl2=clahe.apply(cl1)
#cl2 = cv2.equalizeHist(cl1)
cl2 = wavelet_denoising2(cl2,lvl=3)
cl3=cl2/np.max(cl2)*255

cl3 = clahe.apply(cl3.astype(np.uint8))
#cl2 = rank.median(cl2, disk(5))


#Comparison between different methods
path = os.path.join("..",'data','microglia','RFP1_denoised','filtered_Scene1Interval'+str(frameNum)+'_RFP.png')

imge = Image.open(path)

ime = np.asarray(imge)
plt.figure()
plt.subplot(221)
plt.imshow(im)
plt.title("original image")
plt.subplot(222)
plt.imshow(ime)
plt.title("global histogram equalization + wavelet denoising")
plt.subplot(223)
plt.imshow(cl1)
plt.title("local histogram equalization")
plt.subplot(224)
plt.imshow(cl2)
plt.title("local histogram equalization+ wavelet denoising")
"""
for i in range(2):
    im=process(im)
    im = im/np.max(im)
    im*=255
    im=im.astype(np.uint8)
m.si(im)
"""
"""
for i in range(1,10):
    m.si(blobDet(img,i),"Blob sigma:"+str(i))"""

"""
gray = np.asarray(img)
t = skimage.filters.threshold_local(gray,501)
t = skimage.filters.threshold_otsu(gray)
t = gray>t
t=t.astype(np.uint8)*255
m.si(t)

kernel= np.ones((3,3),np.uint8)
#eroded = cv2.morphologyEx(t,cv2.MORPH_OPEN,kernel, iterations = 2)
eroded = cv2.erode(t,kernel,iterations = 4)
m.si2(gray,eroded,"base image","erosion")

"""
def gaussian(size,sigma):
    """Generates a square gaussian mask with size*size pixels and std sigma"""
    a,b=np.ogrid[-size/2:size/2,-size/2:size/2]
    mask = a**2+b**2
    mask = np.exp(-mask.astype('float')/(2*float(sigma**2)))
    return mask
#Try Template matching
#Gets a gaussian in the image:
def try_template_matching(img,threshold=0.8):
    x=1195
    y=370
    size = 60
    cl2 = img.copy()
    if cl2.dtype!="uint8":
        cl2/=np.max(cl2)
        cl2*=255
        cl2=cl2.astype(np.uint8)
    
    template = cl2[(y-size//2):(y+size//2),(x-size//2):(x+size//2)]
    #Simulated gaussian
    template = gaussian(size,25)
    template/=template.max()
    template*=255
    template = template.astype(np.uint8)
    #cv2.imwrite("template.tif",template)
    c= scipy.signal.convolve2d(cl2,template)
    m.si(c,"Image convolved")
        
    #OpenCv template matching
    
    w, h = template.shape[::-1]
    
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    img2 = cl2.copy()
    for meth in methods:
        img = img2.copy()
        method = eval(meth)
    
        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
        print res.dtype,res.max()
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            loc = np.where( res <= (1-threshold)*np.max(res))
        else:
            loc = np.where( res >= threshold*np.max(res))
            
        #loc = np.where( res >= threshold)
        
        #cv2.rectangle(img,top_left, bottom_right, 255, 2)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 255, 2)
        
        plt.figure()
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        #plt.colorbar()
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
    
        plt.show()

try_template_matching(cl2)

#Try ICA
from sklearn import decomposition
clf = decomposition.FastICA(n_components=2)
centered = cl1-np.mean(cl1)
centered = centered.astype(np.float)
sh = centered.shape
centered = centered.reshape(-1,1)
clf.fit(centered)
tr = clf.transform(centered)

centered = centered.reshape(sh)
tr=tr.reshape(sh)
m.si2(centered,tr,"original","ICA transofrmed")

