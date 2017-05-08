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
import scipy.ndimage as ndi
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
    
        print meth,res.min(),res.max()
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
        plt.colorbar()
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
    
        plt.show()
import time
"""
t=time.time()
try_template_matching(cl1)
print "elapsed time: ", time.time()-t
"""

def optimal_template_matching(img):
    """Optimal template matching based on experimentation"""
    method = 'cv2.TM_CCOEFF_NORMED'
    size = 60
    template = gaussian(size,25)
    template/=template.max()
    template*=255
    template = template.astype(np.uint8)
    
    threshold = 0.9
    w, h = template.shape[::-1]
    
    img2 = img.copy()
    meth = eval(method)

    # Apply template Matching
    res = cv2.matchTemplate(img2,template,meth)
    #Filters location map so that only one gaussian is found per contiguous location
    location_map =  res >= threshold*np.max(res)
    location_map,nr = ndi.label(location_map)
    print "Nb of contiguous zones detected:",nr
    list_x = []
    list_y = []
    for label in range(1,nr+1):
        tmp=location_map==label
        if np.count_nonzero(tmp)>1:
            points = np.where(tmp)
            l = len(points[0])
            cx = (np.sum(points[0]) + l/2)/l
            cy = (np.sum(points[1]) + l/2 )/l
            list_x.append(cx)
            list_y.append(cy)
    loc= (np.asarray(list_x),np.asarray(list_y))
    stack_to_remove = np.zeros((size,size,len(loc[0])))
    i=0
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img2, pt, (pt[0] + w, pt[1] + h), 255, 2)
        stack_to_remove[:,:,i] = img[pt[1]:pt[1]+w,pt[0]:pt[0]+h]
        
        i+=1
    plt.figure()
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.colorbar()
    plt.subplot(122),plt.imshow(img2,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(method)

    plt.show()
    return stack_to_remove,loc
t=time.time()
stack_to_remove,locs=optimal_template_matching(cl1)
print "elapsed time: ", time.time()-t

#visualizing the objects
"""
for i in range(stack_to_remove.shape[2]/2):
    m.si2(stack_to_remove[:,:,2*i],stack_to_remove[:,:,2*i+1])"""

import scipy.optimize

def gaussian2((a,b),size,sigma,am,off,xoff,yoff):
    """Generates a square gaussian mask with size*size pixels and std sigma"""
    mask = (a-xoff)**2+(b-yoff)**2
    mask = am*np.exp(-mask.astype('float')/(2*float(sigma**2)))+off
    return mask.ravel()

def gaussianFit(image):
    w = image.shape[0]
    a,b=np.ogrid[-w/2:w/2,-w/2:w/2]
    f = lambda (x,y),sigma,a,off,xoff,yoff : gaussian2((x,y),w,sigma,a,off,xoff,yoff)
    xdata = a**2+b**2
    xdata = (a,b)
    ydata = image.ravel()
    #sigma,amplitude,offset,xoff,yoff
    bounds_inf = [1,0,-255,-w/2,-w/2]
    bounds_sup = [w,255,255,w/2,w/2]
    bds=(bounds_inf,bounds_sup)
    fit = scipy.optimize.curve_fit(f,xdata,ydata,bounds=bds)
    return fit

popt,pcov=gaussianFit(stack_to_remove[:,:,0])
perr=  np.sqrt(np.diag(pcov))
print "error estination:",perr
w = stack_to_remove[:,:,0].shape[0]
a,b=np.ogrid[-w/2:w/2,-w/2:w/2]
new_img = cl1.copy()

for i in range(stack_to_remove.shape[2]):
    print np.max(stack_to_remove[:,:,i])
    popt,pcov=gaussianFit(stack_to_remove[:,:,i])
    simul = gaussian2((a,b),w,popt[0],popt[1],popt[2],popt[3],popt[4]).reshape(w,w)
    pt=(locs[0][i],locs[1][i])
    new_img[pt[0]:pt[0]+w,pt[1]:pt[1]+w] -=simul.astype(np.uint8)
    #m.si2(stack_to_remove[:,:,i],stack_to_remove[:,:,i]-simul,
    #     "Original image nr "+str(i),"Difference with simulation")
m.si2(cl1,new_img,"Original","Gaussian substracted")