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
import time
import glob
import platform
"""Attempt to remove gaussian spots using wavelets"""
plt.close('all')

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


def wavelet_denoising2(im,wlt='sym2',lvl=5,fraction=0.76):
    coeffs_trous = pywt.swt2(im,wlt,lvl,start_level=0)
    total = np.ones(im.shape)
    #Add Gaussian blur
    for elts in coeffs_trous:
        cA,(cH,cV,cD) = elts
        var = m.getNoiseVar(cA,fraction)        
        cA = m.abe(cA,var)
        total*=cA
    return total

def gaussian(size,sigma):
    """Generates a square gaussian mask with size*size pixels and std sigma"""
    a,b=np.ogrid[-size/2:size/2,-size/2:size/2]
    mask = a**2+b**2
    mask = np.exp(-mask.astype('float')/(2*float(sigma**2)))
    return mask

def find_gaussian(img,sigma=25):
    """Optimal template matching to find a gaussian with std sigma in img"""
    method = 'cv2.TM_CCOEFF_NORMED'
    size=3*sigma
    template = gaussian(size,sigma)
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
    #m.si(img2)
    return stack_to_remove,loc

import scipy.optimize

def gaussian2((a,b),sigma,am,off,xoff,yoff):
    """Generates a square gaussian mask with std sigma"""
    mask = (a-xoff)**2+(b-yoff)**2
    mask = am*np.exp(-mask.astype('float')/(2*float(sigma**2)))+off
    return mask.ravel()

def gaussianFit(image):
    w = image.shape[0]
    a,b=np.ogrid[-w/2:w/2,-w/2:w/2]
    f = lambda (x,y),sigma,a,off,xoff,yoff : gaussian2((x,y),sigma,a,off,xoff,yoff)
    xdata = a**2+b**2
    xdata = (a,b)
    ydata = image.ravel()
    #sigma,amplitude,offset,xoff,yoff
    bounds_inf = [1,0,-255,-w/2,-w/2]
    bounds_sup = [w,255,255,w/2,w/2]
    bds=(bounds_inf,bounds_sup)
    try:
        fit = scipy.optimize.curve_fit(f,xdata,ydata,bounds=bds)
    except:
        print "optimisation failed"
        fit=-1
    return fit

def filter_out_gaussians(img):
    """Uses template matching to find different sizes of gaussians in 
    img, fits them and then removes them from img."""
    list_of_sigmas = [40,30,20]
    new_img = img.copy()
    for sigma in list_of_sigmas:
        stack_to_remove,locs=find_gaussian(new_img.astype(np.uint8),sigma)
        w = stack_to_remove[:,:,0].shape[0]
        a,b=np.ogrid[-w/2:w/2,-w/2:w/2]
        new_img=new_img.astype(np.float)
        for i in range(stack_to_remove.shape[2]):
            pt=(locs[0][i],locs[1][i])
            popt,pcov=gaussianFit(stack_to_remove[:,:,i])
            simul = gaussian2((a,b),popt[0],popt[1],popt[2],popt[3],popt[4]).reshape(w,w)
            new_img[pt[0]:pt[0]+w,pt[1]:pt[1]+w] -=simul
            new_img[new_img<0]=0
            #m.si2(stack_to_remove[:,:,i],stack_to_remove[:,:,i]-simul)
    
    #m.si2(img,new_img,"Original","Gaussian substracted")
    #plt.colorbar()
    return new_img

def deGaussianStack(path,target_dir):
    elements = glob.glob(path+"/*.png")
    if platform.system()=='Windows':
        separator="\\"
    else:
        separator="/"
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    for elt in elements:
        print "processing",elt.split(separator)[-1]
        img = Image.open(elt)
        img = np.asarray(img)
        filtered = filter_out_gaussians(img)
        cv2.imwrite(os.path.join(target_dir,'filtered_'+elt.split(separator)[-1]),filtered)
to_degaussian = os.path.join("..",'data','microglia','RFP1_denoised')
write_degaussian = os.path.join("..",'data','microglia','RFP1_degaussianed')
deGaussianStack(to_degaussian,write_degaussian)

frameNum = 11
path = os.path.join("..",'data','microglia','RFP1_denoised','filtered_Scene1Interval'+str(frameNum)+'_RFP.png')
path = os.path.join("..",'data','microglia','RFP1','Scene1Interval0'+str(frameNum)+'_RFP.png')

img = Image.open(path)

im = np.asarray(img)
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(30,30))
cl1 = clahe.apply(im)

cl2=clahe.apply(cl1)
#cl2 = cv2.equalizeHist(cl1)
cl2 = wavelet_denoising2(cl2,lvl=3)
m.si(cl2,"Image denoised used as work base")
cl2 = cl2*255/np.max(cl2)
cl2 = cl2.astype(np.uint8)

filtered = filter_out_gaussians(cl2)
m.si(filtered,"image filtered from gaussians")
filtered = filtered/np.max(filtered)*255
filtered=filtered.astype(np.uint8)
filtered=clahe.apply(filtered)
filtered=filtered.astype(np.float)
filtered= wavelet_denoising2(filtered,lvl=2,fraction=0.5)
m.si(filtered,"filtered denoised")
from skimage.filters import try_all_threshold
import skimage.filters
#try_all_threshold(filtered)

#t = skimage.filters.threshold_otsu(filtered)
t=skimage.filters.threshold_adaptive(filtered,501)
m.si2(filtered,t,"original","thresholded")

label,nr = ndi.label(t)
label=m.filter_by_size(label,40)
print "Evolution of the number of matches:",nr,np.max(label)
m.si(label,"labeled image")

t2=skimage.filters.threshold_adaptive(cl2,501)
kernel = np.ones((5,5),np.uint8)
eroded = cv2.erode(t2.astype(np.uint8),kernel,iterations=2)
m.si(eroded,"image eroded")
label_2,nr2 = ndi.label(t2)
label_2 = m.filter_by_size(label_2,40)
print "Evolution of the number of matches:",nr2,np.max(label_2)

