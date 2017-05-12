b# -*- coding: utf-8 -*-
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
import glob
import os
from PIL import Image
from skimage.filters import try_all_threshold
import platform

def gf(img,sigma):
    img2 = skimage.filters.gaussian(img,sigma)
    return img2

def removeGaussians(img):
    """Convolves the original image with different scales of gaussians
    If an object is gaussian, the values of the object convolved should be high."""
    #filtered = m.wavelet_denoising(img,lvl=2)
    filtered = np.copy(img)
    filtered = filtered/np.max(filtered)
    #Tries different thresholds
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
    total_maxwise = np.zeros(img.shape)
    for i in range(15,16):
        out = gf(filtered,i)
        total*=out
        total_maxwise[out>total_maxwise] = out[out>total_maxwise]
        
    m.si(total,"product of stuff")
    m.si(total_maxwise,"Maxima")
    print np.count_nonzero(total)
    ###################################################################
    ## Find the elements which have a high 'total' value corresponding to the out of focus stuff
    ## Inch Allah
    #######################################################################
    mask = np.logical_and(total<1.2,total>0.8)
    try_all_threshold(total)
    m.si(mask,"mask of the elements to remove")
    labels_to_remove = labels[mask]
    labels_to_remove = np.unique(labels_to_remove)
    
    filtered_array = np.copy(filtered)
    for lab in labels_to_remove:
        filtered_array[labels==lab]=0
    
    print "Nb elements supprimes: ", labels_to_remove.size
    m.si2(img,filtered_array,"Original image","Degaussianed image")
    
def denoiseStack(path,target_dir):
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
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(30,30))
        cl1 = clahe.apply(img)
        cl2 = clahe.apply(cl1)
        cl2 = m.wavelet_denoising2(cl2,lvl=3)
        cl2 = cl2*255/np.max(cl2)
        cl2 = cl2.astype(np.uint8)
        cv2.imwrite(os.path.join(target_dir,'filtered_'+elt.split(separator)[-1]),cl2)
plt.close('all')

frame_number=130
#Opening desired image
name = os.path.join("..",'data','microglia','RFP1','Scene1Interval'+str(frame_number)+'_RFP.png')
img = Image.open(name)
img = np.asarray(img)

removeGaussians(img)
"""Use Watershed!!!"""

#equilibrium = cv2.equalizeHist(img.astype(np.uint8))
#m.si(equilibrium,'Histogram equilibrated image')
filtered = m.wavelet_denoising2(img,wlt='sym2',lvl=6,fraction=0.8)
print np.max(filtered)
filtered = filtered/np.max(filtered)*255
filtered = filtered.astype(np.uint8)
equilibrium = cv2.equalizeHist(filtered)
m.si2(equilibrium,filtered,'Histogram equilibrated image','image filtered')
#cA= np.arctan(img/np.max(img)*np.pi*5)
#for i in range(1,7):
#    m.si(m.wavelet_denoising2(equilibrium,wlt='sym2',lvl=i,fraction=0.8),'Denoising lvl '+str(i))
#removeGaussians(filtered)

#Saves the denoised images
path = os.path.join("..",'data','microglia','RFP2')
new_path = os.path.join("..",'data','microglia','RFP2_denoised')

#############################################

from skimage.morphology import watershed, disk
from skimage.filters import rank
cA = img

cA=cA.astype(np.float)
cA=np.arctan(cA/np.max(cA)*np.pi*3)

filt =  m.wavelet_denoising2(cA,wlt='sym2',lvl=6,fraction=0.2)
m.si2(cA,filt,"Image with arctan","Arctan denoised")

cA = cA/np.max(cA)
denoised = rank.median(cA, disk(2))
eq = cv2.equalizeHist(img)
denoised2 = m.wavelet_denoising2(eq,wlt='sym2',lvl=6,fraction=0.2)
denoised2 = denoised2*255/np.max(denoised2)
denoised2=denoised2.astype(np.uint8)
denoised2 = rank.median(denoised2, disk(4))
m.si2(denoised,denoised2,"denoising after arctan","denoising after histo eq")

denoiseStack(path,new_path)