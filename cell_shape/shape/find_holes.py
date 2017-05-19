# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:30:08 2017

@author: univ4208
"""

import os
import numpy as np
from PIL import Image
import sys
sys.path.append(os.path.join("..","segmentation"))
import methods as m
import skimage.filters as f
import matplotlib.pyplot as plt
import cv2
import skimage.filters
import scipy.ndimage as ndi
from skimage import img_as_ubyte
from skimage.morphology import disk
import skimage.morphology
plt.close('all')

global_path = os.path.join("..",'data','microglia','RFP1_cropped')
def open_frame(frameNum):
    first_path=os.path.join("..",'data','microglia','RFP1_denoised')
    frameNum=str(frameNum)
    nb = '000'
    nb=nb[0:len(nb)-len(frameNum)]+frameNum
    path = os.path.join(first_path,'filtered_Scene1Interval'+nb+'_RFP.png')
    img = Image.open(path)
    im = np.asarray(img)
    return im

def open_cropped_frame(frameNum):
    frameNum=str(frameNum)
    nb = '000'
    nb=nb[0:len(nb)-len(frameNum)]+frameNum
    path = os.path.join(global_path,'RFP1_denoised'+nb+'.png')
    img = Image.open(path)
    im = np.asarray(img)
    return im

def open_raw_frame(frameNum):
    frameNum=str(frameNum)
    nb = '000'
    nb=nb[0:len(nb)-len(frameNum)]+frameNum
    path = os.path.join("..",'data','microglia','RFP1','Scene1Interval'+nb+'_RFP.png')
    img = Image.open(path)
    im = np.asarray(img)
    return im

def try_template_matching(img,template):
    """Tries template matching for template in image using 6 different methods and
    plots the result"""
    img2 = img.copy()
    w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        res-=np.min(res)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        threshold=0.9*np.max(res)
        
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            loc = np.where( res <=0.1*np.max(res))
        else:
            loc = np.where( res >= threshold)
        
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 255, 2)
        plt.figure()
        plt.subplot(121),plt.imshow(image_normal,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()
"""
try_template_matching(im,template)

out=removing_oof.gaussianFit(np.max(template)-template)
w = template.shape[0]+2
a,b=np.ogrid[-w/2+1:w/2+1,-w/2+1:w/2+1]
out=out[0]
fit = removing_oof.gaussian2((a,b),out[0],out[1],out[2],out[3],out[4])
fit = fit.reshape((w,w))
fit = np.max(fit)-fit
#im = img_as_ubyte(f.sobel(im))
fit = fit/np.max(fit)
fit = img_as_ubyte(f.sobel(fit))

m.si(fit,"model of hole")
fit = (fit/np.max(fit)*255).astype(np.uint8)
try_template_matching(im,fit)

#Try to detect holes based on their position in the histogram of the egmented cells
# See if we can do it cell-wise
labels,nr = ndi.label(thresh_hard)
print "Number of components deected",nr
m.si(labels)
mask = labels==49
dist_transform = cv2.distanceTransform(thresh_hard.astype(np.uint8),cv2.DIST_L2,5)
m.si(dist_transform,"distance transform")

data = image_normal[np.logical_and(mask,dist_transform>5)]
raw_data = raw_im[np.logical_and(mask,dist_transform>5)]

plt.figure()
plt.subplot(121)
plt.hist(raw_data,bins=500)
plt.title("histo o raw data")
plt.subplot(122)
plt.hist(data,bins=500)
plt.title("histo of data processed")

print "Mean:",np.mean(data),"std:",np.std(data),"mean- 3std",np.mean(data)-3*np.std(data)

kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(thresh_hard.astype(np.uint8), cv2.MORPH_OPEN, kernel,iterations=1)
diff = closing-thresh_hard
m.si2(closing,diff)"""

#Try minimum filtering
def hysteresis_thresholding(image,th1,th2):
    
    hard_th = (image>max(th1,th2)).astype(np.uint8)*255
    soft_th = (image>min(th1,th2)).astype(np.uint8)*255
    added_elements = np.zeros(image.shape,dtype=np.uint8)
    
    labels,nr = ndi.label(soft_th-hard_th)
    kernel = np.ones((2,2),dtype=np.uint8)
    hard_th_exp = skimage.morphology.binary_dilation(hard_th,selem=kernel)
    connected_labels = np.unique( labels[np.logical_and(labels,hard_th_exp)] )
    connected_labels = [x for x in connected_labels if x>0]
    for lab in connected_labels:
        added_elements[labels==lab]=255
    return added_elements+hard_th

def find_local_minima(image):
    """Finds local minima in segmeted image (holes)"""
    mini_mask = disk(3)
    img = skimage.filters.gaussian(image,0.5)
    thresh = (img>f.threshold_li(img))
    filtered = ndi.filters.minimum_filter(img,footprint=mini_mask)
    is_local_minimum = filtered==img
    is_local_minimum = np.logical_and(is_local_minimum,thresh)
    return is_local_minimum

def local_maxima(image,radius=12):
    """Finds local minima in segmeted image (holes)"""
    maxi_mask = disk(radius)
    
    maxima = ndi.filters.maximum_filter(image,footprint=maxi_mask)
    return maxima-image

def findHoles(image):
    """Finds local minima in an image and counts them as holes if they are not 
    within a gaussian"""
    local_minima=find_local_minima(image)
    mask_gaussians=m.where_are_gaussians(image)
    im_holes=m.show_holes_on_img(np.logical_and(local_minima,~mask_gaussians),image_normal)
    local_minima = np.logical_and(local_minima,~mask_gaussians)
    local_minima = local_minima.astype(np.uint8)
    return lab,im_holes

def classifyPhaseImage():
    phase_path = os.path.join("..",'data','microglia','Beacon-1 unst',"Scene1Interval"+str(fr_nb)+"_PHASE.png")
    
    phase= Image.open(phase_path)
    phase = np.asarray(phase)
    X=phase.reshape(-1,1)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3).fit(X)
    classified = kmeans.labels_
    classified=classified.reshape(phase.shape)
    m.si2(phase,classified,"Phase image","Classification")

def cell_arms(image,size_to_remove=64):
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(50,50))
    img = clahe.apply(image)
    threshold = img>skimage.filters.threshold_li(img)
    threshold=threshold.astype(np.uint8)
    kernel = np.ones((5,5),dtype = np.uint8)
    threshold_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel,iterations=3)
    diff = threshold_open-threshold
    lab,_ = ndi.label(diff)
    lab = skimage.morphology.remove_small_objects(lab,size_to_remove)
    lab = (lab>0).astype(np.uint8)*255
    return lab

def save_cell_arms():
    for i in range(1,150):
        im=open_frame(i)
        mask = cell_arms(im,150)
        m.cv_overlay_mask2image(im,mask)
        savepath = os.path.join("..",'data','microglia','cell_arms',str(i)+'.png')
        cv2.imwrite(savepath,im)
 
fr_nb = 211
def processFrame(fr_nb):
    im=open_frame(fr_nb)
    img=im.copy()
    im=img_as_ubyte(im)
    mask_h = hysteresis_thresholding(img,6,10)
    out = m.cv_overlay_mask2image(mask_h,im)
    m.si(out)
    
    ksize=5
    kernel = np.ones((ksize,ksize),dtype = np.uint8)
    kernel = disk(ksize)

    #ots = int(1.3*f.threshold_otsu(img))
    ots = int(f.threshold_otsu(img))
    mask = img>ots
    mask = img_as_ubyte(mask)
    
    mask = cv2.morphologyEx(mask_h, cv2.MORPH_OPEN, kernel,iterations=2)
    diff = mask_h-mask
    lab,_ = ndi.label(diff)
    
    filtered_size = skimage.morphology.remove_small_objects(lab,60)   #Only temporary, to track only the biggest
    out = m.cv_overlay_mask2image(filtered_size,img)
    return out

savepath=os.path.join("..","data","microglia","cell_arms_2")
for i in range(31,241):
    print "processing frame"+str(i)
    cv2.imwrite(os.path.join(savepath,str(i)+".png"),processFrame(i))