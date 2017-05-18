# -*- coding: utf-8 -*-
"""
Created on Wed May 03 11:20:33 2017

@author: univ4208
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.feature
import skimage.filters
import cv2
import scipy.ndimage as ndi
import pywt
import glob
import platform
#plt.close("all")

def openFrame(number):
    name = os.path.join("..","data","yoshi_mov_2",str(number)+".tif")
    img = Image.open(name)
    img = np.asarray(img)
    return img

def si(img,title=None):
    plt.figure()
    plt.imshow(img,cmap='gray')
    if title:
        plt.title(title)
        
def si2(im1,im2,title1=None,title2=None):
    plt.figure()
    plt.subplot(121)
    plt.imshow(im1)
    if title1:
        plt.title(title1)
    plt.subplot(122)
    plt.imshow(im2)
    if title2:
        plt.title(title2)

def displayWlt(wlt):
    cA, (cH, cV, cD)=wlt
    shapes = (cA.shape[0]+cV.shape[0],cA.shape[1]+cV.shape[1])
    out = np.zeros(shapes)
    out[:cA.shape[0],:cA.shape[1]]=cA
    out[cA.shape[0]:,cA.shape[1]:]=cD
       
    out[:cH.shape[0],cA.shape[1]:]=cH
    out[cA.shape[0]:,:cV.shape[1]]=cV
    return out

def displayMulti(wlt_list):
    cA = wlt_list[0]
    
    for i in range(1,len(wlt_list)):
        print i, cA.shape,wlt_list[i][0].shape
        cA = displayWlt((cA,wlt_list[i]))
    plt.figure()
    plt.imshow(cA,cmap="gray")
    plt.title("Wavelet decomposition")
    
def abe(img,variance):
    """proceeds to the Amplitude-scale invariant Bayes Estimation (ABE)"""
    nominator = img**2-3*variance
    nominator[nominator<0] = 0
    out = np.divide(nominator,img)
    out[img==0]=0
    return out

def getNoiseVar(img,fraction=0.95):
    """Gets the nth% of lower intensity pixels in an image correspondin to the noise
    n is determined empirically."""
    last_val = np.percentile(img,fraction)
    #si(img<last_val,title="Pixel values considered as noise")
    return np.var(img[img<last_val])

def filter_by_size(img_segm,mini_nb_pix):
    """filters a segmented image by getting rid of the components with too few pixels"""
    numbers = np.zeros(int(np.max(img_segm)))
    for i in range(1,int(np.max(img_segm))+1):
        numbers[i-1] = np.sum(img_segm==i)
        
    indexes = np.arange(1,np.max(img_segm)+1)
    #indexes = indexes[numbers>np.mean(numbers)] #Deletes the 1-pixel elements
    indexes = indexes[numbers>mini_nb_pix] #Deletes the 1-pixel elements
    
    segm_filtered = np.zeros(img_segm.shape)
    j=1
    for i in (indexes):
        segm_filtered[img_segm==i] = j
        j+=1
    return segm_filtered

def fillHoles(img):
    out,contour,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    i=0
    for cnt in contour:
            cv2.drawContours(img,contour,i,255,-1)
            i+=1
    return img

def modulusMaximum(amplitude,angle):
    """Unused method which computes for each pixel if it is the maximum of its neighbors
    along the axis defined by angle"""
    angle = np.mod(angle+np.pi,np.pi)/np.pi  #The sign does not matter anyway, goes between 0 and 1
    
    #Do 4 cases: angle <0.25,0.5,0.75,>0.75
    results_plus = np.zeros(amplitude.shape,dtype = np.uint8)
    results_minus = np.zeros(amplitude.shape,dtype = np.uint8)
    
    ampl_pad = np.pad(amplitude,1,'constant')
    
    #Case angle is between 0 and 45deg, ie angle is between 0 and 0.25
    print "max angle",np.max(angle)
    tmp = amplitude>ampl_pad[1:-1,2:]
    results_plus[angle<0.25] = tmp[angle<0.25]
    tmp = amplitude>ampl_pad[1:-1,:-2]
    results_minus[angle<0.25] = tmp[angle<0.25]
    
    #Case angle between 0.25 and 0.5
    tmp = amplitude>ampl_pad[:-2,2:]
    results_plus[np.logical_and(angle>=0.25,angle<0.5)] = tmp[np.logical_and(angle>=0.25,angle<0.5)]
    tmp = amplitude>ampl_pad[2:,:-2]
    results_minus[np.logical_and(angle>=0.25,angle<0.5)] = tmp[np.logical_and(angle>=0.25,angle<0.5)]
       
    #Case angle between 0.5 and 0.75
    tmp = amplitude>ampl_pad[2:,1:-1]
    results_plus[np.logical_and(angle>=0.5,angle<0.75)] = tmp[np.logical_and(angle>=0.5,angle<0.75)]
    tmp = amplitude>ampl_pad[:-2,1:-1]
    results_minus[np.logical_and(angle>=0.5,angle<0.75)] = tmp[np.logical_and(angle>=0.5,angle<0.75)]
    
    #Case angle >0.75
    tmp = amplitude>ampl_pad[:-2,:-2]
    results_plus[angle>=0.75] = tmp[angle>=0.75]
    tmp = amplitude>ampl_pad[2:,2:]
    results_minus[angle>=0.75] = tmp[angle>=0.75]
    
    return np.logical_and(results_plus,results_minus)

def saveFrame(name,total):
    if total.dtype!='uint8':
        total = total/np.max(total)*255
        total = total.astype(np.uint8)
    cv2.imwrite(name,total)

def wavelet_denoising(im,wlt='sym2',lvl=5):
    coeffs_trous = pywt.swt2(im,wlt,lvl,start_level=0)
    
    total = np.ones(im.shape)
    #Add Gaussian blur
    for elts in coeffs_trous:
        cA,(cH,cV,cD) = elts
        var = getNoiseVar(cA)        
        cA = abe(cA,var)
        cA= np.arctan(cA/np.max(cA)*np.pi*3)
        #m.si(tata)
        total*=cA
    return total

def segmentation(total):
    t = skimage.filters.threshold_li(total)
    mask = (total>t).astype(np.uint8)
    #mask = cv2.dilate(mask,np.ones((4,4)),iterations = 1)
    kernel = np.ones((5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #m.fillHoles(mask)
    si(mask)
    mask,nr = ndi.label(mask)
    mask = filter_by_size(mask,500)
    return mask.astype(np.uint8)

def wavelet_denoising2(im,wlt='sym2',lvl=5,fraction=0.76):
    coeffs_trous = pywt.swt2(im,wlt,lvl,start_level=0)
    total = np.ones(im.shape)
    #Add Gaussian blur
    for elts in coeffs_trous:
        cA,(cH,cV,cD) = elts
        var = getNoiseVar(cA,fraction)        
        cA = abe(cA,var)
        #m.si(tata)
        total*=cA
    return total

def gaussian(size,sigma):
    """Generates a square gaussian mask with size*size pixels and std sigma"""
    a,b=np.ogrid[-size/2:size/2,-size/2:size/2]
    mask = a**2+b**2
    mask = np.exp(-mask.astype('float')/(2*float(sigma**2)))
    return mask

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
        cl2 = wavelet_denoising2(cl2,lvl=3)
        cl2 = cl2*255/np.max(cl2)
        cl2 = cl2.astype(np.uint8)
        cv2.imwrite(os.path.join(target_dir,'filtered_'+elt.split(separator)[-1]),cl2)

#----------------Handling gaussians------------------------------------------
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

def where_are_gaussians(img):
    """Finds gaussians in img and returns a mask of pixels in a gaussian."""
    list_of_sigmas = [40,30,20,10]
    mask=np.zeros(img.shape,dtype=bool)
    for sigma in list_of_sigmas:
        stack_to_remove,locs=find_gaussian(img.astype(np.uint8),sigma)
        w = stack_to_remove[:,:,0].shape[0]
        a,b=np.ogrid[-w/2:w/2,-w/2:w/2]
        for i in range(stack_to_remove.shape[2]):
            pt=(locs[0][i],locs[1][i])
            mask[pt[0]:pt[0]+w,pt[1]:pt[1]+w] = True
    return mask

#-------------Display functions---------------------------------------
def show_points_on_img(mask,img):
    """Shows the points encoded in mask on img"""
    labeled, num_objects = ndi.label(mask)
    slices = ndi.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    plt.figure()
    plt.imshow(img)
    plt.autoscale(False)
    plt.plot(x,y, "o")

def show_holes_on_img(mask,img):
    """Shows the points encoded in mask on img"""
    labeled, num_objects = ndi.label(mask)
    slices = ndi.find_objects(labeled)
    radius=9
    out_image = img.copy()
    out_image = cv2.cvtColor(out_image, cv2.COLOR_GRAY2RGB)
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2    
        center=(x_center,y_center)
        cv2.circle(out_image, center, radius,(111,17,108),thickness=2)

    plt.figure()
    plt.imshow(out_image)
    plt.autoscale(False)
    return out_image
# display results
def overlay_mask2image(img,mask,title=None):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    #ax = axes.ravel()
    ax = axes
    ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax.imshow(mask, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
    if title:
        ax.set_title(title)
    
    fig.tight_layout()
    plt.show()

def cv_overlay_mask2image(mask,img):
    """Overlay a mask to an image using opencv"""
    transparency=0.2
    image=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    if mask.dtype==np.int:
        mask = mask>0
        mask = mask.astype(np.uint8)*255
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
    mask[:,:,0]=0
    mask[:,:,2]=0
    cv2.addWeighted(mask,transparency,image,1-transparency,0,image)
    return image

