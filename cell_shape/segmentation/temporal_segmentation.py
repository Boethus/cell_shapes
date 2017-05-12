#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:59:25 2017

@author: aurelien
"""

import matplotlib.pyplot as plt
import methods as m
import os
from PIL import Image
import numpy as np
import cv2
import skimage.filters
import skimage.filters as f
import scipy.ndimage as ndi
import time
from skimage.filters import try_all_threshold
import ftracker

plt.close('all')

def proba_map_gaussian(img):
    """Optimal template matching to find a gaussian with std sigma in img"""
    list_of_sigmas = [40,30,20]
    list_of_sigmas = range(9,40)
    list_of_results = []
    max_map = np.zeros(img.shape)
    coeffs_map = np.zeros(img.shape)
    method = 'cv2.TM_CCOEFF_NORMED'
    for sigma in list_of_sigmas:
        size=3*sigma
        if size%2==0:
            size+=1
        template = m.gaussian(size,sigma)
        template/=template.max()
        template*=255
        template = template.astype(np.uint8)
        
        w, h = template.shape[::-1]
        
        img2 = img.copy()
        meth = eval(method)
        # Apply template Matching
        res = cv2.matchTemplate(img2,template,meth)
        res = np.pad(res,size/2,mode="constant")
        indices = res>max_map
        max_map[indices] = res[indices]
        coeffs_map[indices] = sigma
        list_of_results.append(np.max(res))
    plt.figure()
    plt.plot(list_of_sigmas,np.asarray(list_of_results))
    plt.xlabel("sigma")
    plt.ylabel("maximum proba map value")
    plt.title("Finding a gaussian mask of size sigma in frame 130")
    m.si2(img,max_map,'original image','Map of maximum probabilities for template matching')
    return max_map,coeffs_map

def custom_threshold(im):
    th1=im>skimage.filters.threshold_niblack(im,55,-0.5)
    th1=np.logical_and(th1,im>skimage.filters.threshold_otsu(im))
    #m.si(th1,"1st threshold before morphology")
    kernel = np.ones((5,5),np.uint8)
    opening1 = cv2.morphologyEx(th1.astype(np.uint8), cv2.MORPH_OPEN, kernel,iterations=2)
    #m.si(opening1,"After opening")
    opening1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel,iterations=2)
    return opening1

def find_cells(frameNum,sizeFilter=150,cinema=False):
    frameNum=str(frameNum)
    nb = '0000'
    nb=nb[0:4-len(frameNum)]+frameNum
    path = os.path.join("..",'data','microglia','RFP1_cropped','RFP1_denoised'+nb+'.png')
    img = Image.open(path)
    im = np.asarray(img)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(30,30))
    im = clahe.apply(im)
    #thresh_hard = custom_threshold(im)
    thresh_hard = (im>f.threshold_li(im)).astype(np.uint8)
    if sizeFilter>0:
        kernel = np.ones((5,5),np.uint8)
        thresh_hard = cv2.morphologyEx(thresh_hard.astype(np.uint8), cv2.MORPH_OPEN, kernel,iterations=1)
        
    labels,nr = ndi.label(thresh_hard)
    if sizeFilter>0:
        labels = m.filter_by_size(labels,sizeFilter)
        
    if cinema:
        im.setflags(write=1)
        im[labels==0]=0
        cv2.imshow("Frame",im)
        cv2.waitKey(60)
    return labels

def openImage(frameNum):
    frameNum=str(frameNum)
    nb = '0000'
    nb=nb[0:4-len(frameNum)]+frameNum
    path = os.path.join("..",'data','microglia','RFP1_cropped','RFP1_denoised'+nb+'.png')
    img = Image.open(path)
    im = np.asarray(img)
    return im

class Cell(object):
    def __init__(self,nb,n_frames):
        self.number=nb
        self.n_frames = n_frames
        self.list_of_ids = []
        self.gaussian_coeffs=[]
        
def movie():
    points=[]
    for i in range(150):
        n=find_cells(i,True)
        points.append(np.max(n))
    cv2.destroyAllWindows()
    plt.figure()
    plt.plot(points)
    plt.title("Number of elements detected")
"""
plt.close('all')
frameNum = 32
frameNum=str(frameNum)
nb = '0000'
nb=nb[0:4-len(frameNum)]+frameNum
path = os.path.join("..",'data','microglia','RFP1_cropped','RFP1_denoised'+nb+'.png')
img = Image.open(path)
im = np.asarray(img)


clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(30,30))
im = clahe.apply(im)
m.si(im,"Local histo equalization on image")
max_map,coeffs_map=proba_map_gaussian(im)
print np.max(max_map)
plt.figure()
plt.subplot(131)
plt.imshow(im)
plt.subplot(132)
plt.imshow(max_map)
plt.subplot(133)
plt.imshow(coeffs_map)

elt = find_cells(frameNum)
print np.max(elt)
#movie()

img_underlined = im.copy()
img_underlined[elt>0]=255
m.si(img_underlined,"image underlined")
"""
def overlay(img,thresh):
    out = img.copy()
    if thresh.dtype==bool:
        out[~thresh]=0
    else:
        out[thresh==0]=0
    return out
"""
img_underlined2 = im.copy()
th_li = f.threshold_li(im)
img_underlined2[im<th_li]=0
tli = im>th_li
lab,nr = ndi.label(tli)
lab=m.filter_by_size(lab,20)
nr=np.max(lab)
m.si2(img_underlined2,lab,"Threshold li","labeled "+str(nr)+" components")
opli = cv2.morphologyEx(tli.astype(np.uint8), cv2.MORPH_OPEN, kernel,iterations=2)
m.si2(im,overlay(im,opli),"original","pverlaid")
"""
class FrameInfo(object):
    """Class remembering the information from a labeled frame"""
    def __init__(self,frame_nr,frame):
        self.nr = frame_nr
        self.n_objects = int(np.max(frame))
        self.objects_size = []
        self.objects_centroids = []
        for i in range(1,self.n_objects+1):
            pos_list = np.where(frame==i)
            xc = np.mean(pos_list[0])
            yc = np.mean(pos_list[1])
            self.objects_centroids.append((xc,yc))
            self.objects_size.append( len(pos_list[0]))
        
class Tracker(object):
    """Object iterating through a set of frames contiguous in time.
    Segments each frame and looks for correspondances in the previous frame
    usin the hungarian algorithm. Stores the main results in info_list and
    correspondance_lists
    
    We use here a quite heavily filtered by size image, as we are not interested 
    in small pieces here for main tracking"""
    def __init__(self):
        self.info_list=[]
        self.correspondance_lists=[]
        self.first_frame = 0
        self.last_frame = 150
        
        #Unused yet
        self.path = os.path.join("..",'data','microglia','RFP1_cropped')
        
    def preprocess(self,first_frame = -1, last_frame = -1):
        if first_frame>=0:
            self.first_frame = first_frame
        if last_frame>=0 and last_frame>first_frame:
            self.last_frame = last_frame
            
        first_labels = find_cells(self.first_frame)
        
        labels_buffer = np.zeros((first_labels.shape[0],first_labels.shape[1],2))
        labels_buffer[:,:,0]=first_labels
        
        self.info_list.append(FrameInfo(self.first_frame,first_labels))
        
        for i in range(self.first_frame+1,self.last_frame+1):
            print "Tracking iteration ",i
            labels = find_cells(i)
            prev_labels = labels_buffer[:,:,(i-1)%2]
            match_list = ftracker.w_hungarian(prev_labels,labels)
            labels_buffer[:,:,(i)%2] = labels
            self.correspondance_lists.append(match_list)
            self.info_list.append(FrameInfo(i,labels))
            
    def showTrajectory(self,cell_of_interest=5,overlay=False,plot=False,wait=50):
        #Print evolution of different parameters for one sinle cell
        
        sizes = []
        sizes.append(self.info_list[0].objects_size[cell_of_interest])
        info_index=1
        cell_list=[cell_of_interest]
        for elements in self.correspondance_lists:
            corresp = -1
            for (u,v) in elements:
                if u==cell_of_interest:
                    corresp = v
            if corresp==-1:
                print "Correspondace lost."
                print "Helloooooo"
                break
            cell_of_interest = corresp
            if cell_of_interest>=len(self.info_list[info_index].objects_size):
                print "Target lost"
                break
            cell_list.append(cell_of_interest)
            sizes.append(self.info_list[info_index].objects_size[cell_of_interest])
            info_index+=1
            
        if plot:
            plt.figure()
            plt.plot(sizes)
            plt.title("Evolution of the size of a cell")
        
        speeds = []
        index=0
        xs=0
        ys=0
        speed=0
        for i in cell_list:
            lab = find_cells(self.first_frame+index)
            #Get centroid
            pos_list = np.where(lab==i+1)
            if index>0:
                speed = np.sqrt( (xs-np.mean(pos_list[0]))**2 + (ys-np.mean(pos_list[1]))**2 )
                speeds.append(speed)
            xs = np.mean(pos_list[0])
            ys = np.mean(pos_list[1])
            if overlay:
                im=openImage(self.first_frame+index)
                im.setflags(write=1)
                im[lab!=i+1]=0
                cv2.imshow("frame",im)
            else:
                cv2.imshow("frame",(lab==i+1).astype(np.float))
            cv2.waitKey(wait)
            index+=1
        plt.figure()
        plt.plot(speeds)
        plt.title("Motion of the detected cell in pixels")
            
    def showMovie(self,first_frame = -1, last_frame = -1,wait=50):
        if first_frame<0:
            first_frame = self.first_frame
        if last_frame<0:
            last_frame = self.last_frame
        for i in range(first_frame,last_frame+1):
            cv2.imshow("Not processed movie",openImage(i))
            cv2.waitKey(wait)
            
    def showLabels(self,first_frame = -1, last_frame = -1,wait=50):
        if first_frame<0:
            first_frame = self.first_frame
        if last_frame<0:
            last_frame = self.last_frame
        for i in range(first_frame,last_frame+1):
            cv2.imshow("Not processed movie",find_cells(i))
            cv2.waitKey(wait)

trac = Tracker()
trac.preprocess(20,21)
trac.showTrajectory(4,overlay=True,plot=True,wait=400)
#trac.showMovie(wait=200)
c= find_cells(11)
d=m.filter_by_size(c,150)
m.si2(c,d,"normal","filtered by size")
m.si2(find_cells(20),find_cells(21),"frame 20","frame21")
niou_list = [(u+1,v+1) for u,v in trac.correspondance_lists[0]]

f20 = find_cells(20)
f21 = find_cells(21)

unfiltered20 = find_cells(20,sizeFilter=0)
unfiltered21 = find_cells(21,sizeFilter=0)

unfiltered20[f20>0]=0
unfiltered21[f21>0]=0
m.si2(unfiltered20,unfiltered21,"remaining in fr 20","in fr 21")