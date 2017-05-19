# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:25:22 2017

@author: univ4208
"""
import matplotlib.pyplot as plt
import methods as m
import os
from PIL import Image
import numpy as np
import cv2
import skimage.filters as f
import scipy.ndimage as ndi
import ftracker

def openImage(frameNum):
    frameNum=str(frameNum)
    nb = '0000'
    nb=nb[0:4-len(frameNum)]+frameNum
    path = os.path.join("..",'data','microglia','RFP1_cropped','RFP1_denoised'+nb+'.png')
    img = Image.open(path)
    im = np.asarray(img)
    return im

def find_cells(frameNum,sizeFilter=150,cinema=False):
    """Thresholding+opening+labeling+size filtering"""
    frameNum=str(frameNum)
    nb = '0000'
    nb=nb[0:4-len(frameNum)]+frameNum
    path = os.path.join("..",'data','microglia','RFP1_cropped','RFP1_denoised'+nb+'.png')
    img = Image.open(path)
    im = np.asarray(img)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(30,30))
    im = clahe.apply(im)
    
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
    def __init__(self,path=None):
        self.info_list=[]
        self.correspondance_lists=[]
        self.first_frame = 0
        self.last_frame = 150
        
        #Unused yet
        if path:
            self.path = path
        else:
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
            
    def segment(self,path=None):
        """Uses the information extracted fron the preprocessing step to segment
        the images"""
        if path==None:
            path = path = os.path.join("..",'data','microglia','RFP1_cropped_segmented')
        
        first_labels = find_cells(self.first_frame)
        
        labels_buffer = np.zeros((first_labels.shape[0],first_labels.shape[1],2))
        labels_buffer[:,:,self.first_frame%2]=first_labels
        
        current_index=0   #Index in the referential of Tracker (equal to 0 at self.first_frame)
        for i in range(self.first_frame+1,self.last_frame+1):
            
            labels=find_cells(i)
            labels_buffer[:,:,i%2]= labels
            correspondance = self.correspondance_lists[current_index]
            disappear=[]    # Elements disappearing in the frame before
            appear = []   #Elements appearing in the current frame
            for bef,aft in correspondance:
                print bef,aft
                if bef==-1:
                    appear.append(aft)
                if aft==-1:
                    disappear.append(bef)
            current_index+=1
            # If there are cells disappearing:
            for index in disappear:
                elts_in_contact = labels[labels_buffer[:,:,(i-1)%2]==(index+1)]
                candidates = np.unique(elts_in_contact)
                candidates = [x for x in candidates if x!=0]
                if len(candidates)==1:
                    print "found who it disappeared for"
                else:
                    print "error several candidates index ",i
                
    def find_indices(self,nr_frame,label,forward=True):
        """For a given cell in a given frame, gets the corresponding indices
        in the next (if forward=true) or the previous frames"""
        if forward:
            index = nr_frame-self.first_frame
            label_list=[label]
            #Fetches the 10 first frames. 10 is arbitrary
            n_iterations = min(10,len(self.correspondance_lists)-index-1 )
            for i in range(n_iterations):
                corresp_list = self.correspondance_lists[index+i]
                match = [v for u,v in corresp_list if u==label_list[index+i]]
                match = match[0]
                if match==-1:
                    break
                
                label_list.append(match)
            return label_list
        
        else:
            index = nr_frame-self.first_frame
            label_list=[label]
            #Fetches the 10 first frames. 10 is arbitrary
            n_iterations = min(10,index )
            for i in range(n_iterations):
                corresp_list = self.correspondance_lists[index-i]
                match = [u for u,v in corresp_list if v==label_list[index-i]]
                match = match[0]
                if match==-1:
                    break
                label_list.append(match)
            return label_list
                
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
"""
#20,21 there is an apparition
trac = Tracker()
trac.preprocess(15,30)
trac.segment()
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
m.si2(unfiltered20,unfiltered21,"remaining in fr 20","in fr 21")"""