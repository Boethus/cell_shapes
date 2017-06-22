#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:30:46 2017

@author: aurelien
"""
import methods as m
import cv2
import numpy as np
import os
from find_arms import Experiment,loadObject
from find_arms import *
import glob

def resize_image(image,new_size=80):
    """
    Parameters:
        image: 2-D or 3-D numpy array
        new_size : int, gives the desired maximum dimension
    Returns:
        resized: numpy array"""
    if image.shape[1]>image.shape[0]:
        r = float(new_size)/ image.shape[1]
        dim = (new_size,int(image.shape[0] * r))
    else:
        r = float(new_size)/ image.shape[0]
        dim = ( int(image.shape[1] * r),new_size)
        
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized
def get_arms_list_per_frame(experiment,simple_trajectories):
    """Extracts all arms from a set of manually selected trajectories and
    saves them in a folder
    """
    arms_list=[]
    for i in range(experiment.n_frames-1):
        arms_list.append([])
    
    if type(simple_trajectories[0])==tuple:
        simple_trajs = [y for x,y in simple_trajectories]
    else:
        simple_trajs = simple_trajectories
       
    for traj in simple_trajs:
        for cell in traj.cells:
            frame_number= cell.frame_number
            for arm in cell.arms:
                arms_list[frame_number].append(arm)
    return arms_list

def arm_bb(experiment,frame_number,arm_label):
    """ Returns an image of an arm only using its label and frame number
    """
    frame = m.open_frame(experiment.path,frame_number+1)
    arm = m.open_frame(experiment.arm_path,frame_number+1)
    
    roi = arm==(arm_label+1)
    im2,contours,hierarchy = cv2.findContours((roi).astype(np.uint8), 1, 2)
    if len(contours)==1:
        cnt = contours[0]
    else:
        #If find several contours, takes the largest
        widths=[]
        for i in range(len(contours)):
            cnt = contours[i]
            x,y,w,h = cv2.boundingRect(cnt)
            widths.append(w)
        if max(widths)>600:
            print "gih max in ",frame_number,arm_label
        indices = [i for i,wid in enumerate(widths) if wid==max(widths)]
        indices = indices[0]
        cnt = contours[indices]
    
    x,y,w,h = cv2.boundingRect(cnt)
    sub_frame = frame[y:y+h,x:x+w]
    sub_frame*=int(255/np.max(sub_frame))  #To have balanced histograms
    sub_roi = roi[y:y+h,x:x+w]
    sub_frame[sub_roi==0]=0
    return sub_frame

def get_longer_arm(experiment,frame_number,arm_labels):
    """ Returns an image of an arm only using its label and frame number
    """
    frame = m.open_frame(experiment.path,frame_number+1)
    arm = m.open_frame(experiment.arm_path,frame_number+1)
    length_list = []
    for arm_label in arm_labels[frame_number]:
        roi = arm==(arm_label+1)
        im2,contours,hierarchy = cv2.findContours((roi).astype(np.uint8), 1, 2)
        if len(contours)==1:
            cnt = contours[0]
        else:
            #If find several contours, takes the largest
            widths=[]
            for i in range(len(contours)):
                cnt = contours[i]
                x,y,w,h = cv2.boundingRect(cnt)
                widths.append(w)
            indices = [i for i,wid in enumerate(widths) if wid==max(widths)]
            indices = indices[0]
            cnt = contours[indices]
        
        x,y,w,h = cv2.boundingRect(cnt)
        length_list.append(max(w,h))
    max_size = max(length_list) 
    index_max_size = [i for i,z in enumerate(length_list) if z==max_size]
    return arm_bb(experiment,frame_number,arm_labels[frame_number][index_max_size[0]])

def get_all_arm_length(experiment,arm_labels):
    """ Returns an image of an arm only using its label and frame number
    """
    length_list = []
    for frame_number in range(experiment.n_frames-1):
        arm = m.open_frame(experiment.arm_path,frame_number+1)
    
        for arm_label in arm_labels[frame_number]:
            roi = arm==(arm_label+1)
            im2,contours,hierarchy = cv2.findContours((roi).astype(np.uint8), 1, 2)
            if len(contours)==1:
                cnt = contours[0]
            else:
                #If find several contours, takes the largest
                widths=[]
                for i in range(len(contours)):
                    cnt = contours[i]
                    x,y,w,h = cv2.boundingRect(cnt)
                    widths.append(w)
                indices = [i for i,wid in enumerate(widths) if wid==max(widths)]
                indices = indices[0]
                cnt = contours[indices]
            
            x,y,w,h = cv2.boundingRect(cnt)
            length_list.append(max(w,h))
    return length_list
from skimage.measure import block_reduce
def make_square_image(image,down_factor=2,new_size=64):
    """
    Parameters:
        image: 2-D or 3-D numpy array
        new_size : int, gives the desired maximum dimension
    Returns:
        resized: numpy array"""
    resized = np.zeros( (new_size,new_size),dtype=np.uint8)
    #If downsampling down_factor times makes it not small enough:
    if max(image.shape)>down_factor*new_size:
        if image.shape[1]>image.shape[0]:
            r = float(down_factor*new_size)/ image.shape[1]
            dim = (down_factor*new_size,int(image.shape[0] * r))
        else:
            r = float(down_factor*new_size)/ image.shape[0]
            dim = ( int(image.shape[1] * r),down_factor*new_size)
            
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    img_down = block_reduce(image,block_size=(2,2),func=np.max)
    
    y=new_size/2-img_down.shape[0]/2
    x = new_size/2-img_down.shape[1]/2
        
    resized[y:y+img_down.shape[0],x:x+img_down.shape[1]] = img_down
        
    return resized
#---opening everything
path = os.path.join("..",'data','microglia','RFP1_denoised')
path_centers = os.path.join("..",'data','microglia','1_centers_improved') 
path_arms = os.path.join("..",'data','microglia','1_arms')  

experiment1 = Experiment(path,path_centers,path_arms)
experiment1.load()
#experiment1.track_arms_and_centers()
simple_trajectories1 = loadObject("corrected_normal_exp1")
simple_trajectories1 = filter(lambda x:x[0]!="g",simple_trajectories1)

path2 = os.path.join("..",'data','microglia','8_denoised')
path_centers2 = os.path.join("..",'data','microglia','8_centers') 
path_arms2 = os.path.join("..",'data','microglia','8_arms')  
experiment2 = Experiment(path2,path_centers2,path_arms2)
experiment2.load()
simple_trajectories2 = loadObject("corrected_normal_exp8")
simple_trajectories2 = filter(lambda x:x[0]!="g",simple_trajectories2)

#Actual work

out_arms=get_arms_list_per_frame(experiment1,simple_trajectories1)
out_arm_big = get_longer_arm(experiment1,20,out_arms)
out_resized = resize_image(out_arm_big,new_size=32)
m.si2(out_arm_big,out_resized,"max size : "+str(max(out_arm_big.shape)),"resized")

#ll = get_all_arm_length(experiment1,out_arms ) 
out_rechanged = resize_image(out_resized,max(out_arm_big.shape))
m.si2(out_arm_big,out_rechanged,"original","reconstructed")

#Get the maximum size of an arm in the entire experiment:
mm=[]
for i in range(experiment1.n_frames-1):
    print i
    out = get_longer_arm(experiment1,i,out_arms)
    mm.append(max(out.shape))
print mm
outsiders  =[i for i,x in enumerate(mm) if x>100]
for j in outsiders:
    m.si(get_longer_arm(experiment1,j,out_arms),title=str(mm[j]))

def bb_arm(frame,arm,arm_label):
    """ Returns an image of an arm only using its label and frame number
    """
    
    roi = arm==(arm_label+1)
    im2,contours,hierarchy = cv2.findContours((roi).astype(np.uint8), 1, 2)
    if len(contours)==1:
        cnt = contours[0]
    else:
        #If find several contours, takes the largest
        widths=[]
        for i in range(len(contours)):
            cnt = contours[i]
            x,y,w,h = cv2.boundingRect(cnt)
            widths.append(w)
        indices = [i for i,wid in enumerate(widths) if wid==max(widths)]
        indices = indices[0]
        cnt = contours[indices]
    
    x,y,w,h = cv2.boundingRect(cnt)
    sub_frame = frame[y:y+h,x:x+w]
    sub_roi = roi[y:y+h,x:x+w]
    sub_frame[~sub_roi]=0
    sub_frame*=int(255/np.max(sub_frame))  #To have balanced histograms
    return sub_frame

savepath=os.path.join("..",'data','microglia','arms_for_deep_learning')

def preprocess_arms_dl(experiment,trajectories,savepath,replace=False):
    out_arms=get_arms_list_per_frame(experiment,trajectories)
    glob_nr=0  #To change with names in savepath
    new_size=40
    if not replace:
        elts_already_in = len(glob.glob(savepath+"/*"))
        glob_nr+=elts_already_in
    for frame_nr,arms_list in enumerate(out_arms):
        frame = m.open_frame(experiment.path,frame_nr+1)
        arm = m.open_frame(experiment.arm_path,frame_nr+1)
        for arm_label in arms_list:
            out = bb_arm(frame,arm,arm_label)
            out = make_square_image(out,new_size = new_size)
            cv2.imwrite(os.path.join(savepath,str(glob_nr)+".png"),out)
            glob_nr+=1
preprocess_arms_dl(experiment2,simple_trajectories2,savepath,False)
            