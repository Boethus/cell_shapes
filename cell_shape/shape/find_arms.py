# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:34:26 2017

@author: univ4208
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:30:08 2017

@author: univ4208
"""

import os
import sys
sys.path.append(os.path.join(".","..","segmentation"))
print os.getcwd()
import methods as m
import numpy as np
import scipy.ndimage as ndi
import time
import cv2
import matplotlib.pyplot as plt
import glob
import platform
from PIL import Image

plt.close('all')

path = os.path.join("..",'data','microglia','RFP1_denoised')
def segment_arms_n_centers(path,nr):
    centers,arms = m.find_arms(path,nr)
    label_arm ,nr_elts_arms = ndi.label(arms)
    lab,nr = ndi.label(centers)
    lab = lab.astype(np.uint8)
    label_arm = m.filter_by_size(label_arm,60)
    return lab,label_arm

def segmentStack(path,target_dir_center,target_dir_arms):
    elements = glob.glob(path+"/*RFP.png")
    if platform.system()=='Windows':
        separator="\\"
    else:
        separator="/"
    if not os.path.isdir(target_dir_center):
        os.mkdir(target_dir_center)
    if not os.path.isdir(target_dir_arms):
        os.mkdir(target_dir_arms)
    for i in range(1,len(elements)+1):
        label_center,label_arms = segment_arms_n_centers(path,i)
        cv2.imwrite(os.path.join(target_dir_center,str(i)+".png"),label_center)
        cv2.imwrite(os.path.join(target_dir_arms,str(i)+".png"),label_arms)
        print "file ",i
  
path_centers = os.path.join("..",'data','microglia','1_centers') 
path_arms = os.path.join("..",'data','microglia','1_arms')     
#segmentStack(path,path_centers,path_arms)

#track_centers = m.Tracker()
track_arms = m.Tracker()

for i in range(1,241):
    #centers = m.open_frame(path_centers,i)
    arms = m.open_frame(path_arms,i)
    print "Maxima, center and arms:",np.max(arms)
    info_arms = m.FrameInfo(i,arms)
    #info_centers = m.FrameInfo(i,centers)
    
    #track_centers.info_list.append(info_centers)
    track_arms.info_list.append(info_arms)


#track_centers.preprocess(1,240)
track_arms.preprocess(1,240)
import cPickle as pickle
"""
with open('track_centers.pkl','wb') as out:
    pickle.dump(track_centers,out)"""
with open('track_arms.pkl','wb') as out:
    pickle.dump(track_arms,out)

def assign_arm(path_centers,path_arms,center_tracker,arm_tracker):
    """Assigns each arm to a center. Returns a list of lists, 
    for each arm in each time frame"""
    arms_assignment_list=[]
    for i in range(1,241):
        print "iteration",i
        arm_center_corresp=[]
        centers = m.open_frame(path_centers,i)
        arms = m.open_frame(path_arms,i)
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(centers,kernel,iterations = 1)
        #Loops over each arm
        for arm_label in range(1,1+np.max(arms)):
            candidates = np.unique(dilation[arms==arm_label])
            if candidates[0]==0:
                candidates= candidates[1:]   #We dont keep zero as it is background
         
            if candidates.size==1:
                arm_center_corresp.append( (arm_label,candidates[0]) )
        arms_assignment_list.append(arm_center_corresp)
    return arms_assignment_list
          
liste = assign_arm(path_centers,path_arms,0,0)

def check_frame(nr,liste):
    """Check how previous step worked. nr goes between 1 and 241"""          
    corresps = liste[nr-1]
    centers = m.open_frame(path_centers,i)
    arms = m.open_frame(path_arms,i)
    arms_out=arms.copy()
    out = centers.copy()
    for u,v in corresps:
        out+= (arms==u).astype(np.uint8) * v
        arms_out[arms==u]=0
    m.si2(centers,out,"centers","arms associated")
    m.si2(arms,arms_out,"arms","arms remaining")
    
check_frame(5,liste) 

"""
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(lab,kernel,iterations = 2)

sizes = [np.count_nonzero(arms==x) for x in range(1,np.max(arms)+1) ]

arms_remaining = np.unique(label_arm[dilation==0])
arms_to_merge = [x for x in range(1,nr_elts_arms+1) if not x in arms_remaining]
for elt in arms_to_merge:
    mask = label_arm==elt
    value = dilation[mask][0]
    label_arm[mask]=0
    lab += mask.astype(np.uint8)*value
m.si2(lab,label_arm,"adding everything")
label_arm = m.filter_by_size(label_arm,60)
m.si(label_arm,"size filtered")
"""