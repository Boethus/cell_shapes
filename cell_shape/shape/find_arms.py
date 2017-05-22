# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:34:26 2017

@author: univ4208
"""

import os
import sys
sys.path.append(os.path.join(".","..","segmentation"))
print os.getcwd()
import methods as m
import numpy as np
import scipy.ndimage as ndi
import cv2
import matplotlib.pyplot as plt
import glob
import platform
import cPickle as pickle

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

def track_bits(save=True):
    """Sets up the trackre for the arms and bodies"""
    track_centers = m.Tracker()
    track_arms = m.Tracker()
    
    for i in range(1,242):
        centers = m.open_frame(path_centers,i)
        arms = m.open_frame(path_arms,i)
        print "Maxima, center and arms:",np.max(arms)
        info_arms = m.FrameInfo(i,arms)
        info_centers = m.FrameInfo(i,centers)
        
        track_centers.info_list.append(info_centers)
        track_arms.info_list.append(info_arms)
    
    
    track_centers.preprocess()
    track_arms.preprocess()
    if save:
        with open('track_centers.pkl','wb') as out:
            pickle.dump(track_centers,out)
        with open('track_arms.pkl','wb') as out:
            pickle.dump(track_arms,out)
        
    return track_centers,track_arms
track_centers,track_arms = track_bits()
#Tracking done between fr 1 and 241 Excluded-> Last frame missing
"""
track_centers=0
with open('track_centers.pkl','rb') as out:
    track_centers = pickle.load(out)
track_arms = 0
with open('track_arms.pkl','rb') as out:
    track_arms = pickle.load(out)"""

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
          
#liste = assign_arm(path_centers,path_arms,0,0)
def check_frame(nr,liste):
    """Check how previous step worked. nr goes between 1 and 241"""          
    corresps = liste[nr-1]
    centers = m.open_frame(path_centers,nr)
    arms = m.open_frame(path_arms,nr)
    arms_out=arms.copy()
    out = centers.copy()
    for u,v in corresps:
        out+= (arms==u).astype(np.uint8) * v
        arms_out[arms==u]=0
    m.si2(centers,out,"centers","arms associated")
    m.si2(arms,arms_out,"arms","arms remaining")
    
def reOrderList(liste):
    """liste is a correspondance list. Each of these correspondance 
    lists assigns a body to a cell arm. This method assigns all of its arms to 
    each cell body"""
    max_body=0
    for arm,body in liste:
        max_body = max(max_body,body)
    #List indexing statrs from 0 but body indexes start from 1 Careful
    ordered_list = []
    for i in range(max_body):
        ordered_list.append([])
    for arm,body in liste:
        l = ordered_list[body-1]
        l.append( arm )
    return ordered_list
list_ordered = map(reOrderList,liste)

class Cell(object):
    """A Cell object consists of a cell body number in a frame and the frame number
    Frame starts with 0, corresponding to the first frame in the stack.
    Body starts with 0, corresponding to the label 1 in the image"""
    def __init__(self,frame,body):
        """A cell is initiated with just the number of a cell body"""
        self.frame_number = frame
        self.body = body
        self.arms=[]

class Trajectory(object):
    def __init__(self,first_cell,beginning=0,end=239):
        """Precises between which and which frame to look for a trajectory.
        first_cell is an instance of the class Cell
        Beginning and end are indexed from 0"""
        self.beginning = beginning
        self.end = end
        self.cells = [first_cell]   #List containing the labels of the tracked cell over frames
        
    def compute_trajectory(self,list_of_arms,body_tracker):
        """frame_info_list has been previously ordered with reOrdrerList"""
        current_cell = self.cells.pop()

        correspondances = body_tracker.correspondance_lists
        
        for fr_nb in range(current_cell.frame_number,self.end):
            label = current_cell.body
            print len(list_of_arms[fr_nb]),"frame number",fr_nb,"label: ",label
            arms = list_of_arms[fr_nb][label]
            current_cell.arms.extend(arms)
            self.cells.append(current_cell)
            
            corresp_in_frame = correspondances[fr_nb]
            next_element = self.find_correspondance(corresp_in_frame,label)
            current_cell = Cell(fr_nb+1,next_element)
            if next_element==-1:
                self.cells.append(current_cell)
                self.end = fr_nb
                return
            
    def find_correspondance(self,corresp_in_frame,nr_center):
        """In a list of correspondances between two frames, looks for the 
        element associated with nr_center in the next frame"""
        corresp = [y for x,y in corresp_in_frame if x==nr_center]
        
        if len(corresp)!=1:
            print "error too  many matches found ", corresp
        return corresp[0]

traj = Trajectory(Cell(2,5),2,30)
traj.compute_trajectory(list_ordered,track_centers)

trajectory_list = []
start_frame = 0  #ie in real life it is frame number3

for label in range(0,len(list_ordered[start_frame])):
    traj = Trajectory(Cell(start_frame,label),start_frame,239)
    traj.compute_trajectory(list_ordered,track_centers)
    trajectory_list.append(traj)
    
size_list = [len(x.cells) for x in trajectory_list]

indices_of_longest_traj = [i for i,j in enumerate(size_list) if j==max(size_list)]

best_trajectory = trajectory_list[indices_of_longest_traj[0]]
cells_bodies_in_best_traj = [x.body for x in best_trajectory.cells]

def show_trajectory(traj,path_im,path_body,path_arm):
    cells_list = traj.cells
    for cell in cells_list:
        frame_number = cell.frame_number+1
        img = m.open_frame(path_im,frame_number)
        body = m.open_frame(path_body,frame_number)
        arms = m.open_frame(path_arm,frame_number)
        mask = (body==(cell.body+1)).astype(np.uint8)*255
        for arm in cell.arms:
            mask+=(arms==(arm+1)).astype(np.uint8)*120
        overlaid = m.cv_overlay_mask2image(mask,img)
        cv2.imshow("Trajectory",overlaid)
        
        cv2.waitKey(0)
    cv2.destroyAllWindows()
show_trajectory(best_trajectory,path,path_centers,path_arms)