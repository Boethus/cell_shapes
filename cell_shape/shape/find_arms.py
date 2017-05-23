# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:25:34 2017

@author: univ4208
"""

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
    
def reorder_list(liste):
    """liste is a correspondance list. Each of these correspondance 
    lists assigns a body to a cell arm. This method assigns all of its arms to 
    each cell body"""
    max_body=0
    for arm,body in liste:
        max_body = max(max_body,body)
    #List indexing statrs from 0 but body indexes start from 1 Careful
    ordered_list = []
    for i in range(max_body+1):
        ordered_list.append([])
    for arm,body in liste:
        ordered_list[body].append( arm )
    return ordered_list

class Cell(object):
    """A Cell object consists of a cell body number in a frame and the frame number
    Frame starts with 0, corresponding to the first frame in the stack.
    Body starts with 0, corresponding to the label 1 in the image"""
    def __init__(self,frame,body):
        """A cell is initiated with just the number of a cell body"""
        self.frame_number = frame
        self.body = body
        self.arms=[]
        self.trajectories = []
        
    def __str__(self):
        """Print operator shows the tuple (frame, cell body)"""
        return str((self.frame_number,self.body))

class Trajectory(object):
    def __init__(self,first_cell):
        """Precises between which and which frame to look for a trajectory.
        first_cell is an instance of the class Cell
        Beginning and end are indexed from 0"""
        self.beginning = first_cell.frame_number
        self.end = 240
        self.cells = [first_cell]   #List containing the labels of the tracked cell over frames
        
    def compute_trajectory(self,list_of_arms,body_tracker):
        """frame_info_list has been previously ordered with reOrdrerList"""
        current_cell = self.cells.pop()

        correspondances = body_tracker.correspondance_lists
        
        for fr_nb in range(current_cell.frame_number,self.end):
            label = current_cell.body
            if label<len(list_of_arms[fr_nb]):
                arms = list_of_arms[fr_nb][label]
            else:
                arms = []
            current_cell.arms.extend(arms)
            self.cells.append(current_cell)
            
            corresp_in_frame = correspondances[fr_nb]
            index_of_element = [i for i,(x,y) in enumerate(corresp_in_frame) if x==label]
            index_of_element = index_of_element[0]
            next_element = body_tracker.correspondance_lists[fr_nb].pop(index_of_element)[1]
            current_cell = Cell(fr_nb+1,next_element)
            if next_element==-1:
                self.cells.append(current_cell)
                self.end = fr_nb
                return


def show_trajectory(traj,path_im,path_body,path_arm,wait=50):
    cells_list = traj.cells
    for cell in cells_list:
        frame_number = cell.frame_number+1
        img = m.open_frame(path_im,frame_number)
        body = m.open_frame(path_body,frame_number)
        arms = m.open_frame(path_arm,frame_number)
        mask = (body==(cell.body+1)).astype(np.uint8)*255
        mask_arms = np.zeros(mask.shape,dtype=np.uint8)
        for arm in cell.arms:
            mask_arms+=(arms==(arm+1)).astype(np.uint8)*255
        overlaid = m.cv_overlay_mask2image(mask,img,"green")
        overlaid = m.cv_overlay_mask2image(mask_arms,overlaid,"red")
        cv2.imshow("Trajectory",overlaid)
        
        cv2.waitKey(wait)
    cv2.destroyAllWindows()

class Experiment(object):
    """Class doing all the job of an expreiment: sgmentation, tracking
    and other improvements"""
    def __init__(self,path,body_path,arm_path):
        """path is the path to the data, body_path to the labeled centers
        and arm to the arm centers"""
        self.path = path
        self.body_path = body_path
        self.arm_path = arm_path
        self.n_frames = len (glob.glob(path+"/*.png"))
        
        #the trackers are initialized afterwards
        self.arm_tracker = []
        self.body_tracker=[]
 
        self.arms_list=[]
        #arms unsure is a list of size n_frames. unsure_arms_list[i] is the list
        #of arms associated with all the possible nucleus they come from
        self.unsure_arms_list=[]
        #Free arms list contains all the arms which are independant
        self.free_arms_list = []
        
        self.trajectories = []
    
        self.apparitions_events = []
        self.disparition_events=[]
    def segmentStack(self):
        """Segments every element in the stack"""
        if not os.path.isdir(self.body_path):
            os.mkdir(self.body_path)
        if not os.path.isdir(self.arm_path):
            os.mkdir(self.arm_path)
        success = True
        for i in range(1,self.n_frames+1):
            label_center,label_arms = self.segment_arms_n_centers(i)
            success = success and cv2.imwrite(os.path.join(self.body_path,str(i)+".png"),label_center)
            success = success and cv2.imwrite(os.path.join(self.arm_path,str(i)+".png"),label_arms)
            print "file ",i
    
    def segment_arms_n_centers(self,nr):
        """Segments frame number nr into bodies and arms """
        centers,arms = m.find_arms(self.path,nr)
        label_arm ,nr_elts_arms = ndi.label(arms)
        lab,nr = ndi.label(centers)
        lab = lab.astype(np.uint8)
        label_arm = m.filter_by_size(label_arm,60)
        return lab,label_arm
    
    def track_arms_and_centers(self,save=True):
        """Sets up the tracker for the arms and bodies"""
        track_centers = m.Tracker()
        track_arms = m.Tracker()
        
        for i in range(1,self.n_frames+1):
            print "track arms and centers iter",i
            centers = m.open_frame(self.body_path,i)
            arms = m.open_frame(self.arm_path,i)
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
        self.arm_tracker = track_arms
        self.body_tracker = track_centers
        return
    
    def load_arms_and_centers(self):
        """Loads the presaved Tracker instances for arms and centers"""
        with open('track_centers.pkl','rb') as out:
            self.body_tracker = pickle.load(out)
        with open('track_arms.pkl','rb') as out:
            self.arm_tracker = pickle.load(out)     
            
    def assign_arm(self):
        """Assigns each arm to a center. Returns a list of lists, 
        for each arm in each time frame"""
        arms_assignment_list=[]
        arm_unsure_list = []
        free_arms_list=[]
        for i in range(1,241):
            print "iteration",i
            arm_center_corresp=[]
            arm_unsure_frame=[]
            free_arms_frame=[]
            centers = m.open_frame(self.body_path,i)
            arms = m.open_frame(self.arm_path,i)
            
            kernel = np.ones((5,5),np.uint8)
            dilation = cv2.dilate(centers,kernel,iterations = 1)
            #Loops over each arm
            for arm_label in range(1,1+np.max(arms)):
                candidates = np.unique(dilation[arms==arm_label])
                
                if candidates[0]==0:
                    candidates= candidates[1:]   #We dont keep zero as it is background
                if candidates.size==0:
                    free_arms_frame.append(arm_label)
                elif candidates.size==1:
                    arm_center_corresp.append( (arm_label-1,candidates[0]-1) )
                else:
                    arm_unsure_frame.append( (arm_label,[x-1 for x in candidates]) )
                    
            arms_assignment_list.append(arm_center_corresp)
            arm_unsure_list.append(arm_unsure_frame)
            free_arms_list.append(free_arms_frame)
            
        self.arms_list = map(reorder_list,arms_assignment_list)
        self.unsure_arms_list = arm_unsure_list
        self.free_arms_list = free_arms_list
    
    def compute_trajectories_in_frame(self,start_frame):
        """Computes all trajectories starting in frame nr"""
        trajectory_list = []
        
        if len(self.arms_list)==0:
            print "Arms list is empty. Please initialize it first using assign_arm()"
            return
        corres= self.body_tracker.correspondance_lists[start_frame][:]
        for label,osef in corres:
            if label!=-1:
                cell_trajectory = Trajectory(Cell(start_frame,label))
                cell_trajectory.compute_trajectory(self.arms_list,self.body_tracker)
                trajectory_list.append(cell_trajectory)
        return trajectory_list
    
    def compute_all_trajectories(self):
        """Calculates all the trajectories, removes them from self.body_tracker
        and adds them in the list self.trajectories"""
        for frame_nb in range(self.n_frames-1):
            trajectory_list = self.compute_trajectories_in_frame(frame_nb)
            self.trajectories.append(trajectory_list)
            
    def nature_event(self,cell_body,index1,index2):
        """Determines the nature of an apparition or disparition event for the
        cell labeled cell_body, in the frame with index1:
        -Transformation body->arm
        -Fusion/division with neighbouring cell"""
        bodies1 = m.open_frame(self.body_path,index1+1) 
        #arms1 = m.open_frame(self.arm_path,index1+1) 
        
        bodies2 = m.open_frame(self.body_path,index2+1)
        arms2 = m.open_frame(self.arm_path,index2+1)
        
        mask = bodies1==(cell_body+1)
        nb_pixels = np.count_nonzero(mask)
        if cell_body==84:
            print "frame",index1+1,"Value we look fr:",cell_body+1
            m.si2(mask,bodies1)
        projection_arms = arms2[mask]    #List of arms pixels overlapping the cell body in previous frame
        counts = np.bincount(projection_arms)
        label_arms = np.argmax(counts)
        #Frequency of pixels in the cell body overlapping an the arm label_arms in frame 2
        frequency_arms = float(counts[label_arms])/nb_pixels
        
        #Same with cell bodies in frame 2:
        projection_bodies = bodies2[mask]
        counts = np.bincount(projection_bodies)
        label_bodies = np.argmax(counts)
        #Frequency of pixels in the cell body overlapping an the arm label_arms in frame 2
        frequency_bodies = float(counts[label_bodies])/nb_pixels
        return (frequency_arms,label_arms-1),(frequency_bodies,label_bodies-1)
    
    def classify_events(self):
        """Loops over each "-1" type events and determines what they correspond to.
        Must be called after compute_all_trajectories.
        Classified in two types: fusion with a body (True) or with an arm(False)"""
        
        #Apparitions
        print "apparitions"
        apparition_list=[]
        for i in range(self.n_frames):
            apparitions = [y for x,y in self.body_tracker.correspondance_lists[i] ]
            print i,self.body_tracker.correspondance_lists[i]
            classifications=[]
            for body_label in apparitions:
                print "find label",body_label,"in frame",i
                (p1,l1),(p2,l2)=self.nature_event(body_label,i+1,i)
                if l1==-1:
                    classifications.append((body_label,l2,True))  #Fusion with a body
                elif l2==-1:
                    classifications.append((body_label,l1,False)) #Fusion with an arm
                elif p2>p1:   #Otherwise get the more likely
                    classifications.append((body_label,l2,True))
                elif p1>=p2:
                    classifications.append((body_label,l1,False))
            apparition_list.append(classifications)
        print "disparitions"
        #Disparitions: Loop over the trajectories
        disparition_list = []
        for i in range(self.n_frames-1):
            #Disparitions is a list of tuples (frame_number,body_disappearing)
            disparitions = [ (x.end,x.cells[-1].body) for x in self.trajectories[i]]
            classifications = []
            for index,body_label in disparitions:
                (p1,l1),(p2,l2)=self.nature_event(body_label,index,index+1)
                if l1==-1:
                    classifications.append((body_label,l2,True))  #Fusion with a body
                elif l2==-1:
                    classifications.append((body_label,l1,False)) #Fusion with an arm
                elif p2>p1:   #Otherwise get the more likely
                    classifications.append((body_label,l2,True))
                elif p1>=p2:
                    classifications.append((body_label,l1,False))
            disparition_list.append(classifications)
        self.apparitions_events = apparition_list
        self.disparition_events = disparition_list
    def save(self):
        name="experiment"
        with open(os.path.join(path,name),'wb') as out:
            pickle.dump(self.__dict__,out)
            
    def load(self):
        name="experiment"
        with open(os.path.join(path,name),'rb') as dataPickle:
            self.__dict__ = pickle.load(dataPickle)
#--------------------------Script-----------------------------------------------
path = os.path.join("..",'data','microglia','RFP1_denoised')
path_centers = os.path.join("..",'data','microglia','1_centers') 
path_arms = os.path.join("..",'data','microglia','1_arms')    

experiment1 = Experiment(path,path_centers,path_arms)

experiment1.track_arms_and_centers()
experiment1.assign_arm()
experiment1.save()
experiment1.load()
experiment1.compute_all_trajectories()
experiment1.classify_events()
print experiment1.nature_event(57,0,1)

"""
trajectory_list = experiment1.compute_trajectories_in_frame(0)
best_trajectory = trajectory_list[6]
cells_bodies_in_best_traj = [x.body for x in best_trajectory.cells]

show_trajectory(best_trajectory,path,path_centers,path_arms,100)
"""

#Frame numbers start from 0. Indexes from 1!!!
disparitions_from_nucl_to_arm = [(0,31),(0,23),(20,71)]    #List of manually annotated disparitions
disp_fusion = [(0,58,57),(2,8,4),(2,40,39)]    #Frame, cell disappearing, cell with which it merges
apparitions_decluster = [(0,45,44),(0,4,4),(0,48,46),(1,85,78),(2,59,61),(20,3,1)]   #Frame, cell appearing, n of cluster it belonged before
apparition_from_arm_to_body = [(0,47),(0,41),(0,29),(20,82)]

apparition_from_arm_to_body_exterior = [(2,85),(20,56)]  #If transformation close to the edge we can forgive
def show_disparitions(corresp):
    return [x+1 for (x,y) in corresp if y==-1]
def show_apparitions(corresp):
    return [y+1 for (x,y) in corresp if x==-1]
def show_app_and_dis(corres,nr):
    print "apparitions:"
    print show_apparitions(corres[nr])
    print "disparitions:"
    print show_disparitions(corres[nr])

def test_prediction(experiment):
    disparitions_from_nucl_to_arm = [(0,31),(0,23),(20,71)]    #List of manually annotated disparitions
    disp_fusion = [(0,58,57),(2,8,4),(2,40,39)]    #Frame, cell disappearing, cell with which it merges
    apparitions_decluster = [(0,45,44),(0,4,4),(0,48,46),(1,85,78),(2,59,61),(20,3,1)]   #Frame, cell appearing, n of cluster it belonged before
    apparition_from_arm_to_body = [(0,47),(0,41),(0,29),(20,82)]
    apparition_from_arm_to_body_exterior = [(2,85),(20,56)]
    
    print "disparitions: from noyau to arm"
    for (u,v) in disparitions_from_nucl_to_arm:
        (p1,l1),(p2,l2)= experiment.nature_event(v-1,u,u+1)
        print "arm:",p1,l1,"body",p2,l2
    print "disparitions: fusion"
    for (u,v,w) in disp_fusion:
        (p1,l1),(p2,l2)= experiment.nature_event(v-1,u,u+1)
        
        print "arm:",p1,l1,"body",p2,l2
        
    print "apparitions:decluster"
    for (u,v,w) in apparitions_decluster:
        (p1,l1),(p2,l2)= experiment.nature_event(v-1,u+1,u)
        print "arm:",p1,l1,"body",p2,l2
        
    print "apparitions:arm t body"
    for (u,v) in apparition_from_arm_to_body:
        (p1,l1),(p2,l2)= experiment.nature_event(v-1,u+1,u)
        print "arm:",p1,l1,"body",p2,l2
test_prediction(experiment1)
im1 = m.open_frame(path_centers,1)
im2 = m.open_frame(path_centers,2)
mask = im1==58