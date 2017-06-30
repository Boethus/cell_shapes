# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:25:34 2017

@author: univ4208
"""


import os
import sys
sys.path.append(os.path.join(".","..","segmentation"))
import dahlia_methods as m
import numpy as np
import scipy.ndimage as ndi
import cv2
import matplotlib.pyplot as plt
import glob
from screeninfo import get_monitors
import cPickle as pickle
from sklearn.neighbors import KNeighborsClassifier
import copy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random

#import process_trajectories as pt


monitor = get_monitors()[0]
width_monitor = monitor.width/2
height_monitor = monitor.height/2

plt.close('all')

def number_elts_in_folder(folder):
    l = glob.glob(folder+"/*.png")
    return len(l)
def reorder_list(liste,max_body):
    """liste is a correspondance list. Each of these correspondance 
    lists assigns a body to a cell arm. This method assigns all of its arms to 
    each cell body
    max_body is the maximum index of a body in the frame of interest"""
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
            next_element =  [y for x,y in corresp_in_frame if x==label][0]
            current_cell = Cell(fr_nb+1,next_element)
            if next_element==-1:
                self.end = fr_nb
                return
    def __eq__(self,other_traj):
        """overloading operator ="""
        return (self.beginning==other_traj.beginning and self.cells[0].body == other_traj.cells[0].body)

    def show(self,experiment,wait=50):
        """Displays on screen the tempral trajectory in experiment"""
        if sys.platform == 'linux2':
            width = 1366
            height= 768
        else:
            monitor = get_monitors()[0]
            width = monitor.width
            height = monitor.height
        cv2.namedWindow("Trajectory", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Trajectory', width, height)
        
        path_im = experiment.path
        path_body = experiment.body_path
        path_arm = experiment.arm_path
        cells_list = self.cells
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
        cv2.destroyWindow("Trajectory")

class Complex_Trajectory(list):
    """"a Complex trajectory is a sequence of Trajectories and arms trajectories
    corresponding to the same cell"""
    def __init__(self,*args):
        list.__init__(self,*args)
        
    def show(self,experiment,wait=50):
        """Displays this complex trajectory"""
        path_im = experiment.path
        path_body = experiment.body_path
        path_arm = experiment.arm_path
        previous = 'trajectory'
        
        monitor = get_monitors()[0]
        width = monitor.width
        height = monitor.height
        cv2.namedWindow("Complex Trajectory", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Complex Trajectory', width, height)
        for i,elt in enumerate(self):
            if type(elt)==tuple:
                #raccomodation
                if i>0:
                    if type(self[i-1])==tuple:
                        previous='tuple'
                    else:
                        previous='trajectory'
                
                if previous=='trajectory':
                    #looks after
                    stop_frame = 239
                    if i!=len(self)-1:
                        index_next1,index_next_2,_ = self[i+1]
                        next_traj = experiment.trajectories[index_next1][index_next_2]
                        stop_frame = next_traj.beginning
                    
                    last_frame = self[i-1].end
                    next_frame = last_frame+1
                    label = elt[2]
                    while next_frame<stop_frame and experiment.arm_tracker.next_cell(next_frame,label)!=-1:
                         
                         img = m.open_frame(path_im,next_frame+1)
                         arms = m.open_frame(path_arm,next_frame+1)
                         mask = (arms==(label+1)).astype(np.uint8)*255
                         label = experiment.arm_tracker.next_cell(next_frame,label)
                         next_frame+=1
                         overlaid = m.cv_overlay_mask2image(mask,img,"red")
                         cv2.imshow("Complex Trajectory",overlaid)
                         cv2.waitKey(wait)
            else:
                cells_list = elt.cells        
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
                    cv2.imshow("Complex Trajectory",overlaid)
                    
                    cv2.waitKey(wait)
                    
        cv2.destroyWindow("Complex Trajectory")
    
def show_trajectory(traj,experiment,wait=50):
    """Displays on screen a temporal trajectory traj."""
    width = width_monitor
    height = height_monitor
    cv2.namedWindow("Trajectory", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trajectory', width, height)
    
    path_im = experiment.path
    path_body = experiment.body_path
    path_arm = experiment.arm_path
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
    #cv2.destroyAllWindows()

def show_complex_trajectory(comp_traj,experiment,wait=50):
    """Displays a complex trajectory"""
    path_im = experiment.path
    path_body = experiment.body_path
    path_arm = experiment.arm_path
    previous = 'trajectory'
    
    width = width_monitor
    height = height_monitor
    cv2.namedWindow("Trajectory", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trajectory', width, height)
    for i,elt in enumerate(comp_traj):
        if type(elt)==tuple:
            #raccomodation
            if i>0:
                if type(comp_traj[i-1])==tuple:
                    previous='tuple'
                else:
                    previous='trajectory'
            
            if previous=='trajectory':
                #looks after
                stop_frame = 239
                if i!=len(comp_traj)-1:
                    index_next1,index_next_2,_ = comp_traj[i+1]
                    next_traj = experiment.trajectories[index_next1][index_next_2]
                    stop_frame = next_traj.beginning
                
                last_frame = comp_traj[i-1].end
                next_frame = last_frame+1
                label = elt[2]
                while next_frame<stop_frame and experiment.arm_tracker.next_cell(next_frame,label)!=-1:
                     
                     img = m.open_frame(path_im,next_frame+1)
                     arms = m.open_frame(path_arm,next_frame+1)
                     mask = (arms==(label+1)).astype(np.uint8)*255
                     label = experiment.arm_tracker.next_cell(next_frame,label)
                     next_frame+=1
                     overlaid = m.cv_overlay_mask2image(mask,img,"red")
                     cv2.imshow("Trajectory",overlaid)
                     cv2.waitKey(wait)
        else:
            cells_list = elt.cells        
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
                
    #cv2.destroyAllWindows()
    

#-------Morphological operations to tell nuclei apart-----------
def morphology_split(frame,label,number,max_labels_in_frame):
    """Splits the object with label in frame in number pieces"""
    if number<=1:
        return
    im2,contours,hierarchy = cv2.findContours((frame==label).astype(np.uint8), 1, 2)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    
    sub_frame = (frame[y:y+h,x:x+w]==label).astype(np.uint8)
    kernel = np.ones((3,3),np.uint8)
    erosion = sub_frame.copy()
    #For machine learning
    neigh = KNeighborsClassifier(n_neighbors=1)
    lab = 0
    success = False
    for i in range(15):
        erosion = cv2.erode(erosion,kernel,iterations = 1)
        lab,nr = ndi.label(erosion)
        if nr==number:
            #Correct separation
            n=0 #Current index
            n_training_values = np.count_nonzero(lab)
            X = np.zeros( (n_training_values,2))
            Y = np.zeros( n_training_values)
            for i in range(nr):
                    xlab,ylab = np.where(lab==i+1)
                    ref_pts = np.concatenate((xlab.reshape(-1,1),ylab.reshape(-1,1)),axis=1)
                    n_points_added = ref_pts.shape[0]
                    X[n:n+n_points_added,:] = ref_pts
                    Y[n:n+n_points_added] = i
                    n+=n_points_added
            neigh.fit(X,Y)
            to_predict_x,to_predict_y = np.where(np.logical_and(lab==0,sub_frame>0))
            to_predict = np.concatenate((to_predict_x.reshape(-1,1),to_predict_y.reshape(-1,1)),axis=1)
            results = neigh.predict(to_predict)+1
            lab[to_predict_x,to_predict_y] = results
            success=True
            break
    if success:
        #Replace by new labels
        lab[lab==1]=label
        for i in range(1,number):
            lab[lab==i+1] = max_labels_in_frame+i
        
        lab[frame[y:y+h,x:x+w]!=label]=frame[y:y+h,x:x+w][frame[y:y+h,x:x+w]!=label]
        frame.setflags(write=1)
        frame[y:y+h,x:x+w]=lab
    return success
        
def redefine_labels(path_centers):
    """In case an index disappeared"""
    for i in range(0,241):
        print i
        frame = m.open_frame(path_centers,i+1)
        elts_in_frame = np.unique(frame)
        missing = []
        for j in range(np.max(frame)):
            if not j in elts_in_frame:
                print "element missing"
                missing.append(j)
        for elts in missing:
            mm=np.max(frame)
            frame[frame==mm]=elts
        cv2.imwrite(os.path.join(path_centers,str(i+1)+".png"),frame)
        
#--------------Complex trajectories----------------------------------------------
def raccomodate_after_to_before(index1,index2,number,before_raccomodations,experiment):
    """returns the index in before_raccomodation corresponding to the tuple
    (frame,label,number) in after_raccomodation"""
    traj = experiment.trajectories[index1][index2]
    nframe = traj.end
    next_label = number
    for i in range(nframe+1,239):
        candidates = [j for j,(x,y,z) in enumerate(before_raccomodations) if z==next_label and x==i+1] 
        next_label = experiment.arm_tracker.next_cell(i,next_label)
        if next_label==-1:
            return -1
        if len(candidates)==1:
                return candidates[0]
    return -1

#list of raco : after,before
#eah consist of (index_traj_1,index_traj_2,label_of_new_element

def assemble_complex_trajectories(complex_trajectory,z,list_of_raccom,raco_before,raco_after,experiment):
    """A complex trajectory is a list of [traj, raccomodation_after,raccomodation_before,traj2,...]
    z is the index in the list of complex trajectories corresponding to an apparition/reapparition."""
    l,k = list_of_raccom.pop(z)
    traj1 = experiment.trajectories[raco_before[k][0]][raco_before[k][1]]
    if len(complex_trajectory)==0:
        traj0 = experiment.trajectories[raco_after[l][0]][raco_after[l][1]]
        complex_trajectory.extend([traj0,raco_after[l],raco_before[k],traj1])
    else:
        complex_trajectory.extend([raco_after[l],raco_before[k],traj1])
    
    for l_prime, (i1,i2,label) in enumerate(raco_after):
        #Find if there is a racomodation after
        if traj1==experiment.trajectories[i1][i2]:
            #Find if this raccomoadtion can itself be raccomodated
            candidates = [i for i,(x,y) in enumerate(list_of_raccom) if x==l_prime]
            if len(candidates)==1:
                assemble_complex_trajectories(complex_trajectory,candidates[0],list_of_raccom,raco_before,raco_after,experiment)
            else:
                complex_trajectory.append((i1,i2,label))
                return
    return

def get_complex_trajectories(experiment):
    """Wraps up the different methods to get the complex trajectories in experiment"""
    raco_before,raco_after=experiment.trajectory_racomodation()
    
    list_of_raccomodated = []
    for i,(frame,label,number) in enumerate(raco_after):
        index = raccomodate_after_to_before(frame,label,number,raco_before,experiment)
        if index!=-1:
            list_of_raccomodated.append((i,index))
    
    list_of_racc_disposable = list_of_raccomodated[:]
    total_merged_trajectories = []  #List of complex trajectories
    for i in range(len(list_of_racc_disposable)):
        comp_traj = []
        if i>=len(list_of_racc_disposable):
            break
        assemble_complex_trajectories(comp_traj,i,list_of_racc_disposable,raco_before,raco_after,experiment)
        total_merged_trajectories.append(comp_traj)
    return total_merged_trajectories
#----------------------Assign arm to correct nucleus-----------
def find_arm_in_frame(frame,label,experiment):
    """for arm with number label in frame, determines if it is bound to one cell
    only, to several or if it is free"""
    
    type_of_arm = "unsure"
    
    #1: check if it is bound to only one frame
    list_of_arms = experiment.arms_list[frame]
    label_in_list = [i for i,x in enumerate(list_of_arms) if label in x]
    if len(label_in_list)>1:
        print "error too many matches in find_arm_in_frame"
    if len(label_in_list)==1:
        cell_body = label_in_list[0]
        type_of_arm = "sure"
        return type_of_arm,cell_body
    #2: check if it is a free arm
    if label in experiment.free_arms_list[frame]:
        type_of_arm = "free"
        return type_of_arm,-1
    #3 Verifies that it is in unsure list
    candidates_unsure = [(x, body_list) for x,body_list in experiment.unsure_arms_list[frame] if x==label]
    if len(candidates_unsure)==1:
        type_of_arm = "unsure"
        return type_of_arm,-1
    raise ValueError('Arm not found')

def assign_unsure_arm(frame,label,experiment):
    """Gets the trajectory of each arm in unsure_list and if """
    verbose = False
    frame_arr,labels_arr = experiment.find_trajectory(frame,label,arm=True)
    types = []   #contains the type of each earm tracked: sure,unsure,free
    bodies = []
    for u,v in zip(frame_arr,labels_arr):
        arm_type,cell_body = find_arm_in_frame(u,v,experiment)
        types.append(arm_type)
        bodies.append(cell_body)
    if verbose:
        print types
    if "free" in types:
        #If it is free at one point we can't say yet
        pass
        #return
    if "sure" in types:
        frame_arr = np.asarray(frame_arr)
        labels_arr = np.asarray(labels_arr)        
        where_sure = np.asarray([x=="sure" for x in types])
        frames_sure = frame_arr[where_sure]
        bodies_sure = np.asarray(bodies)[where_sure]
        if verbose:
            print frames_sure
            print bodies_sure
        where_unsure = np.asarray([x=="unsure" for x in types])
        frames_unsure = frame_arr[where_unsure]
        labels_unsure = labels_arr[where_unsure]
        #For each frame unsure, gets the closest sure frame
        #Then looks for the corresponding cell body
        for fr,lab in zip(frames_unsure,labels_unsure):
            closest_sure_frame = np.abs(frames_sure-fr)
            closest_sure_frame = frames_sure[closest_sure_frame==np.min(closest_sure_frame)]
            closest_sure_frame = closest_sure_frame[0]
            closest_body = bodies_sure[frames_sure==closest_sure_frame][0]
            if verbose:
                print "closest association:",closest_sure_frame,closest_body
            indices,cells = experiment.find_trajectory(closest_sure_frame,closest_body,arm=False)
            indices = np.asarray(indices)
            cells = np.asarray(cells)
            #If trajectory of body goes dar enough 
            if fr in indices:
                new_label = cells[indices==fr][0]
                candidates = [(j,list_joints) for j,(labb,list_joints) in enumerate(experiment.unsure_arms_list[fr]) if labb==lab]
                if len(candidates)!=1:
                    raise IndexError('cant find an appropriate number of candidates')
                index,list_to_check = candidates[0]
                if not new_label in list_to_check:
                    print "no comprendo la correspondencia"
                else:
                    experiment.unsure_arms_list[fr].pop(index)
                    print "in frame",fr,"associating",lab,"to body",new_label
                    experiment.arms_list[fr][new_label].append(lab)
                    
def process_unsure_arms(experiment):
    """Assigns the unsure arms based on temporal information"""
    for i in range(experiment.n_frames):
        list_of_arms = [x for x,y in experiment.unsure_arms_list[i]]
        for arm in list_of_arms:
            assign_unsure_arm(i,arm,experiment)
            
#--------------Class Experiment--------------------------------------------------
class Experiment(object):
    """Class doing all the job of an expreiment: sgmentation, tracking
    and other improvements"""
    def __init__(self,path,body_path=None,arm_path=None):
        """path is the path to the data, body_path to the labeled centers
        and arm to the arm centers"""
        self.path = path
        self.body_path = body_path
        self.arm_path = arm_path
        self.n_frames = len (glob.glob(path+"/*.png"))
        
        #the trackers are initialized afterwards
        self.arm_tracker = []
        self.body_tracker=[]
 
        #arms_list[i][j] gives the arms of cell j in frame i
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
            print "Segmenting image ",i
    
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
        for i in range(1,self.n_frames+1):
            print "iteration",i
            arm_center_corresp=[]
            arm_unsure_frame=[]
            free_arms_frame=[]
            centers = m.open_frame(self.body_path,i)
            arms = m.open_frame(self.arm_path,i)
            max_centers=np.max(centers)
            
            kernel = np.ones((5,5),np.uint8)
            dilation = cv2.dilate(centers,kernel,iterations = 1)
            #Loops over each arm
            for arm_label in range(1,1+np.max(arms)):
                candidates = np.unique(dilation[arms==arm_label])
                
                if candidates[0]==0:
                    candidates= candidates[1:]   #We dont keep zero as it is background
                if candidates.size==0:
                    free_arms_frame.append(arm_label-1)
                elif candidates.size==1:
                    arm_center_corresp.append( (arm_label-1,candidates[0]-1) )
                else:
                    arm_unsure_frame.append( (arm_label-1,[x-1 for x in candidates]) )
                    
            arms_assignment_list.append(reorder_list(arm_center_corresp,max_centers) )
            arm_unsure_list.append(arm_unsure_frame)
            free_arms_list.append(free_arms_frame)
            
        self.arms_list = arms_assignment_list
        self.unsure_arms_list = arm_unsure_list
        self.free_arms_list = free_arms_list
    
    def compute_trajectories_in_frame(self,start_frame):
        """Computes all trajectories starting in frame nr"""
        trajectory_list = []
        
        if len(self.arms_list)==0:
            print "Arms list is empty. Please initialize it first using assign_arm()"
            return
        if start_frame==0:
            corres= self.body_tracker.correspondance_lists[start_frame][:]
            for label,osef in corres:
                if label!=-1:
                    cell_trajectory = Trajectory(Cell(start_frame,label))
                    cell_trajectory.compute_trajectory(self.arms_list,self.body_tracker)
                    trajectory_list.append(cell_trajectory)
        else:
             corres= self.body_tracker.correspondance_lists[start_frame-1][:]   #Just loop for he cells appearing
             for appear,label in corres:
                    if appear==-1:
                        cell_trajectory = Trajectory(Cell(start_frame,label))
                        cell_trajectory.compute_trajectory(self.arms_list,self.body_tracker)
                        trajectory_list.append(cell_trajectory)
        return trajectory_list
    
    def compute_all_trajectories(self):
        """Calculates all the trajectories, removes them from self.body_tracker
        and adds them in the list self.trajectories"""
        self.trajectories = []
        for frame_nb in range(self.n_frames-1):
            trajectory_list = self.compute_trajectories_in_frame(frame_nb)
            self.trajectories.append(trajectory_list)
            
    def nature_event(self,cell_body,index1,index2):
        """Determines the nature of an apparition or disparition event for the
        cell labeled cell_body, in the frame with index1:
        -Transformation body->arm
        -Fusion/division with neighbouring cell"""
        bodies1 = m.open_frame(self.body_path,index1+1) 
        
        bodies2 = m.open_frame(self.body_path,index2+1)
        arms2 = m.open_frame(self.arm_path,index2+1)
        
        mask = bodies1==(cell_body+1)
        nb_pixels = np.count_nonzero(mask)
        projection_arms = arms2[mask]    #List of arms pixels overlapping the cell body in previous frame
        counts_arms = np.bincount(projection_arms)
        label_arms = np.argmax(counts_arms)
        
        #Same with cell bodies in frame 2:
        projection_bodies = bodies2[mask]
        counts_bodies = np.bincount(projection_bodies)
        label_bodies = np.argmax(counts_bodies)
        
        if label_arms==0 and label_bodies==0:
            counts_arms[label_arms]=0
            counts_bodies[label_bodies]=0
            label_arms = np.argmax(counts_arms)
            label_bodies = np.argmax(counts_bodies)
        #Frequency of pixels in the cell body overlapping an the arm label_arms in frame 2
        frequency_arms = float(counts_arms[label_arms])/nb_pixels
        frequency_bodies = float(counts_bodies[label_bodies])/nb_pixels
        return (frequency_arms,label_arms-1),(frequency_bodies,label_bodies-1)
    
    def classify_events(self):
        """Loops over each "-1" type events and determines what they correspond to.
        Must be called after compute_all_trajectories.
        Classified in two types: fusion with a body (True) or with an arm(False)
        An apparition or disparition event is a tuple, made of:
        (label_involved,label_of_new_item,isNewItemBody)"""
        
        #Apparitions
        print "apparitions"
        apparition_list=[]
        for i in range(self.n_frames-1):
            apparitions = [y for x,y in self.body_tracker.correspondance_lists[i] if x==-1 ]
            classifications=[]
            for body_label in apparitions:
                (p1,l1),(p2,l2)=self.nature_event(body_label,i+1,i)
                if p1<0.1 and p2<0.1:
                    pass    #if probas too low do nothing
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
            #Dispqrition involvesan object being there in a frame and not there
            #in the next one so we need to rule out the last element
            disparitions = [x for x,y in self.body_tracker.correspondance_lists[i] if y==-1 ]
            classifications = []
            for body_label in disparitions:
                (p1,l1),(p2,l2)=self.nature_event(body_label,i,i+1)
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
        
    def split_merged_bodies(self):
        """In the case of a merging event, ie two bodies merge into one, looks for the 
        disparition and tracks the corresponding nucleus until a cell reappears.
        Works only if there are two nuclei in collision with each other"""
        max_nb_objects = max([x.n_objects for x in self.body_tracker.info_list])
        #Clustering matrix contains for each time frame and each label the number of cells
        print "split"
        clustering_matrix = np.zeros((self.n_frames,max_nb_objects))
        for frame in range(self.n_frames-1):
            disparitions = self.disparition_events[frame]
            apparitions = self.apparitions_events[frame]
            #Iterates for each disparition in frame
            for label,new_object_label,is_body in disparitions:
                #Case nucleus-nucleus fusion
                if is_body:
                    #Iterate over the next frames to find a corresponding apparition event
                    
                    label_after = new_object_label
                    nb_cells_merging =  clustering_matrix[frame][label] +1
                    for i in range(frame+1,self.n_frames-1):
                        if label_after==-1:
                            break
                        clustering_matrix[i][label_after]+=nb_cells_merging
                        label_after = self.body_tracker.next_cell(i,label_after)
            for label,object_losing_label,is_body in apparitions:
                #Case nucleus-nucleus fusion
                if is_body and object_losing_label!=-1:
                    #Iterate over the next frames to find a corresponding apparition event
                    
                    label_after = self.body_tracker.next_cell(frame,object_losing_label)
                    #Nb apparitions after this one, meaning ow many cells are in the appearing one
                    
                    for i in range(frame+1,self.n_frames-1):
                        if label_after==-1:
                            break
                        if clustering_matrix[i][label_after]>0:
                            clustering_matrix[i][label_after]-=1
                        label_after = self.body_tracker.next_cell(i,label_after)
        #Processing the clustering matrix
        new_path = os.path.join("..",'data','microglia','1_centers_improved') 
        for nframe in range(self.n_frames):
            print "rewriting frame",nframe+1
            #Spearate each label
            frame = m.open_frame(self.body_path,nframe+1)
            for labels in range(clustering_matrix[nframe,:].shape[0]):
                number = clustering_matrix[nframe,labels]
                number=int(number)
                max_labels_in_frame = np.max(frame)
                if morphology_split(frame,labels+1,number+1,max_labels_in_frame):
                    self.body_tracker.info_list[nframe].n_objects+=1
                    
            cv2.imwrite(os.path.join(new_path,str(nframe+1)+".png"),frame)
        return clustering_matrix
    
    def find_trajectory(self,frame,label,arm=True):
        """Given an arm or a cell, returns all its indexes over different frames"""
        before_indices = []
        before_cells = []
        prev_cell = label
        for i in range(frame):
            if arm:
                prev_cell = self.arm_tracker.prev_cell(frame-i,prev_cell)
            else:
                prev_cell = self.body_tracker.prev_cell(frame-i,prev_cell)
            if prev_cell==-1:
                break
            before_indices.append(frame-i-1)
            before_cells.append(prev_cell)
        before_cells.reverse()
        before_indices.reverse()
        
        after_indices=[]
        after_cells = []
        next_cell = label
        for i in range(frame,self.n_frames-1):
            if arm:
                next_cell = self.arm_tracker.next_cell(i,next_cell)
            else:
                next_cell = self.body_tracker.next_cell(i,next_cell)
            if next_cell==-1:
                break
            after_indices.append(i+1)
            after_cells.append(next_cell)
        before_cells.append(label)
        before_cells.extend(after_cells)
        before_indices.append(frame)
        before_indices.extend(after_indices)
        return before_indices,before_cells
    
    def trajectory_racomodation(self):
        """if a nucleus disappears to ive birth to an arm. We racomodate these here"""
        raccomodations = []
        raccomodations_before = []
        for i in range(self.n_frames-1):
            traj_frame = self.trajectories[i]
            for index,traj in enumerate(traj_frame):
                endFrame = traj.end
                last_cell = traj.cells[-1].body
                if endFrame<self.n_frames-1:
                    disparitions = self.disparition_events[endFrame]
                    corresponding_disparition = [new_label for (label,new_label,isBody) in disparitions if label==last_cell and not isBody]
                    #We only consider the case where isBody is Flase, ie body disappears for an arm.
                    if len(corresponding_disparition)==1:
                        new_arm = corresponding_disparition[0]
                        raccomodations.append((i,index,new_arm))   #raccomodates self.trajectories[i][index]  to new_arm
                first_cell = traj.cells[0].body
                if i>0:
                    apparitions = self.apparitions_events[i-1]
                    corresponding_apparition = [new_label for (label,new_label,isBody) in apparitions if label==first_cell and not isBody]
                    if len(corresponding_apparition)==1:
                        prev_arm = corresponding_apparition[0]
                        raccomodations_before.append(( (i),index,prev_arm ))
        return raccomodations_before, raccomodations           
            
    def save(self):
        name="experiment"
        with open(os.path.join(self.path,name),'wb') as out:
            pickle.dump(self.__dict__,out)

    def load(self):
        print "loading trois petits points"
        name="experiment"
        with open(os.path.join(self.path,name),'rb') as dataPickle:
            self.__dict__ = pickle.load(dataPickle)
    def process_from_scratch(self):
        """used if need to process a whole new set of data"""
        print "stack segmentation"
        self.segmentStack()
        print "track arms and centers..."
        self.track_arms_and_centers()
        print "arms assignment..."
        self.assign_arm()
        print "processing unsure arms"
        process_unsure_arms(self)
        print "compute trajectories"
        self.compute_all_trajectories()
        self.classify_events()
        self.save()
        
    def process_tracking(self):
        """Does all the tracking from a """
        print "track arms and centers..."
        self.track_arms_and_centers()
        print "arms assignment..."
        self.assign_arm()
        print "processing unsure arms"
        process_unsure_arms(self)
        print "compute trajectories"
        self.compute_all_trajectories()
        self.classify_events()
        self.save()

def classify_complex_trajectory(traj,experiment):
    """Displays traj and prompts the user about what to do"""
    show_complex_trajectory(traj[0],experiment,50)
    possible_answers = ['r','w','t','m','q','e']
    inp = ''
    while(not inp in possible_answers):
        inp= raw_input("""how would you classify this image: ramified, withdrawal, transitional,
                      motile, error (r/w/t/m/e)? Press q to see the sequence again\n""")
    
    if inp=='q':
        inp = classify_complex_trajectory(traj,experiment)
    return inp
        
def saveClassif(classification):
    with open('classification_results.pkl','wb') as out:
        pickle.dump(classification,out)
def saveObject(name,obj):
    with open(name,'wb') as out:
        pickle.dump(obj,out) 
def loadObject(name):
    with open(name,'rb') as out:
        obj = pickle.load(out)
    return obj
def loadClassif():
    with open('classification_results.pkl','rb') as out:
        classification = pickle.load(out)
    return classification

#Find all trajectories which are not complex
def find_simple_trajectories(experiment,complex_trajectories):
#1: find all trajectories involved in a complex one
    trajectories_in_complex = []
    for comp_traj in complex_trajectories:
        for elt in comp_traj:
            if type(elt)!=tuple:  #Case it is a trajectory
                trajectories_in_complex.append(elt)
    
    trajectories_remaining = []  #This list will contain the trajectories which have not been processed yet
    for traj_list in experiment.trajectories:
        for traj in traj_list:
            if not traj in trajectories_in_complex:
                trajectories_remaining.append(traj)
    return trajectories_remaining

#Untested
def gaussian_score_list(path,path_bodies,nr):
    """computes the gaussian score for frame number nr in path.
    returns the maximum of this score in each segment"""
    image = m.open_frame(path,nr)
    bodies = m.open_frame(path_bodies,nr)
    bodies_proba = np.zeros(bodies.shape)
    scores=[]
    score_frame = m.gaussian_proba_map(image)
    
    for i in range(np.max(bodies)):
        score_i = np.max(score_frame[bodies==i+1])
        scores.append(score_i)
        bodies_proba[bodies==i+1] = score_i
    return scores

def gs_score_experiment(experiment):
    """computes the gaussian score for every element of 
    every frame of an experiment"""
    scores_list=[]
    for i in range(experiment.n_frames):
        print "computation of gaussian scores in frame",i+1
        path=experiment.path
        body_path = experiment.body_path
        scores = gaussian_score_list(path,body_path,i+1)
        scores_list.append(scores)
    return scores_list
#scores_list = loadObject("gs_scores_exp8.pkl")
def gs_score_trajectory(experiment,trajectory,score_list):
    """computes the gaussian score of a trajectory in experiment,
    using the list of gaussian scores in every frame score_list"""
    frame = trajectory.beginning
    traj_score=[]
    for cell in trajectory.cells:
        print cell.body,frame,len(score_list[frame])
        score = score_list[frame][cell.body]
        traj_score.append(score)
        frame+=1
    return np.mean(traj_score)

def gs_score_each_traj(experiment,simple_trajectories,score_list):
    """given a list of simple trajectories, returns a list
    with their mean gaussian scores"""
    results=[]
    i=0

    for traj in simple_trajectories:
        print i
        i+=1
        score = gs_score_trajectory(experiment,traj,score_list)
        results.append(score)
    return results
#--------------------------See each trajectory-----------------------------------------------
def classify_trajectory2(traj,experiment):
    """Displays traj and prompts the user about what to do"""
    show_trajectory(traj,experiment,50)
    possible_answers = ['r','m','e','a']
    inp = ''
    while(not inp in possible_answers):
        inp= raw_input("""how would you classify this image: relevant,maybe, error (r/m/e)? Press a to see the sequence again\n""")
    
    if inp=='a':
        inp = classify_trajectory2(traj,experiment)
    return inp

def classify_simple_trajectories(trajectories,experiment):
    classifications_normal = []
    
    for i in range(len(trajectories)):
        print "index",i
        #cv2.destroyAllWindows()
        traj = trajectories[i]
        inp= classify_trajectory2(traj,experiment)
        if inp!='e' and inp !='a':    #otherwise it is an error
            classifications_normal.append((inp,traj))
    return classifications_normal

def w_trajectory_classification(experiment,name):
    """Wraps up all methods of trajectory classification.
    Parameters:
        experiment: instance of the class Experiment already processed
        name: the name with which the classification will be saved"""
    
    if os.path.isdir(name):
        print "Careful name already exsits. Proceed?"
    file_name = name.split("\\")[-1]
    path = name[:-len(file_name)]
    tmp_dir = os.path.join(path,"tmp_"+file_name)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)   #Store intermediate results
    trajectories = find_simple_trajectories(experiment,[])
    classifs=[]
    
    for i in range(5):
        beg = int(float(len(trajectories)*i)/5)
        end=int(float(len(trajectories)*(i+1))/5)
        print "Chunk",i
        if i==4:
            end = len(trajectories)-1
        clf1 = classify_simple_trajectories(trajectories[beg:end],experiment)
        saveObject( os.path.join(tmp_dir,name+str(i)+".pkl") ,clf1)
        classifs.extend(clf1)
    saveObject(name+".pkl",classifs)
    
def get_from_folder(folder_name):
    """gets all the bits of experiment in a foler and gathers them together"""
    l=[]
    path = os.path.join(folder_name,folder_name[4:])
    for i in range(5):
        tmp_l = loadObject(path+str(i)+".pkl")
        l.extend(tmp_l)
    return l


#-------------Classification--------------------------------
def distribution_vector(list_of_elts):
    """returns a 3*1 vector containing [mean,min,max] of list"""
    out = np.zeros(3)
    if len(list_of_elts)>0:
        out[0] = np.mean(np.asarray(list_of_elts))
        out[1] = min(list_of_elts)
        out[2] = max(list_of_elts)
    return out

def centroid(img,label):
    """returns x,y the position of the centroid of label in img"""
    x,y = np.where(img==label)
    return (np.mean(x),np.mean(y))

class Feature_Extractor(object):
    """Class meant to extract features from trajectories in a certain experiment"""
    def __init__(self,experiment):
        self.experiment = experiment
        self.trajectory = 0
    
    def set_trajectory(self,trajectory):
        self.trajectory = trajectory
        
    def open_arm(self,nr):
        """opens the arms corresponding to frame nr"""
        arms = m.open_frame(self.experiment.arm_path,nr+1)
        return arms
    def open_body(self,nr):
        """Opens the image containing the labeled bodies in frame nr"""
        body = m.open_frame(self.experiment.body_path,nr+1)
        return body

    def find_distance(self):
        """finds the distance of the tip (ie most distant point) of an arm to the
        cell body
        Returns :
            -distance_list: a list with size nr frames in the trajectory.
            Contains a list of arms distances in each frame
            -trajectories_container: a list containing the trajectories,
            each trajectory being here a tuple (frame number,arm size)
            -distance_dict_list: a list with size nr frames in the trajectory
            contains a list of dictionnaries associating arm label with length"""
        verification = False
        
        cells = self.trajectory.cells
        distance_list = []
        distance_dict_list = []
        label_list = []   #Contains the labels corresopnding to the same index in distance list
        for cell in cells:
            frame_number = cell.frame_number
            body = cell.body
            arms = cell.arms
            #Don't forget the -1 because we start indexing from 0
            frame_body = self.open_body(frame_number)-1
            frame_arm = self.open_arm(frame_number)-1
            body_mask = (frame_body==body).astype(np.uint8)
            xcb,ycb = centroid(frame_body,body)
            """kernel = np.ones((5,5),np.uint8)
            dilation = cv2.dilate(body_mask,kernel,iterations = 1)"""
            distance_arms=[]
            distance_dict = {}
            for arm in arms:
                arm_mask = frame_arm==arm
                """arm_root = dilation.copy()
                arm_root[~arm_mask]=0
                #(xc,yc are approximately the coordinates of the contact between arm and body)
                xc,yc = np.where(arm_root!=0)
                xc = np.mean(xc)
                yc = np.mean(yc)"""
                
                #Compute the distance between each pixel in arm and the root
                distance_x,distance_y = np.where(arm_mask)
                distances = np.sqrt((distance_x-xcb)**2+(distance_y-ycb)**2)
                distance = np.max(distances)
                
                #Just used fordebugging
                if cell.frame_number==5 and verification:
                    m.si(2*body_mask+arm_mask.astype(np.uint8))
                    xd = distance_x[distances==distance]
                    yd = distance_y[distances==distance]
                    plt.plot(yd,xd,"o")
                #End debugging
                label_list.append(arm)
                distance_arms.append(distance)
                distance_dict[arm] = distance
            distance_list.append(distance_arms)
            distance_dict_list.append(distance_dict)
        dist_dict_list = copy.deepcopy(distance_dict_list)
        #What we want out: a list of "trajectories", each of them being a list of:
        #-Frame numbers
        #-distance
        beginning = self.trajectory.beginning
        end = self.trajectory.end
        #Because badly coded from the beginning
        if end==240:
            end-=1
        trajectories_container = []
        for i,dict_list in enumerate(distance_dict_list):
            frame_nr = beginning+i
            to_pop_list = []
            for arm in dict_list:
                trajectories_list = []
                indices_arm,labels_arm = self.experiment.find_trajectory(frame_nr,arm)
                indices_arm = np.asarray(indices_arm)
                labels_arm = np.asarray(labels_arm)
                index_mask = np.logical_and(indices_arm>=beginning,indices_arm<=end)
                
                for j,lab in zip(indices_arm[index_mask],labels_arm[index_mask]):
                    dicto_list = distance_dict_list[j-beginning]
                    if lab in dicto_list:
                        trajectories_list.append((j,dicto_list[lab]))
                        
                        to_pop_list.append((j-beginning,lab))
                trajectories_container.append(trajectories_list)
            for k,l in to_pop_list:
                distance_dict_list[k].pop(l)
            to_pop_list = []
        return distance_list,trajectories_container,dist_dict_list

    def plot_distances(self,size_filter=0,new_fig=True):
        """Plots the distances profiles for all arms trajectories longer than
        size filter"""
        _,traj_list,_ = self.find_distance()
        print traj_list
        if new_fig:
            plt.figure()
            plt.title("Evolution of ramification length")
            plt.ylabel("length")
            plt.xlabel("frame nr")
            
        for lists in traj_list:
            x=[]
            y=[]
            if len(lists)>size_filter:   #show only longer, relevant trajectories
                for (u,v) in lists:
                    x.append(u)
                    y.append(v)
                    plt.plot(x,y)

                    
    def feature_vector(self,thickness_list):
        """extracts a n dimensional feature vector for each frame. 
        Parameters:
            -thickness_list: a list of list with the thickness of each
            arm in each frame
        Returns:
            -Arm length (mean/min/max)
            -Arm width (mean/min_length/max_length)
            -body radius
            -Nr arms"""
        n=3  #Number of parameters
        n_cells = len(self.trajectory.cells)
        feature_vector_total = np.zeros((n,n_cells))
        distance_list,_,distance_dict_list = self.find_distance()
        
        i=0 #Iteration counter
        for cell in self.trajectory.cells:
            feature_vector = np.zeros(n)
            distances = distance_list[i]
            distance_dict = distance_dict_list[i]
            thicknesses = thickness_list[cell.frame_number]
                
            feature_vector[0:3] = distribution_vector(distances)
           
            if len(distance_dict)>0:
                feature_vector[1] = max(thicknesses)
                
            else:
                feature_vector[1] = 0
            feature_vector_total[:,i]=feature_vector
            i+=1
        return feature_vector_total

def compute_thickness_list(path,nr):
    """Computes the thickness of each arm in a frame and returns them as
    a list
    Parameters:
        -path: path to the arms stack
        -nr: number of the frame of interst (starting from1)
    Returns:
        -thickness_list: a list containing in position i the thickness of
        the arm i"""
    arms = m.open_frame(path,nr)
    thick = cv2.distanceTransform((arms>0).astype(np.uint8),cv2.DIST_L2,3)
    thickness_list=[]
    for i in range(np.max(arms)):
        thickness = np.max(thick[arms==i+1])
        thickness_list.append(thickness)
    return thickness_list    
    
def get_all_thicknesses(experiment):
    all_thickness_list = []
    for i in range(1,242):
        print i
        all_thickness_list.append( compute_thickness_list(experiment.arm_path,i))
    return all_thickness_list
"""keep only 0,5,2"""
def extract_feature_vectors(experiment,simple_trajectories):
    """Extracts the n dimensional feature vector from predefined trajectories and returns them as 
    an unique array for clustering
    Parameters:
        experiment: instance of the class Experiment
        simple_trajectories: list of simple trajectories
    Returns:
        feature_vector: (n_features*n_cells) numpy array
        """
    feature_extractor = Feature_Extractor(experiment)
    
    #Compute first feature vector to initialize the array
    first_traj = simple_trajectories[0]
    first_traj = first_traj[1]
    feature_extractor.set_trajectory(first_traj)
    print "computing all thicknesses list"
    all_thickness_list = get_all_thicknesses(experiment)
    feature_vector=feature_extractor.feature_vector(all_thickness_list)
    for i in range(1,len(simple_trajectories)):
        traj = simple_trajectories[i][1]
        print "in simple trajectories loop"
        feature_extractor.set_trajectory(traj)
        new_vector = feature_extractor.feature_vector(all_thickness_list)
        feature_vector = np.concatenate((feature_vector,new_vector),axis=1)
    return feature_vector

def w_classification(path_list):
    """Wraps up the classification methods for the classified trajectories in
    path_list
    Parameters:
        path_list: list of strings (paths) to the different processed datasets.
    """
    #1/ Extract feature vectors from all dataset
    if len(path_list)==0:
        return
    fv=0
    for i,path in enumerate(path_list):
        trajectories = loadObject(os.path.join(path,"traj_selection.pkl"))
        experiment = Experiment(path)
        experiment.load()
        if i==0:
            fv = extract_feature_vectors(experiment,trajectories)
            fv=fv.transpose()
        else:
            fv2 = extract_feature_vectors(experiment,trajectories)
            fv2 = fv2.transpose()
            fv = np.concatenate((fv,fv2),axis=0)
    #2/Run Knn on this dataset.
    scaler = StandardScaler()
    fv = scaler.fit_transform(fv)
    kmeans = KMeans(n_clusters=3,n_init=400)
    
    predictions = kmeans.fit_predict(fv)
    return predictions,kmeans

def correspondance_vector(trajectories):
    """computes the correspondance vector as above"""
    vector_correspondance = [ zip( [i]*len(x[1].cells) ,range(len(x[1].cells))) for i,x in enumerate(trajectories)]
    correspondance=[]
    for lists in vector_correspondance:
        correspondance.extend(lists)
    return correspondance

def cell_bounding_box(experiment,cell,color='green'):
    """given a cell in an experiment, returns a picture centered on this cell
    overlaid with a certain color
    Parameters:
        experiment: instance of the class Experiment
        cell: instance of the class Cell, found in experiment
        color: string, specifies the color which needs to be overlaid.
    Returns:
        overlay: 3-D numpy array, image of the cell with a color mask
    """
    frame_number = cell.frame_number
    frame = m.open_frame(experiment.path,frame_number+1)
    body = m.open_frame(experiment.body_path,frame_number+1)
    arm = m.open_frame(experiment.arm_path,frame_number+1)
    
    rois = body==(cell.body+1)
    for elt in cell.arms:
        rois = np.logical_or(rois,arm==elt+1)
    im2,contours,hierarchy = cv2.findContours((rois).astype(np.uint8), 1, 2)
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
    sub_frame*=int(255/np.max(sub_frame))  #To have balanced histograms
    sub_rois = rois[y:y+h,x:x+w]
    sub_rois=sub_rois.astype(np.uint8)*255
    overlay = m.cv_overlay_mask2image(sub_rois,sub_frame,color)
    return overlay

def get_random_image(experiment,simple_trajs,correspondances,predictions,show=False):
    """Returns a random image centered on a cell in a list of trajectories
    Parameters: 
        experiment: instance of the class Experiment
        simple_trajs: list of trajectories
        correspondances: list of tuples. correspondances[i] is (traj_number, cell_number)
            corresponding to predictions[i]
        predictions: numpy array containing the predicted class of each cell
        show: bool, if True shows the image in a pyplot window
    Returns:
        image: numpy array, image centered on a cell with a color corresponding
            to its class
        label: int, specifies the class of the cell displayed
    """
    
    index = int(random.random()*len(correspondances))
    traj_index,cell_index = correspondances[index]
    cell = simple_trajs[traj_index][1].cells[cell_index]
    colors = ['green','red','blue','pink','yellow']
    label = predictions[index]
    image = cell_bounding_box(experiment,cell,colors[label%len(colors)])
    if show:
        plt.imshow(image,cmap='gray')
        plt.title(str(label))
    return image,label

def show_multiple_on_scale(experiment,simple_trajs,correspondances,predictions):
    """shows multiple images together. This method of display respects the scales of each 
    image.
    Paramters:
        experiemnt: instance of the Experiment class
        simple_trajs: list of simple trajectories
        correspondances: list of tuples (trajectory_index,cell_index)
        predictions: numpy array containing the predicted class of each cell
    Returns:
        out: composite image of classified cells"""
    n_images = 5
    im_list = []
    max_dim1=0
    max_dim2=0
    for i in range(n_images**2):
        im,lab = get_random_image(experiment,simple_trajs,correspondances,predictions)
        im_list.append(im)
        max_dim1 = max(im.shape[0],max_dim1)
        max_dim2 = max(im.shape[1],max_dim2)
    out = np.zeros((max_dim1*n_images,max_dim2*n_images,3),dtype=np.uint8)
    for i in range(n_images**2):
        k=i//n_images
        l=i%n_images
        out[k*max_dim1:k*max_dim1 + im_list[i].shape[0], l*max_dim2:l*max_dim2 + im_list[i].shape[1],:] = im_list[i]
    return out

def temporal_evolution(experiment,simple_trajectories,predictions,correspondances):
    """Monitors the temporal evolution in terms of number of cells per class
    """
    class_list=[]
    for i in range(experiment.n_frames-1):
        class_list.append([])
        
    for i,(index_traj,index_cell) in enumerate(correspondances):
        traj = simple_trajectories[index_traj][1]
        cell=traj.cells[index_cell]
        pred = predictions[i]
        class_list[cell.frame_number].append(pred)
    
    n_classes=np.max(predictions)+1
    fractions = np.zeros((experiment.n_frames,n_classes))
    for i,classes in enumerate(class_list):
        elements = np.asarray(classes)
        n_elts_in_frame = elements.size
        for j in range(n_classes):
            fractions[i,j] = float(np.count_nonzero(elements==j))/n_elts_in_frame
    return fractions


def plot_clf_vector(fv):
    plt.figure()
    for i in range(fv.shape[1]):
        plt.plot(fv[:,i])

def get_fractions(predictions):
    """Returns the fraction of each class in predictions"""
    n_classes=int(np.max(predictions))+1
    results = np.zeros(n_classes)
    nb_elements = predictions.size
    for i in range(n_classes):
        results[i] = float(np.count_nonzero(predictions==(i)))/nb_elements
    return results

def write_movie(experiment,name,simple_trajectories,predictions,correspondances):
    """Overlays all shape classifications to an entire movie, and writes it in 
    a new folder
    Parameters:
        experiment: instance of the class Experiment
        name: string, name of the folder where the new movie will be written
        simple_trajectories: list of Trajectory
        predictions: numpy array containing the predicted class of each cell
        correspondances: list of tuples. correspondances[i] is (traj_number, cell_number)
            corresponding to predictions[i]
    """
            
    path = os.path.join("..","data","microglia",name)
    colors = ['green','red','blue','pink','yellow']
    if not os.path.isdir(path):
        os.mkdir(path)
    #Separate each cell from each trajectory
    cell_list=[]
    for i in range(241):
        cell_list.append([])
        #Copy the frames in new directory. these frames will be modified by the loop
        frame = m.open_frame(experiment.path,i+1)
        cv2.imwrite(os.path.join(path,str(i+1)+".png"),frame)
        
    for i,(index_traj,index_cell) in enumerate(correspondances):
        traj = simple_trajectories[index_traj][1]
        cell=traj.cells[index_cell]
        pred = predictions[i]
        cell_list[cell.frame_number].append((cell,pred))
    n_pred = np.max(predictions)+1
    for frame_nr,cells in enumerate(cell_list):
        print "processing frame nr",frame_nr+1
        frame = m.open_frame(path,frame_nr+1)
        body = m.open_frame(experiment.body_path,frame_nr+1)
        arm = m.open_frame(experiment.arm_path,frame_nr+1)
        mask = np.zeros((frame.shape[0],frame.shape[1],n_pred),dtype=np.uint8)
        out = np.zeros((frame.shape[0],frame.shape[1],3),dtype=np.uint8)
        for cell,pred in cells:
            
            mask[:,:,pred] += (body==cell.body+1).astype(np.uint8)
            for elt in cell.arms:
                mask[:,:,pred]+=(arm==elt+1).astype(np.uint8)
        mask*=255
        for i in range(n_pred):
            out+= m.cv_overlay_mask2image(mask[:,:,i],frame,color=colors[i])/n_pred
        cv2.imwrite(os.path.join(path,str(frame_nr+1)+".png"),out)
        
        
class Cell_Classifier(object):
    """Class containing a reference to the dataset used for calssification,
    the k-means classifier"""
    def __init__(self,path_list,path = ".",name='cell-classifier'):
        self.path=path
        self.path_list = path_list
        self.classifier = None
        self.predictions = None
        self.name = name
        self.trajectories = []
        for path in self.path_list:
            tuple_traj = loadObject(os.path.join(path,"traj_selection.pkl"))
            self.trajectories.append( tuple_traj )
        
        
    def process(self):
        self.predictions, self.classifier = w_classification(self.path_list)
        
    def save(self):
        with open(os.path.join(self.path,self.name),'wb') as out:
            pickle.dump(self.__dict__,out)

    def load(self):
        print "loading trois petits points"
        with open(os.path.join(self.path,self.name),'rb') as dataPickle:
            self.__dict__ = pickle.load(dataPickle)
            
    def show_random_images(self):
        n_exp = int(random.random()*len(self.path_list))
        path_exp = self.path_list[n_exp]
        experiment = Experiment(path_exp)
        experiment.load()
        beg_index = 0
        for i in range(n_exp):
            beg_index+=sum([len(x[1].cells) for x in self.trajectories[i]])
        end_index = beg_index+sum([len(x[1].cells) for x in self.trajectories[n_exp]])
        predictions = self.predictions[beg_index:end_index]
        trajectories = self.trajectories[n_exp]
        correspondances = correspondance_vector(trajectories)
        out = show_multiple_on_scale(experiment,trajectories,correspondances,predictions)
        m.si(out)
        
    def plot_evolution(self):
        
        for i in range(len(self.path_list)):
            experiment = Experiment(self.path_list[i])
            experiment.load()
            beg_index = 0
            for j in range(i):
                beg_index+=sum([len(x[1].cells) for x in self.trajectories[j]])
            end_index = beg_index+sum([len(x[1].cells) for x in self.trajectories[i]])
            predictions = self.predictions[beg_index:end_index]
            trajectories = self.trajectories[i]
            correspondances = correspondance_vector(trajectories)
            vt = (experiment,trajectories,predictions,correspondances)
            plot_clf_vector(vt)
            plt.title(self.path_list[i])
            plt.show()
            frac = get_fractions(vt)
            plt.figure()
            plt.bar([0,1,2],frac)
            plt.title("histogram of trajectories in path:\n"+self.path_list[i])
            plt.xlabel('class')
            plt.ylabel('fractions')
            
    def write_entire_movie(self):
        for i in range(len(self.path_list)):
            experiment = Experiment(self.path_list[i])
            experiment.load()
            beg_index = 0
            for j in range(i):
                beg_index+=sum([len(x[1].cells) for x in self.trajectories[j]])
            end_index = beg_index+sum([len(x[1].cells) for x in self.trajectories[i]])
            predictions = self.predictions[beg_index:end_index]
            name=experiment.path+"_all_classified"
            trajectories = self.trajectories[i]
            correspondances = correspondance_vector(trajectories)
            write_movie(experiment,name,trajectories,predictions,correspondances)