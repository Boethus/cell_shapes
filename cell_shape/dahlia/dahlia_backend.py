# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:25:34 2017

@author: univ4208
"""


import os
import sys
sys.path.append(os.path.join(".","..","segmentation"))
import methods as m
import numpy as np
import scipy.ndimage as ndi
import cv2
import matplotlib.pyplot as plt
import glob
from screeninfo import get_monitors
import cPickle as pickle
from sklearn.neighbors import KNeighborsClassifier
#import process_trajectories as pt


monitor = get_monitors()[0]
width_monitor = monitor.width
height_monitor = monitor.height

plt.close('all')


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

def process_mergings(mergings):
    path_clusters = os.path.join("..","data","microglia","1_centers_cluster")
    for i in range(241):
        print "process merging iteration ",i
        centers = m.open_frame(path_centers,i+1)
        indexes_merged = np.where(mergings[i,:]!=0)[0]
        values_merged = np.zeros(indexes_merged.shape)
    
        for k,elt in enumerate(indexes_merged):
            values_merged[k] = mergings[i,elt]
        labels_merged = indexes_merged+1
        out = np.zeros(centers.shape,dtype=np.int)
        out[centers>0] = 1
        for label,value in zip(labels_merged,values_merged):
            if value>0:
                out[centers==label] = value+1
        cv2.imwrite(os.path.join(path_clusters,str(i)+".png"), out )
        
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
    show_complex_trajectory(complex_trajectories[i],experiment,50)
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

def classify_trajectory(traj,experiment):
    """Displays traj and prompts the user about what to do"""
    show_trajectory(traj,experiment,50)
    possible_answers = ['r','w','t','m','q','e']
    inp = ''
    while(not inp in possible_answers):
        inp= raw_input("""how would you classify this image: ramified, withdrawal, transitional,
                      motile, error (r/w/t/m/e)? Press q to see the sequence again\n""")
    
    if inp=='q':
        classify_trajectory(traj,experiment)
    return inp

classifying = False
if classifying:
    classifications = []
    for i in range(84,len(complex_trajectories)):
        print i
        #cv2.destroyAllWindows()
        traj = complex_trajectories[i]
        inp= classify_complex_trajectory(traj,experiment3)
        if inp!='e' and inp!='q':    #otherwise it is an error
            classifications.append((inp,traj))

            
def replace_classification_results(classif_results,complex_traj):
    """Replaces the trajectories in classif_results by the ones in complex_traj"""
    for i,(label,traj) in enumerate(classif_results):
        new_traj = [x for x in complex_traj if x[0]==traj[0] ]
        if len(new_traj)!=1:
            print "eror shape"
        else:
            new_traj = new_traj[0]
            classif_results[i]=(label,new_traj)
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
#--------------------------Script-----------------------------------------------

def classify_trajectory2(traj,experiment):
    """Displays traj and prompts the user about what to do"""
    show_trajectory(traj,experiment,50)
    possible_answers = ['g','r','m','e','a']
    inp = ''
    while(not inp in possible_answers):
        inp= raw_input("""how would you classify this image: gaussian,relevant,maybe, error (g/r/m/e)? Press a to see the sequence again\n""")
    
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
            classifications_normal.append((inp,i))
    return classifications_normal

#if t: gaussian
# if 
"""
classifs=[]
for i in range(5):
    beg = int(float(len(traj_n_score)*i)/5)
    end=int(float(len(traj_n_score)*(i+1))/5)
    print "blaaaaaa",i
    if i==4:
        end = len(traj_n_score)-1
    clf1 = classify_simple_trajectories(traj_n_score[beg:end],experiment3)
    saveObject("rfp1_simple_trajs_part"+str(i),clf1)
    classifs.append(clf1)

#WTF: index 52
results="eeeeeeeeeeeeeeegegegeeemeemmereererremrreeeegeeemrreereremeemeeereeeeeeeeemeermrr"

total_classifs = []
for i in range(5):
    beg = int(float(len(traj_n_score)*i)/5)
    end=int(float(len(traj_n_score)*(i+1))/5)
    if i==4:
        end = len(traj_n_score)-1
    new_list = classifs[i]
    new_list = [(x,y+beg) for x,y in new_list]
    total_classifs.extend(new_list)
    
total_classifs_w_traj = [(x,traj_n_score[y]) for x,y in total_classifs]

indexes_to_change = []

for i,(x,elt) in enumerate(total_classifs_w_traj):
    if x=='a':
        new_val = classify_trajectory2(elt,experiment3)
        indexes_to_change.append((i,new_val))

corrected_classifs = total_classifs_w_traj[:]
for i,val in indexes_to_change:
    corrected_classifs[i] = (val,total_classifs_w_traj[i][1])
corrected_classifs = filter((lambda x:x[0]!="e"),corrected_classifs)
"""
"""
selected_trajs = loadObject("classification_normal_exp8")
trajecs_n_score = zip(simple_trajectories,gs_scores)
corresp_scores = []
for traj in selected_trajs:
    corresp_score = [score for x,score in trajecs_n_score if x==traj[1]]
    if len(corresp_score)!=1:
        print "error length"
    corresp_scores.append(corresp_score)"""
#Experiment 8:
#56+81 seen twice? same 63,71
#Analysis on this dataset: we find
#From Gaussian score>0.76, only gaussians if mini gaussian size 21
#if mini gaussian size =11 then 0.9 is the smallest score

#in simple trajectory : 74 out of 580 are medium or good, soit 12%

#Experiment RFP: 163 might be ok


"""Tous les ==a sont à refaire"""
