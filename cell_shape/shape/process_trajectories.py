#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:28:04 2017

@author: aurelien
"""
import cv2
import numpy as np
import methods as m
import matplotlib.pyplot as plt
import copy

plt.close('all')
#-------------Helper functions------------------------------------
def thickness_list(path,nr):
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

def centroid(img,label):
    """returns x,y the position of the centroid of label in img"""
    x,y = np.where(img==label)
    return (np.mean(x),np.mean(y))

def distribution_vector(list_of_elts):
    """returns a 3*1 vector containing [mean,min,max] of list"""
    out = np.zeros(3)
    if len(list_of_elts)>0:
        out[0] = np.mean(np.asarray(list_of_elts))
        out[1] = min(list_of_elts)
        out[2] = max(list_of_elts)
    return out
#------------------Feature extractors--------------------------------------
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
        feature_vector_total = np.zeros((n_cells,n))
        distance_list,_,distance_dict_list = self.find_distance()
        
        i=0 #Iteration counter
        for cell in self.trajectory.cells:
            feature_vector = np.zeros(n)
            distances = distance_list[i]
            thicknesses = thickness_list[cell.frame_number]
            feature_vector[0:3] = distribution_vector(distances)
            feature_vector[1] = 0
            thicknesses_cell=[]
            for arm_label in cell.arms:
                thicknesses_cell.append(thicknesses[arm_label])
            if len(thicknesses_cell)>0:
                feature_vector[1] = max(thicknesses_cell)
            feature_vector_total[i,:]=feature_vector
            i+=1
        return feature_vector_total
        
class Complex_Feature_Extractor(Feature_Extractor):
    """Feature extractor for a complex trajectory"""
    def __init__(self,*args):
        Feature_Extractor.__init__(self,*args)
        
    def find_distance(self):
        all_traj=[]
        extractor = Feature_Extractor(self.experiment)
        for elt in self.trajectory:
            if type(elt)!=tuple:
                extractor.set_trajectory(elt)
                _,traj_list = extractor.find_distance()
                all_traj.extend(traj_list)
        return all_traj
    def plot_distances(self,size_filter=0,new_fig=True):
        """Plots the distances profiles for all arms trajectories longer than
        size filter"""
        traj_list = self.find_distance()
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
                    
    def speed(self):
        extractor = Feature_Extractor(self.experiment)
        total_speeds=[]
        counter=0
        for i,elt in enumerate(self.trajectory):
            if type(elt)!=tuple:
                extractor.set_trajectory(elt)
                speeds = extractor.speed()
                total_speeds.extend(speeds)
            else:
                if type(self.trajectory[i-1])==tuple:
                    beginning = self.trajectory[i-2].end
                    if i==len(self.trajectory):  #Means there is no trajectory afterwards
                        end=240
                    else:
                        end = self.trajectory[i+1].beginning
                    next_label = self.trajectory[i-1][2]
                    first_body = self.open_body(beginning)
                    prev_position_x,prev_position_y = centroid(first_body,1+self.trajectory[i-2].cells[-1].body)
                   
                    for frame in range(beginning+1,end):
                        centroids_list  = self.experiment.arm_tracker.info_list[frame].centroids
                        xc = centroids_list[0][next_label]
                        yc = centroids_list[1][next_label]
                        speed = np.sqrt( (xc-prev_position_x)**2 + (yc-prev_position_y)**2  )
                        counter+=1
                        total_speeds.append(speed)
                        next_label = self.experiment.arm_tracker.next_cell(frame,next_label)
                        if next_label==-1:
                            print "End of arm trajectory"
                            break
                        prev_position_x = xc
                        prev_position_y = yc
                    if i!=len(self.trajectory):
                        last_body = self.trajectory[i+1].cells[0].body
                        centroids_list = self.experiment.body_tracker.info_list[end].centroids
                        last_x = centroids_list[0][last_body]
                        last_y = centroids_list[1][last_body]
                        speed = np.sqrt( (last_x-xc)**2 + (last_y-yc)**2 )
                        counter+=1
                        total_speeds.append(speed)
        return total_speeds
        
#test : 
"""
extractor = Feature_Extractor(experiment2)
extractor.set_trajectory(classification_results[3][1][0])
dist_list,traj_list= extractor.find_distance()"""
def plot_multiple_caracs(experiment, classification_results,x,y,carac="distance"):
    ce = Complex_Feature_Extractor(experiment)
    
    plt.figure()
    if carac=="distance":
        plt.suptitle("Evolution fo distance arms-body with time")
    else:
        plt.suptitle("Speed profiles")
    for i in range(x*y):
        plt.subplot(x,y,i+1)
        ce.set_trajectory(classification_results[i][1])
        if carac=="distance":
            ce.plot_distances(new_fig=False)
        else:
            plt.plot(ce.speed())
        #plt.title(str(i)+"Class: "+classification_results[i][0])


def plot_caracs(experiment, classification_results,x,y,carac="distance"):
    ce = Feature_Extractor(experiment)
    
    plt.figure()
    if carac=="distance":
        plt.suptitle("Evolution fo distance arms-body with time")
    else:
        plt.suptitle("Speed profiles")
    for i in range(x*y):
        plt.subplot(x,y,i+1)
        ce.set_trajectory(classification_results[i][1])
        if carac=="distance":
            ce.plot_distances(new_fig=False)
        else:
            plt.plot(ce.speed())            
"""
ce = Feature_Extractor(experiment1)
ce.set_trajectory(simple_trajectories1[15][1])
ce.plot_distances()"""