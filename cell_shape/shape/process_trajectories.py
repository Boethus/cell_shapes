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

plt.close('all')

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
        cell body"""
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
            kernel = np.ones((5,5),np.uint8)
            dilation = cv2.dilate(body_mask,kernel,iterations = 1)
            distance_arms=[]
            distance_dict = {}
            for arm in arms:
                arm_mask = frame_arm==arm
                arm_root = dilation.copy()
                arm_root[~arm_mask]=0
                #(xc,yc are approximately the coordinates of the contact between arm and body)
                xc,yc = np.where(arm_root!=0)
                xc = np.mean(xc)
                yc = np.mean(yc)
                
                #Compute the distance between each pixel in arm and the root
                distance_x,distance_y = np.where(arm_mask)
                distances = np.sqrt((distance_x-xcb)**2+(distance_y-ycb)**2)
                distance = np.max(distances)
                
                #Just used of debugging
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
                        #distance_dict_list[j-beginning].pop(lab)
                        to_pop_list.append((j-beginning,lab))
                trajectories_container.append(trajectories_list)
            for k,l in to_pop_list:
                distance_dict_list[k].pop(l)
            to_pop_list = []
        return distance_list,trajectories_container

    def speed(self):
        """extracts the speed profile of a trajectory. """
        cells = self.trajectory.cells
        length = len(cells)
        first_body = self.open_body(cells[0].frame_number)
        prev_position_x,prev_position_y = centroid(first_body,cells[0].body+1)
        speeds = []
        for i in range (1,length):
            body_img = self.open_body(cells[i].frame_number)
            new_position_x,new_position_y = centroid(body_img,cells[i].body+1)
            speed = np.sqrt((new_position_x-prev_position_x)**2 + (new_position_y-prev_position_y)**2)
            prev_position_x = new_position_x
            prev_position_y = new_position_y
            speeds.append(speed)
        return speeds
    
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

extractor = Feature_Extractor(experiment2)
extractor.set_trajectory(classification_results[3][1][0])
dist_list,traj_list= extractor.find_distance()

ce = Complex_Feature_Extractor(experiment2)

plt.figure()
plt.suptitle("Evolution fo distance arms-body with time")
for i in range(len(classification_results)):
    plt.subplot(5,3,i+1)
    ce.set_trajectory(classification_results[i][1])
    #ce.plot_distances(new_fig=False)
    plt.plot(ce.speed())
    plt.title(str(i)+"Class: "+classification_results[i][0])

ce.set_trajectory(classification_results[3][1])
speed = ce.speed()
plt.figure()
plt.plot(speed)
plt.title("complex speed")
x = []
y = []
for i,elt in enumerate(dist_list):
    for j in elt:
        x.append(i)
        y.append(j)
plt.figure()
plt.scatter(x,y)
plt.title("evolution of distances with time")

#Comparison with the trajectories extracted:
plt.figure()
for lists in traj_list:
    x=[]
    y=[]
    if len(lists)>10:   #show only longer, relevant trajectories
        for (u,v) in lists:
            x.append(u)
            y.append(v)
            plt.plot(x,y)
plt.title("Distances of different arms to the center")
speeds = extractor.speed()
plt.figure()
plt.plot(speeds)
plt.title("speed")
            