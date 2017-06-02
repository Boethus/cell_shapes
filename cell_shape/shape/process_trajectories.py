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
        for cell in cells:
            frame_number = cell.frame_number
            body = cell.body
            arms = cell.arms
            #Don't forget the -1 because we start indexing from 0
            frame_body = self.open_body(frame_number)-1
            frame_arm = self.open_arm(frame_number)-1
            body_mask = (frame_body==body).astype(np.uint8)

            kernel = np.ones((5,5),np.uint8)
            dilation = cv2.dilate(body_mask,kernel,iterations = 1)
            distance_arms=[]
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
                distances = np.sqrt((distance_x-xc)**2+(distance_y-yc)**2)
                distance = np.max(distances)
                
                if cell.frame_number==5 and verification:
                    m.si(2*body_mask+arm_mask.astype(np.uint8))
                    xd = distance_x[distances==distance]
                    yd = distance_y[distances==distance]
                    plt.plot(yd,xd,"o")
                    
                distance_arms.append(distance)
            distance_list.append(distance_arms)
        return distance_list

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
#test : 

extractor = Feature_Extractor(experiment2)
extractor.set_trajectory(trajj)
dist_list= extractor.find_distance()
x = []
y = []
for i,elt in enumerate(dist_list):
    for j in elt:
        x.append(i)
        y.append(j)
plt.figure()
plt.scatter(x,y)
plt.title("evolution of distances with time")
speeds = extractor.speed()
plt.figure()
plt.plot(speeds)
plt.title("speed")
            