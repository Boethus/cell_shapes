#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:06:21 2017

@author: aurelien
"""

"""The idea behind this script is to extracta certainamount of features from
each frame from each trajectories and to cluster them together. Hopefully
this will givea consistent result"""
from find_arms import Experiment,loadObject
import os

path = os.path.join("..",'data','microglia','RFP1_denoised')
path_centers = os.path.join("..",'data','microglia','1_centers_improved') 
path_arms = os.path.join("..",'data','microglia','1_arms')  

experiment1 = Experiment(path,path_centers,path_arms)

simple_trajectories1 = loadObject("corrected_classifs_normal_exp1")
print simple_trajectories1
print len(simple_trajectories1)
    
path2 = os.path.join("..",'data','microglia','8_denoised')
path_centers2 = os.path.join("..",'data','microglia','8_centers_improved') 
path_arms2 = os.path.join("..",'data','microglia','8_arms')  
experiment2 = Experiment(path2,path_centers2,path_arms2)

simple_trajectories2 = loadObject("classification_normal_exp8")
simple_trajectories2 = filter(lambda x:x[0]!="g",simple_trajectories2)
def classify_trajectories():
    pass