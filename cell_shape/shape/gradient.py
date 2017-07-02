#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 20:22:05 2017

@author: aurelien
"""
import os
from find_arms import Experiment,loadObject
from process_trajectories import Feature_Extractor

path = os.path.join("..",'data','microglia','RFP1_denoised')
path_centers = os.path.join("..",'data','microglia','1_centers_improved') 
path_arms = os.path.join("..",'data','microglia','1_arms')  

experiment1 = Experiment(path,path_centers,path_arms)
experiment1.load()
#experiment1.track_arms_and_centers()
simple_trajectories1 = loadObject("corrected_normal_exp1")
simple_trajectories1 = filter(lambda x:x[0]!="g",simple_trajectories1)
feature_extractor = Feature_Extractor(experiment1)

def gradient_in_traj(feature_vector,correspondances,vector1,vector2):
    """Finds the gradients in trajcetory between vector1 and vetor2
    from a kmeans clustering"""
    max_index = correspondances[-1][0]
    out_vectors = []
    for i in range(max_index):
        out_vectors.append([])
    for i in range(len(correspondances)):
        traj,cell=correspondances[i]
        out_vectors[traj].append(feature_vector[i])
        