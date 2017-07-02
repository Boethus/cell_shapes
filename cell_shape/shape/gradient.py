#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 20:22:05 2017

@author: aurelien
"""
import os
from find_arms import Experiment,loadObject
from process_trajectories import Feature_Extractor
import numpy as np

def feature_vector_per_traj(feature_vector,correspondances):
    """Finds the gradients in trajcetory between vector1 and vetor2
    from a kmeans clustering"""
    max_index = correspondances[-1][0]
    print max_index
    out_vectors = []
    for i in range(max_index):
        out_vectors.append([])
    for i in range(len(correspondances)):
        traj,cell=correspondances[i]
        out_vectors[traj].append(feature_vector[i,:])
    return out_vectors

def gradient_in_traj(feature_vectors,vector1,vector2):
    """A trajectory represented by a set of feature vectors
    in a list of numpy arrays"""
    gradients = []
    direction = vector2-vector1
    for i in range(1,len(feature_vectors)):
        grad = feature_vectors[i]-feature_vector[i-1]
        gradients.append(np.dot(grad,direction))
    return gradients
    
vectors_kmeans = kmeans.cluster_centers_
vec1 = vectors_kmeans[0,:]
vec2 = vectors_kmeans[1,:]
vec3 = vector_kmeans[2,:]   
out = feature_vector_per_traj(fv,correspondance1)
