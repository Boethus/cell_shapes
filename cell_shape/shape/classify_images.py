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
from find_arms import *
from process_trajectories import Feature_Extractor,thickness_list
import os
import numpy as np
import methods as m
import cv2
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import random
    
def get_all_thicknesses(experiment):
    all_thickness_list = []
    for i in range(1,242):
        print i
        all_thickness_list.append( thickness_list(experiment.arm_path,i))
    return all_thickness_list

def extract_feature_vectors(experiment,simple_trajectories):
    """Extracts the n dimensional feature vector from predefined trajectories and returns them as 
    an unique array for clustering"""
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
 
    


def cell_bounding_box(experiment,cell,color='green'):
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

def resize_image(image,new_size=80):
    if image.shape[1]>image.shape[0]:
        r = float(new_size)/ image.shape[1]
        dim = (new_size,int(image.shape[0] * r))
    else:
        r = float(new_size)/ image.shape[0]
        dim = ( int(image.shape[1] * r),new_size)
        
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def show_multiple(experiment,simple_trajs,correspondances,predictions):
    """shows multiple images together"""
    size=80
    n_images = 5
    out = np.zeros((n_images*size,n_images*size,3),dtype=np.uint8)
    for i in range(n_images):
        for j in range(n_images):
            im,lab = get_random_image(experiment1,simple_trajectories1,correspondance1,predictions)
            while im.size<100:
                im,lab = get_random_image(experiment1,simple_trajectories1,correspondance1,predictions)
            im = resize_image(im,size)
            out[i*size:i*size+im.shape[0],j*size:j*size+im.shape[1],:]=im
    return out

def show_multiple_on_scale(experiment,simple_trajs,correspondances,predictions):
    """shows multiple images together. This method of display respects the scales of each 
    image."""
    n_images = 5
    im_list = []
    max_dim1=0
    max_dim2=0
    dims1=[]
    dims2=[]
    for i in range(n_images**2):
        im,lab = get_random_image(experiment1,simple_trajectories1,correspondance1,predictions)
        im_list.append(im)
        #dims1.append()
        max_dim1 = max(im.shape[0],max_dim1)
        max_dim2 = max(im.shape[1],max_dim2)
    out = np.zeros((max_dim1*n_images,max_dim2*n_images,3),dtype=np.uint8)
    for i in range(n_images**2):
        k=i//n_images
        l=i%n_images
        out[k*max_dim1:k*max_dim1 + im_list[i].shape[0], l*max_dim2:l*max_dim2 + im_list[i].shape[1],:] = im_list[i]
    return out


#all_thickness_list = get_all_thicknesses(experiment1)

def test_correspondance(experiment,predictions,kmeans,all_thickness_list,correspondances):
    """Check that the predictions correspond to te actual cell"""
    index=int(random.random()*len(correspondances))
    index_traj,index_cell = correspondances[index]
    
    indices_traj = [i for i,(ind_traj,ind_cell) in enumerate(correspondances) if ind_traj==index_traj]


    traj = simple_trajectories1[index_traj][1]

    fe = Feature_Extractor(experiment)
    fe.set_trajectory(traj)
    vector = fe.feature_vector(all_thickness_list)
    vector = vector.transpose()
    vector = vector[:,0:7]
    vector=scaler.transform(vector)
    new_predictions = kmeans.predict(vector)
    cell_prediction = new_predictions[index_cell]
    
    
    test= cell_prediction==predictions[index]
    test2 = predictions[np.asarray(indices_traj)]==new_predictions
    if test:
        print "test passed"
    else:
        print "test failed, index",index
    if False in test2:
        print "test2 failed"
        print new_predictions
        print predictions[np.asarray(indices_traj)]
    else:
        print "test2 passed"
path = os.path.join("..",'data','microglia','RFP1_denoised')
path_centers = os.path.join("..",'data','microglia','1_centers_improved') 
path_arms = os.path.join("..",'data','microglia','1_arms')  

experiment1 = Experiment(path,path_centers,path_arms)
experiment1.load()
#experiment1.track_arms_and_centers()
simple_trajectories1 = loadObject("corrected_normal_exp1")
simple_trajectories1 = filter(lambda x:x[0]!="g",simple_trajectories1)

path2 = os.path.join("..",'data','microglia','8_denoised')
path_centers2 = os.path.join("..",'data','microglia','8_centers') 
path_arms2 = os.path.join("..",'data','microglia','8_arms')  
experiment2 = Experiment(path2,path_centers2,path_arms2)
experiment2.load()
simple_trajectories2 = loadObject("corrected_normal_exp8")
simple_trajectories2 = filter(lambda x:x[0]!="g",simple_trajectories2)

#Vector correspondance returns for each index of frame processed, a tuple
#(position in traj list,position of frame in traj)
vector_correspondance_1 = [ zip( [i]*len(x[1].cells) ,range(len(x[1].cells))) for i,x in enumerate(simple_trajectories1)]
correspondance1=[]
for lists in vector_correspondance_1:
    correspondance1.extend(lists)

vector_correspondance_2 = [ zip( [i]*len(x[1].cells) ,range(len(x[1].cells))) for i,x in enumerate(simple_trajectories2)]
correspondance2=[]
for lists in vector_correspondance_2:
    correspondance2.extend(lists)
    
fv = extract_feature_vectors(experiment1,simple_trajectories1)
fv2 = extract_feature_vectors(experiment2,simple_trajectories2)

fv=fv.transpose()
fv2=fv2.transpose()

total = np.concatenate((fv,fv2),axis=0)

#Clustering:

#total = total.reshape((total.shape[1]),total.shape[0])
#total = total[:,0:7]
scaler = StandardScaler()
total = scaler.fit_transform(total)
kmeans = KMeans(n_clusters=4)

predictions = kmeans.fit_predict(total)

cv2.imshow("Frames classified",show_multiple(experiment1,simple_trajectories1,correspondance1,predictions))
m.si(show_multiple(experiment1,simple_trajectories1,correspondance1,predictions),title='3-class classification of cell morphology')

cv2.imshow("Frames classified",show_multiple_on_scale(experiment1,simple_trajectories1,correspondance1,predictions))
