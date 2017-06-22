#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 09:06:21 2017

@author: aurelien
"""

"""The idea behind this script is to extract a certain amount of features from
each frame from each trajectories and to cluster them together. Hopefully
this will give a consistent result"""
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

def resize_image(image,new_size=80):
    """
    Parameters:
        image: 2-D or 3-D numpy array
        new_size : int, gives the desired maximum dimension
    Returns:
        resized: numpy array"""
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
        im,lab = get_random_image(experiment1,simple_trajectories1,correspondance1,predictions)
        im_list.append(im)
        max_dim1 = max(im.shape[0],max_dim1)
        max_dim2 = max(im.shape[1],max_dim2)
    out = np.zeros((max_dim1*n_images,max_dim2*n_images,3),dtype=np.uint8)
    for i in range(n_images**2):
        k=i//n_images
        l=i%n_images
        out[k*max_dim1:k*max_dim1 + im_list[i].shape[0], l*max_dim2:l*max_dim2 + im_list[i].shape[1],:] = im_list[i]
    return out

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

total[:,1]=total[:,5]  #Not keep the min length and replace with max width
total = total[:,0:3]
"""
total2 = np.zeros((total.shape[0],4))
total2[:,0:3]=total[:,0:3]
total2[:,3]=total[:,3]
total=total2"""
scaler = StandardScaler()
total = scaler.fit_transform(total)
kmeans = KMeans(n_clusters=4,n_init=400)

predictions = kmeans.fit_predict(total)

#m.si(show_multiple(experiment1,simple_trajectories1,correspondance1,predictions),title='3-class classification of cell morphology')

cv2.imshow("Frames classified",show_multiple_on_scale(experiment1,simple_trajectories1,correspondance1,predictions))

"""Classification with just the first three elements of the feature vector, ie just the ones
relative to arms length seem to be the most pysically relevant"""

def separate_trajectories(predictions,correspondances):
    """Separate the trajectories between the ones staying in a constant class and
    the ones which change class.
    Parameters:
        predictions: an array containing the classification results
        correspondances: list of tuples. correspondances[i] is (traj_number, cell_number)
            corresponding to predictions[i]
    Returns:
        is_constant_class: array of bools, equal to True if all the cells 
            in the corresponding trajectory are classified as the same
        predictions_traj: list of arrays containing the prediction for each cell 
            of each trajectory
            """
    size = correspondances[-1][0]+1
    is_constant_class = np.zeros(size,dtype=bool) #Each element is 0 if not constant, True if constant
    predictions_traj = []
    for i in range(size):
        indices_traj = [j for j,(ind_traj,ind_cell) in enumerate(correspondances) if ind_traj==i]
        indices_traj = np.asarray(indices_traj)
        predictions_i = predictions[indices_traj]
        predictions_traj.append(predictions_i)
        print predictions_i
        values_pred = np.unique(predictions_i)
        if values_pred.size==1:
            is_constant_class[i]=True
    return is_constant_class,predictions_traj

d,preds=separate_trajectories(predictions,correspondance1)
print np.count_nonzero(d),d.size

def bounding_box(experiment,cell):
    """ Returns the position of the rectangular contour of a cell
    Parameters:
        experiment: instance of the class Experiment
        cell: instance of the class Cell whose bounding box we want to extract
    Return:
        (x,y,w,h): tuple, (x,y) and (x+w,y+h) defining two opposite corners
        of the bounding box of the cell
    """
    frame_number = cell.frame_number
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
    return x,y,w,h

def extract_trajectory_movie(experiment,name,trajectory,preds):
    colors = ['green','red','blue','pink','yellow']
    min_x = 5000
    min_y = 5000
    max_x = 0
    max_y = 0
    
    path = os.path.join("..","data","microglia",name)
    if not os.path.isdir(path):
        os.mkdir(path)
        
    for i,cell in enumerate(trajectory.cells):
        x,y,w,h = bounding_box(experiment,cell)
        min_x = min(min_x,x)
        min_y = min(min_y,y)
        
        max_x = max(max_x,x+w)
        max_y = max(max_y,y+h)
    for i,cell in enumerate(trajectory.cells):
        frame_number = cell.frame_number
        frame = m.open_frame(experiment.path,frame_number+1)
        body = m.open_frame(experiment.body_path,frame_number+1)
        arm = m.open_frame(experiment.arm_path,frame_number+1)

        frame = frame[min_y:max_y,min_x:max_x]
        body = body[min_y:max_y,min_x:max_x]
        arm = arm[min_y:max_y,min_x:max_x]
        mask = (body ==(cell.body+1)).astype(np.uint8)
        for elt in cell.arms:
            mask+=(arm==elt+1).astype(np.uint8)
        out = m.cv_overlay_mask2image(mask*255,frame,color=colors[preds[i]])
        cv2.imwrite(os.path.join(path,str(i)+".png"),out)

"""
for i in range(40):
    extract_trajectory_movie(experiment1,"cellShape"+str(10+i),simple_trajectories1[10+i][1],preds[10+i])
    
path_results = "/home/aurelien/Documents/rotation1/cell_shape/data/results/trajectories_classified/"
for i in range(40):
    num=i+10
    curr_path=os.path.join(path_results,"cellShape"+str(num),"curves")
    if not os.path.isdir(curr_path):
        os.mkdir(curr_path)
    fig=plt.figure()
    ce = Feature_Extractor(experiment1)
    ce.set_trajectory(simple_trajectories1[num][1])
    ce.plot_distances(new_fig=False)
    fig.savefig(os.path.join(curr_path,"distances.png"))
    plt.close(fig)
"""
#write_movie(experiment1,"movie_shapes",simple_trajectories1,predictions,correspondance1)

predictions1=predictions[0:len(correspondance1)]
predictions2=predictions[len(correspondance1):]

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

f2 = temporal_evolution(experiment2,simple_trajectories2,predictions2,correspondance2)
f1 = temporal_evolution(experiment1,simple_trajectories1,predictions1,correspondance1)

plt.figure()
for i in range(3):
    plt.plot(f1[:,i])

frac0_1 = float(np.count_nonzero(predictions1==0))/predictions1.size
frac0_2 = float(np.count_nonzero(predictions2==0))/predictions2.size

frac1_1 = float(np.count_nonzero(predictions1==1))/predictions1.size
frac1_2 = float(np.count_nonzero(predictions2==1))/predictions2.size

frac2_1 = float(np.count_nonzero(predictions1==2))/predictions1.size
frac2_2 = float(np.count_nonzero(predictions2==2))/predictions2.size

def get_arms_list_per_frame(experiment,simple_trajectories):
    """Extracts all arms from a set of manually selected trajectories and
    saves them in a folder
    """
    arms_list=[]
    for i in range(experiment.n_frames-1):
        arms_list.append([])
    
    if type(simple_trajectories[0])==tuple:
        simple_trajs = [y for x,y in simple_trajectories]
    else:
        simple_trajs = simple_trajectories
       
    for traj in simple_trajs:
        for cell in traj.cells:
            frame_number= cell.frame_number
            for arm in cell.arms:
                arms_list[frame_number].append(arm)
    return arms_list

def arm_bb(experiment,frame_number,arm_label):
    """ Returns an image of an arm only using its label and frame number
    """
    frame = m.open_frame(experiment.path,frame_number+1)
    arm = m.open_frame(experiment.arm_path,frame_number+1)
    
    roi = arm==arm_label
    im2,contours,hierarchy = cv2.findContours((roi).astype(np.uint8), 1, 2)
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
    sub_roi = roi[y:y+h,x:x+w]
    sub_frame[sub_roi==0]=0
    return sub_frame

out_arms=get_arms_list_per_frame(experiment1,simple_trajectories1)
m.si(arm_bb(experiment1,0,out_arms[0][0]))

def get_longer_arm(experiment,frame_number,arm_labels):
    """ Returns an image of an arm only using its label and frame number
    """
    frame = m.open_frame(experiment.path,frame_number+1)
    arm = m.open_frame(experiment.arm_path,frame_number+1)
    length_list = []
    for arm_label in arm_labels[frame_number]:
        roi = arm==arm_label
        im2,contours,hierarchy = cv2.findContours((roi).astype(np.uint8), 1, 2)
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
        length_list.append(max(w,h))
    max_size = max(length_list) 
    index_max_size = [i for i,z in enumerate(length_list) if z==max_size]
    for i in index_max_size:
        m.si(arm_bb(experiment,frame_number,arm_labels[frame_number][i]),title="size: "+str(max_size))
    return arm_bb(experiment,frame_number,arm_labels[frame_number][index_max_size[0]])

def make_square_image(image,new_size=80):
    """
    Parameters:
        image: 2-D or 3-D numpy array
        new_size : int, gives the desired maximum dimension
    Returns:
        resized: numpy array"""
    if image.shape[1]>image.shape[0]:
        r = float(new_size)/ image.shape[1]
        dim = (new_size,int(image.shape[0] * r))
    else:
        r = float(new_size)/ image.shape[0]
        dim = ( int(image.shape[1] * r),new_size)
        
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized


out_arm_big = get_longer_arm(experiment1,20,out_arms)
out_resized = resize_image(out_arm_big,new_size=64)
m.si2(out_arm_big,out_resized,"max size : "+str(max(out_arm_big.shape)),"resized")

out_rechanged = resize_image(out_resized,max(out_arm_big.shape))
m.si2(out_arm_big,out_rechanged,"original","reconstructed")