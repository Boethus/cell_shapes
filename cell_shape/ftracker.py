# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:55:30 2017

@author: univ4208
"""

import numpy as np
import cv2
import scipy.ndimage as ndi
import skimage
import hungarian
import skimage.feature
from sklearn.neighbors import KNeighborsClassifier
a=10

def filter_by_size(img_segm):
    """filters a segmented image by getting rid of the components with too few pixels"""
    
    numbers = np.zeros(np.max(img_segm-1))
    for i in range(1,np.max(img_segm)):
        numbers[i-1] = np.sum(img_segm==i)
        
    indexes = np.arange(1,np.max(img_segm))
    #indexes = indexes[numbers>np.mean(numbers)] #Deletes the 1-pixel elements
    indexes = indexes[numbers>1000] #Deletes the 1-pixel elements
    
    segm_filtered = np.zeros(img_segm.shape)
    j=1
    for i in (indexes):
        segm_filtered[img_segm==i] = j
        j+=1
    return segm_filtered

def canny_threshold(image):
    """Performs thresholding of an image using contour detection. This algorithm can be decomposed in two steps:
    1/Canny filter to get only edges in the image
    2/Gaussian blur
    3/Contour detection and filling"""
    size = 3
    
    #Step 1 
    canny = skimage.feature.canny(image)
    canny = canny/np.max(canny)*240     #!!Very important otherwise fails miserably. Flagadagada
    canny = canny.astype(np.uint8)
    #Step 2
    canny_blur = cv2.blur(canny,(size,size))
    canny_blur,contour,hierarchy = cv2.findContours(canny_blur,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    #Step 3
    i=0
    img = np.zeros(image.shape)
    for cnt in contour:
            cv2.drawContours(img,contour,i,255,-1)
            i+=1
    return img

def watershed_for_cells(gray,thresh):

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    opening = opening.astype(np.uint8)
    
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    markers, nLabels = ndi.label(sure_fg)

    #ret, markers = cv2.connectedComponents(sure_fg)   Not available with opencv 2.4
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==np.max(unknown)] = 0

    markers = markers.astype(np.int32)
    gray_converted = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cv2.watershed(gray_converted,markers)
    gray_converted[markers == -1] = [255,0,0]
    return gray_converted,markers



            
def centroids(img_segm):
    """Computes the positions of centroids in a segmented image.
    Returns the centroids numbers in order"""
    m = np.amax(img_segm)
    xs = np.zeros(m)
    ys = np.zeros(m)
    
    for i in range(0,m):
        pos_list = np.where(img_segm==i+1)
        xs[i] = np.mean(pos_list[0])
        ys[i] = np.mean(pos_list[1])
    return xs,ys

def centroids2(img_segm,liste):
    """Computes the positions of centroids in a segmented image.
    Returns the centroids numbers in order."""
    m = len(liste)
    xs = np.zeros(m)
    ys = np.zeros(m)
    j=0
    for elt in liste:
        pos_list = np.where(img_segm==elt)
        xs[j] = np.mean(pos_list[0])
        ys[j] = np.mean(pos_list[1])
        j+=1
    return xs,ys

def w_thresh(gray):
    """Wraps up the thresholding method"""
    thresh = canny_threshold(gray)
    kernel = np.ones((4,4), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    return thresh
            

"""niou methods"""
def updateLabels(correspondance_list,labels_list,i,image):
    """updates the list of labels based on the correspondance list
    image is the new image to update"""
    l_prev_index = len(labels_list[(i-1)%2])
    l_curr_index = len(labels_list[i%2])
    prev_index = labels_list[(i-1)%2]
    index_changes = []
    ref_image = np.copy(image)

    if l_prev_index==l_curr_index:   #Same number of cells
        
        for x,y in correspondance_list: 
            labels_list[i%2][y] = prev_index[x]
            image[ref_image==y+1] = prev_index[x]
            
    if l_curr_index > l_prev_index:  # Apparition
        new_index = max(prev_index)+2
        for x,y in correspondance_list: 
            if x<l_prev_index:
                labels_list[i%2][y] = prev_index[x]                
                image[ref_image==y+1] = prev_index[x]
                
            else:
                labels_list[i%2][y] = new_index
                image[ref_image==y+1] = new_index
                index_changes.append(new_index)
                new_index+=1
       
    if l_curr_index < l_prev_index:  # Disparition
        for x,y in correspondance_list: 
            if y<l_curr_index:
                labels_list[i%2][y] = prev_index[x]
                image[ref_image==y+1] = prev_index[x]
            else:
                index_changes.append(prev_index[x])
    return index_changes

def fillCostMatrix(xs0,ys0,xs1,ys1):
    """Computes a cost matrix for the different distances between centroids
    The rows represent the centroids in the previous frame
    The columns the centroids in the next frame
    Returns the cost matrix"""
    M = int ( max(len(xs0),len(xs1)) ) #Number of centroids.
    costMatrix = np.zeros((M,M))
    x_rows = np.zeros(M)
    x_rows[0:len(xs0)] = xs0
    y_rows = np.zeros(M)
    y_rows[0:len(xs0)] = ys0
    
    x_cols = np.zeros(M)
    x_cols[0:len(xs1)] = xs1
    y_cols = np.zeros(M)
    y_cols[0:len(xs1)] = ys1

    for i in range(len(xs0)):
        for j in range(len(xs1)):
            costMatrix[i,j]=(y_rows[i]-y_cols[j])**2
            costMatrix[i,j] += (x_rows[i]-x_cols[j])**2
    return costMatrix


def w_hungarian(centroids_buffer,i,index_disappeared,images_buffer,labels_list):
    #Hungarian algorithm
    xs0,ys0 = centroids_buffer[(i-1)%2]
    xs1,ys1 = centroids_buffer[i%2]
    size_diff = len(xs1)-len(xs0)  #Is positive if a cell appears and negative if one disappears.

    cost_matrix = fillCostMatrix(xs0,ys0,xs1,ys1)
    hungry_test = hungarian.Hungarian(input_matrix=cost_matrix)
    hungry_test.calculate()
    correspondance_list = hungry_test.get_results()
    image_to_update  = images_buffer[:,:,i%2]   #This labeled image will be re labeled by updateLabels
    disappeared = updateLabels(correspondance_list,labels_list,i,image_to_update)
    if size_diff!=0:
        index_disappeared.append((i,size_diff,disappeared))
    return image_to_update               
                



def filter_out_flickers(total_buffer,index_disappeared):
    """Removes flickering artefacts considering that a signal flickering in time is an artefact.
    Can only handle one flickering thing at a time for now."""
    
    wait_for_disparition = False
    candidate_for_disparition = -1
    to_destroy = []   #List of 3D tuples (value,first_index,last_index) of segmented elements to remove from image
    beginning_index = -1
    premier_i =-1
    list_of_is =[]
    
    previous_index, osef2, osef3 = index_disappeared[0]  #Get the index for the first event

    for i in range(0,len(index_disappeared)):
        index,diff,list_index = index_disappeared[i]
        #Remove an appearing and disappearing object from the experiment only if it
        #disappears in the next 5 (arbitrary) frames. If longer, conseder that something relevant
        #happened.
        
        if wait_for_disparition:
            #If sth appeared, destroy it if:
            #-It is the same object that disappears
            #-If the event is a disparition
            #-If it disappears in a time<time_thr
            size = np.count_nonzero(total_buffer[:,:,index-1]==list_index[0])
            if list_index[0]==candidate_for_disparition and diff<0 and size<500:
                to_destroy.append((list_index[0],beginning_index,index))
                list_of_is.append(premier_i)
                list_of_is.append(i)
            wait_for_disparition=False
            
        if diff>0:   #Creation, do wait for disparition
            candidate_for_disparition = list_index[0]
            beginning_index = index
            wait_for_disparition =True
            premier_i = i
            
    return to_destroy,list_of_is

def remove_to_destroy(total_buffer,to_destroy):
    """Removes the segmented elements of low interest"""
    totbuf=np.copy(total_buffer)
    for val,begInd,endInd in to_destroy:
        for j in range(endInd-begInd):
            index_beg = begInd+j
            totbuf[ total_buffer[:,:,index_beg]==val,index_beg]=0
    return totbuf
            
def get_remaining_events(index_disappeared,to_destroy):
    """From the list of events and the list of elements destroyed, returns
    the list of events remaining to be processed."""
    index_cp = index_disappeared[:]
    for i,deb,fin in to_destroy:
        index_cp = [(x,y,z) for x,y,z in index_cp if (x!=deb and x!=fin)]
    return index_cp

def isFusion(event,buff):
    """Tests is the triplet event (index,difference,label) is a fusion event"""
    index,diff,label = event
    label = label[0]
    if diff>0:
        return False,[]
    img_before = np.copy(buff[:,:,index-1])
    img_after = np.copy(buff[:,:,index])
    mask_before = (img_before==label).astype(np.uint8)
    nb_elts_before = np.amax(img_before)
    kernel = np.ones((7,7),np.uint8)
    neighbouring_mask = cv2.dilate(mask_before,kernel,iterations=8)

    new_map = np.multiply(img_before,neighbouring_mask.astype(np.uint8))
    
    #Removing the element we are currently looking at
    new_map[img_before==label]=0
    possible_candidates = []
    for i in range(nb_elts_before):
        if np.any(new_map==i+1):
            possible_candidates.append(i+1)
    #Computes the area of the cells and compares them
    size_cell_disappearing = np.count_nonzero(img_before==label)
    match = []   #lists the ratios sizeAfter/sizeBefore for possible matches
    
    for vals in possible_candidates:
        size_other_cell = np.count_nonzero(img_before==vals)
        size_before = size_other_cell+size_cell_disappearing
        size_after = np.count_nonzero(img_after==vals)
        ratio = float(size_after)/float(size_before)
        if ratio>0.8 and ratio<1.2:
            match.append((vals,abs(1-ratio)))
    if len(match)==0:
        return False,[]
    if len(match)>1:
        #Several matches, so pick the best
        values = [y for x,y in match]
        result_label,osef = match[np.argmin(values)]
    else:
        result_label, osef = match[0]
    return True,result_label

def isDivision(event,buff):
    """Tests is the triplet event (index,difference,label) is a Division event"""
    index,diff,label = event
    label = label[0]
    if diff<0:
        return False,[]
    img_before = np.copy(buff[:,:,index-1])
    img_after = np.copy(buff[:,:,index])
    mask_after = (img_after==label).astype(np.uint8)
    nb_elts_after = np.amax(img_after)
    kernel = np.ones((7,7),np.uint8)
    neighbouring_mask = cv2.dilate(mask_after,kernel,iterations=8)

    
    new_map = np.multiply(img_after,neighbouring_mask.astype(np.uint8))    
    #Removing the element we are currently looking at
    new_map[img_after==label]=0
    possible_candidates = []
    for i in range(nb_elts_after):
        if np.any(new_map==i+1):
            possible_candidates.append(i+1)
    #Computes the area of the cells and compares them
    size_cell_after = np.count_nonzero(img_after==label)
    match = []   #lists the ratios sizeAfter/sizeBefore for possible matches
    for vals in possible_candidates:
        size_before = np.count_nonzero(img_before==vals)
        size_other_cell = np.count_nonzero(img_after==vals)
        size_after = size_cell_after + size_other_cell
        ratio = float(size_after)/float(size_before)
        if ratio>0.8 and ratio<1.2:
            match.append((vals,abs(1-ratio)))
    if len(match)==0:
        return False,[]
    if len(match)>1:
        #Several matches, so pick the best
        values = [y for x,y in match]
        result_label,osef = match[np.argmin(values)]
    else:
        result_label, osef = match[0]
    return True,result_label


def splitCell(buff,index,ref_label,new_label):
    """Splits a cell into two"""
    cell_before = np.copy(buff[:,:,index-1])
    cell_after = np.copy(buff[:,:,index])
    
    mask_after = cell_after ==ref_label
    
    cell_before[np.logical_not(mask_after)] = 0
    
    mask_ref_label = cell_before ==ref_label
    mask_new_label = cell_before==new_label
  
    after_sure_ref = np.logical_and(mask_ref_label,mask_after)
    after_sure_new = np.logical_and(mask_new_label,mask_after)
    after_unsure = np.logical_and(mask_after,np.logical_not(np.logical_or(after_sure_ref,after_sure_new) ) )

    xref,yref = np.where(after_sure_ref)
    ref_pts = np.concatenate((xref.reshape(-1,1),yref.reshape(-1,1)),axis=1)
    xnew,ynew = np.where(after_sure_new)
    new_pts = np.concatenate((xnew.reshape(-1,1),ynew.reshape(-1,1)),axis=1)
    
    labels_ref = np.ones(xref.shape[0])
    labels_new = np.zeros(xnew.shape[0])
    labels = np.concatenate((labels_ref,labels_new),axis=0)
    labels.reshape(-1,1)
    X= np.concatenate((ref_pts,new_pts),axis = 0)
    
    xu,yu = np.where(after_unsure)
    u_pts = np.concatenate((xu.reshape(-1,1),yu.reshape(-1,1)),axis=1)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X, labels)
    pred = neigh.predict(u_pts)
    for i in range(pred.shape[0]):
        #if pred is 1 goes to ref if 0 goes to new
        if pred[i]==1:
            after_sure_ref[u_pts[i,0],u_pts[i,1]]=True
        else:
            after_sure_new[u_pts[i,0],u_pts[i,1]]=True
    #Assigning the new values to the thing:
    buff[after_sure_ref,index] = ref_label
    buff[after_sure_new,index] = new_label
    
def bindEvents(fusionEvents,divisionEvents, buff):
    """Relates a fusion to a division event"""
    #1/Finding correspondances
    fusion_indices = []
    fusion_labels = []
    fusion_labels_2 = []  # In label 2 says with which cell the disappearded one has
    for events,label in fusionEvents:
        index,osef,labels = events
        fusion_indices.append(index)
        fusion_labels.append(labels[0])
        fusion_labels_2.append(label)
        
    division_indices = []
    division_labels = []
    division_labels_2 = []  # Tells in which cell it is created
    for events,label in divisionEvents:
        index,osef,labels = events
        division_indices.append(index)
        division_labels.append(labels[0])
        division_labels_2.append(label)
    
    associated_division_list = []
    associated_indexes = []
    for i in fusion_indices:
        ind = next((x for x in division_indices if x>i),-1)
        if ind>0:
            associated_division_list.append((i,ind))
            corr_ind_fusion = fusion_indices.index(i)
            corr_ind_division = division_indices.index(ind)
            associated_indexes.append((corr_ind_fusion,corr_ind_division))

            
    #2/removing corresponding elements
    for j in range(len(associated_division_list)):
        index_fus, index_div = associated_indexes[j]
        if division_labels_2[index_div]==fusion_labels_2[index_fus]:
            #If they are not equal, means that the process of division/fusion 
            #has not happened on the same blob and hence is not relevant
            big_label = division_labels_2[index_div]
            small_label = fusion_labels[index_fus]
            new_label = division_labels[index_div] #Replace after division this label by small label
            first_index = fusion_indices[index_fus]
            second_index = division_indices[index_div]
            
            for k in range(second_index-first_index):
                splitCell(buff,first_index+k,big_label,small_label)
                
            #Attribution of the new created cells to each one of the previous cells:
            #For this, we take the closest centroid
            #centroid of the big label
            last_image = buff[:,:,second_index]
            xs,ys = centroids2(last_image,[big_label,new_label])
            xs0,ys0 = centroids2(buff[:,:,second_index-1],[big_label,small_label])
            dist_regular = (xs0[0]-xs[0])**2 + (ys0[0]-ys[0])**2 + (xs0[1]-xs[1])**2 + (ys0[1]-ys[1])**2
            dist_inverted = (xs0[0]-xs[1])**2 + (ys0[0]-ys[1])**2 + (xs0[1]-xs[0])**2 + (ys0[1]-ys[0])**2
            
            if dist_regular>dist_inverted:
                print "ca marche pas gael euh quoi?"
                tmp_stack = buff[:,:,second_index:]
                tmp_stack[buff[:,:,second_index:]==big_label]=small_label
                tmp_stack[buff[:,:,second_index:]==new_label]=big_label
                buff[:,:,second_index:] = tmp_stack
                division_labels = [x if (x!=new_label and x!=big_label) else big_label if x==new_label else small_label for x in division_labels]
                fusion_labels = [x if x!=new_label and x!=big_label else big_label if x==new_label else small_label for x in fusion_labels]
                division_labels_2= [x if x!=new_label and x!=big_label else big_label if x==new_label else small_label for x in division_labels_2]
                fusion_labels_2= [x if x!=new_label and x!=big_label else big_label if x==new_label else small_label for x in fusion_labels_2]
            else:
                print "ca marche bien gael"
                """Reassigning new labels"""
                tmp_stack = buff[:,:,second_index:]
                tmp_stack[tmp_stack==new_label] = small_label
                buff[:,:,second_index:] = tmp_stack
                division_labels = [x if x!=new_label else small_label for x in division_labels]
                fusion_labels = [x if x!=new_label else small_label for x in fusion_labels]
                division_labels_2 = [x if x!=new_label else small_label for x in division_labels_2]
                fusion_labels_2 = [x if x!=new_label else small_label for x in fusion_labels_2]
     