# -*- coding: utf-8 -*-
"""
Created on Wed May 03 11:20:33 2017

@author: univ4208
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.feature
import skimage.filters
import skimage.morphology
import cv2
import scipy.ndimage as ndi
import pywt
import glob
import platform
from scipy.optimize import linear_sum_assignment
from skimage import img_as_ubyte
#plt.close("all")

def open_frame(path,number):
    """Open frame with number located at path"""
    num=str(number).zfill(3)   #Zero filling
    name = glob.glob(path+"/*"+num+"*")
    if len(name)==0:
        name = glob.glob(path+"/"+str(number)+".png")
    if len(name)>1:
        print "too many matches ",len(name)," found"
    name = name[0]
    img = Image.open(name)
    img = np.asarray(img)
    img.setflags(write=1)
    return img

def filter_by_size(img_segm,mini_nb_pix):
    """filters a segmented image by getting rid of the components with too few pixels"""
    numbers = np.zeros(int(np.max(img_segm)))
    for i in range(1,int(np.max(img_segm))+1):
        numbers[i-1] = np.count_nonzero(img_segm==i)
        
    indexes = np.arange(1,np.max(img_segm)+1)
    #indexes = indexes[numbers>np.mean(numbers)] #Deletes the 1-pixel elements
    indexes = indexes[numbers>mini_nb_pix] #Deletes the 1-pixel elements
    
    segm_filtered = np.zeros(img_segm.shape)
    j=1
    for i in (indexes):
        segm_filtered[img_segm==i] = j
        j+=1
    return segm_filtered

def fillHoles(img):
    """Fills the holes in a segmented image using contour detection and filling"""
    out,contour,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    i=0
    for cnt in contour:
            cv2.drawContours(img,contour,i,255,-1)
            i+=1
    return img

def modulusMaximum(amplitude,angle):
    """Unused method which computes for each pixel if it is the maximum of its neighbors
    along the axis defined by angle"""
    angle = np.mod(angle+np.pi,np.pi)/np.pi  #The sign does not matter anyway, goes between 0 and 1
    
    #Do 4 cases: angle <0.25,0.5,0.75,>0.75
    results_plus = np.zeros(amplitude.shape,dtype = np.uint8)
    results_minus = np.zeros(amplitude.shape,dtype = np.uint8)
    
    ampl_pad = np.pad(amplitude,1,'constant')
    
    #Case angle is between 0 and 45deg, ie angle is between 0 and 0.25
    print "max angle",np.max(angle)
    tmp = amplitude>ampl_pad[1:-1,2:]
    results_plus[angle<0.25] = tmp[angle<0.25]
    tmp = amplitude>ampl_pad[1:-1,:-2]
    results_minus[angle<0.25] = tmp[angle<0.25]
    
    #Case angle between 0.25 and 0.5
    tmp = amplitude>ampl_pad[:-2,2:]
    results_plus[np.logical_and(angle>=0.25,angle<0.5)] = tmp[np.logical_and(angle>=0.25,angle<0.5)]
    tmp = amplitude>ampl_pad[2:,:-2]
    results_minus[np.logical_and(angle>=0.25,angle<0.5)] = tmp[np.logical_and(angle>=0.25,angle<0.5)]
       
    #Case angle between 0.5 and 0.75
    tmp = amplitude>ampl_pad[2:,1:-1]
    results_plus[np.logical_and(angle>=0.5,angle<0.75)] = tmp[np.logical_and(angle>=0.5,angle<0.75)]
    tmp = amplitude>ampl_pad[:-2,1:-1]
    results_minus[np.logical_and(angle>=0.5,angle<0.75)] = tmp[np.logical_and(angle>=0.5,angle<0.75)]
    
    #Case angle >0.75
    tmp = amplitude>ampl_pad[:-2,:-2]
    results_plus[angle>=0.75] = tmp[angle>=0.75]
    tmp = amplitude>ampl_pad[2:,2:]
    results_minus[angle>=0.75] = tmp[angle>=0.75]
    
    return np.logical_and(results_plus,results_minus)

def saveFrame(name,total):
    if total.dtype!='uint8':
        total = total/np.max(total)*255
        total = total.astype(np.uint8)
    cv2.imwrite(name,total)


def classifyPhaseImage(fr_nb):
    """Uses K means algorithm to find 3 clusters in a phase image, hopefully
    corresponding to background, axons an chunks of neurons"""
    phase_path = os.path.join("..",'data','microglia','Beacon-1 unst',"Scene1Interval"+str(fr_nb)+"_PHASE.png")
    
    phase= Image.open(phase_path)
    phase = np.asarray(phase)
    X=phase.reshape(-1,1)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3).fit(X)
    classified = kmeans.labels_
    classified=classified.reshape(phase.shape)
    si2(phase,classified,"Phase image","Classification")
    return classified
    
#------------------Preprocessing-------------------------------
def displayWlt(wlt):
    """Displays a single unit wavelet decomposition in one image"""
    cA, (cH, cV, cD)=wlt
    shapes = (cA.shape[0]+cV.shape[0],cA.shape[1]+cV.shape[1])
    out = np.zeros(shapes)
    out[:cA.shape[0],:cA.shape[1]]=cA
    out[cA.shape[0]:,cA.shape[1]:]=cD
       
    out[:cH.shape[0],cA.shape[1]:]=cH
    out[cA.shape[0]:,:cV.shape[1]]=cV
    return out

def displayMulti(wlt_list):
    """Dsiplay a cascade of wavelets on only one image."""
    cA = wlt_list[0]
    
    for i in range(1,len(wlt_list)):
        print i, cA.shape,wlt_list[i][0].shape
        cA = displayWlt((cA,wlt_list[i]))
    plt.figure()
    plt.imshow(cA,cmap="gray")
    plt.title("Wavelet decomposition")
    
def abe(img,variance):
    """proceeds to the Amplitude-scale invariant Bayes Estimation (ABE)"""
    nominator = img**2-3*variance
    nominator[nominator<0] = 0
    out = np.divide(nominator,img)
    out[img==0]=0
    return out

def getNoiseVar(img,fraction=0.95):
    """Gets the nth% of lower intensity pixels in an image correspondin to the noise
    n is determined empirically."""
    last_val = np.percentile(img,fraction)
    #si(img<last_val,title="Pixel values considered as noise")
    return np.var(img[img<last_val])

def wavelet_denoising2(im,wlt='sym2',lvl=5,fraction=0.76):
    coeffs_trous = pywt.swt2(im,wlt,lvl,start_level=0)
    total = np.ones(im.shape)
    #Add Gaussian blur
    for elts in coeffs_trous:
        cA,(cH,cV,cD) = elts
        var = getNoiseVar(cA,fraction)        
        cA = abe(cA,var)
        #m.si(tata)
        total*=cA
    return total

def gaussian(size,sigma):
    """Generates a square gaussian mask with size*size pixels and std sigma"""
    a,b=np.ogrid[-size/2:size/2,-size/2:size/2]
    mask = a**2+b**2
    mask = np.exp(-mask.astype('float')/(2*float(sigma**2)))
    return mask

def denoiseStack(path,target_dir):
    elements = glob.glob(path+"/*RFP.png")
    if platform.system()=='Windows':
        separator="\\"
    else:
        separator="/"
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    for elt in elements:
        print "processing",elt.split(separator)[-1]
        img = Image.open(elt)
        img = np.asarray(img)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(30,30))
        cl1 = clahe.apply(img)
        cl2 = clahe.apply(cl1)
        cl2 = wavelet_denoising2(cl2,lvl=3)
        cl2 = cl2*255/np.max(cl2)
        cl2 = cl2.astype(np.uint8)
        cv2.imwrite(os.path.join(target_dir,'filtered_'+elt.split(separator)[-1]),cl2)
        
#---------------------Shape-related functions--------------------------------------
def segmentation(total):
    t = skimage.filters.threshold_li(total)
    mask = (total>t).astype(np.uint8)
    #mask = cv2.dilate(mask,np.ones((4,4)),iterations = 1)
    kernel = np.ones((5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #m.fillHoles(mask)
    si(mask)
    mask,nr = ndi.label(mask)
    mask = filter_by_size(mask,500)
    return mask.astype(np.uint8)

def hysteresis_thresholding(image,th1,th2):
    """Proceeds to hysteresis thresholding of image. Two thresholds are used:
        A strong one: all pixels above this value are considered as signal
        A weak one: pixels whose value is above the small threshold but below
        the strong threshold are selected only if they are in contact with pixels
        selected by the strong threshold."""
    hard_th = (image>max(th1,th2)).astype(np.uint8)*255
    soft_th = (image>min(th1,th2)).astype(np.uint8)*255
    added_elements = np.zeros(image.shape,dtype=np.uint8)
    
    labels,nr = ndi.label(soft_th-hard_th)
    kernel = np.ones((2,2),dtype=np.uint8)
    hard_th_exp = skimage.morphology.binary_dilation(hard_th,selem=kernel)
    connected_labels = np.unique( labels[np.logical_and(labels,hard_th_exp)] )
    connected_labels = [x for x in connected_labels if x>0]
    for lab in connected_labels:
        added_elements[labels==lab]=255
    return added_elements+hard_th

def find_local_minima(image):
    """Finds local minima in segmeted image (holes)"""
    mini_mask = skimage.morphology.disk(3)
    img = skimage.filters.gaussian(image,0.5)
    thresh = (img>skimage.filters.threshold_li(img))
    filtered = ndi.filters.minimum_filter(img,footprint=mini_mask)
    is_local_minimum = filtered==img
    is_local_minimum = np.logical_and(is_local_minimum,thresh)
    return is_local_minimum

def findHoles(image):
    """Finds local minima in an image and counts them as holes if they are not 
    within a gaussian"""
    local_minima=find_local_minima(image)
    mask_gaussians=where_are_gaussians(image)
    im_holes=show_holes_on_img(np.logical_and(local_minima,~mask_gaussians),image)
    local_minima = np.logical_and(local_minima,~mask_gaussians)
    local_minima = local_minima.astype(np.uint8)
    return local_minima,im_holes

def find_arms(path,fr_nb):
    """Find the cell arms in the image number nb located in path. First thresholds,
    than openc to remove the arms an then does the difference.
    Returns mask with only the massive objects (centers) and arms"""
    im=open_frame(path,fr_nb)
    img=im.copy()
    im=img_as_ubyte(im)
    mask_h = hysteresis_thresholding(img,6,10)
    
    ksize=5
    kernel = np.ones((ksize,ksize),dtype = np.uint8)
    kernel = skimage.morphology.disk(ksize)
    
    mask = cv2.morphologyEx(mask_h, cv2.MORPH_OPEN, kernel,iterations=2)
    
    arms = mask_h-mask
    """
    lab,_ = ndi.label(diff)
    
    arms = skimage.morphology.remove_small_objects(lab,60)"""   #Only temporary, to track only the biggest
    return mask,arms

#----------------Handling gaussians------------------------------------------
def try_template_matching(image,template):
    """Tries template matching for template in image using 6 different methods and
    plots the result"""
    img2 = image.copy()
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        res-=np.min(res)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        threshold=0.9*np.max(res)
        
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            loc = np.where( res <=0.1*np.max(res))
        else:
            loc = np.where( res >= threshold)
        
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 255, 2)
        plt.figure()
        plt.subplot(121),plt.imshow(image,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()

def find_gaussian(img,sigma=25):
    """Optimal template matching to find a gaussian with std sigma in img"""
    method = 'cv2.TM_CCOEFF_NORMED'
    size=3*sigma
    template = gaussian(size,sigma)
    template/=template.max()
    template*=255
    template = template.astype(np.uint8)
    
    threshold = 0.9
    w, h = template.shape[::-1]
    
    img2 = img.copy()
    meth = eval(method)

    # Apply template Matching
    res = cv2.matchTemplate(img2,template,meth)
    #Filters location map so that only one gaussian is found per contiguous location
    location_map =  res >= threshold*np.max(res)
    location_map,nr = ndi.label(location_map)
    list_x = []
    list_y = []
    for label in range(1,nr+1):
        tmp=location_map==label
        if np.count_nonzero(tmp)>1:
            points = np.where(tmp)
            l = len(points[0])
            cx = (np.sum(points[0]) + l/2)/l
            cy = (np.sum(points[1]) + l/2 )/l
            list_x.append(cx)
            list_y.append(cy)
    loc= (np.asarray(list_x),np.asarray(list_y))
    stack_to_remove = np.zeros((size,size,len(loc[0])))
    i=0
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img2, pt, (pt[0] + w, pt[1] + h), 255, 2)
        stack_to_remove[:,:,i] = img[pt[1]:pt[1]+w,pt[0]:pt[0]+h]
        i+=1
    return stack_to_remove,loc

def where_are_gaussians(img):
    """Finds gaussians in img and returns a mask of pixels in a gaussian."""
    list_of_sigmas = [40,30,20,10]
    mask=np.zeros(img.shape,dtype=bool)
    for sigma in list_of_sigmas:
        stack_to_remove,locs=find_gaussian(img.astype(np.uint8),sigma)
        w = stack_to_remove[:,:,0].shape[0]
        a,b=np.ogrid[-w/2:w/2,-w/2:w/2]
        for i in range(stack_to_remove.shape[2]):
            pt=(locs[0][i],locs[1][i])
            mask[pt[0]:pt[0]+w,pt[1]:pt[1]+w] = True
    return mask

def gaussian_proba_map(img):
    """Finds the probability that each point in img is a gaussian
    returns an image with same dimensions"""
    method = 'cv2.TM_CCOEFF_NORMED'
    sigmas = [41,31,21,11]
    out = np.zeros(img.shape)
    for sigma in sigmas:
        size=3*sigma
        template = gaussian(size,sigma)
        template/=template.max()
        template*=255
        template = template.astype(np.uint8)
        
        img2 = img.copy()
        meth = eval(method)
        # Apply template Matching
        res = cv2.matchTemplate(img2,template,meth)
        res = np.pad(res,size/2,mode='constant')
        to_replace = res>out
        out[to_replace] = res[to_replace]
    return out

#-------------Display functions---------------------------------------

def si(img,title=None):
    plt.figure()
    plt.imshow(img,cmap='gray')
    if title:
        plt.title(title)
        
def si2(im1,im2,title1=None,title2=None):
    plt.figure()
    plt.subplot(121)
    plt.imshow(im1)
    if title1:
        plt.title(title1)
    plt.subplot(122)
    plt.imshow(im2)
    if title2:
        plt.title(title2)
        
def show_points_on_img(mask,img):
    """Shows the points encoded in mask on img"""
    labeled, num_objects = ndi.label(mask)
    slices = ndi.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    plt.figure()
    plt.imshow(img)
    plt.autoscale(False)
    plt.plot(x,y, "o")

def show_holes_on_img(mask,img):
    """Shows the points encoded in mask on img"""
    labeled, num_objects = ndi.label(mask)
    slices = ndi.find_objects(labeled)
    radius=9
    out_image = img.copy()
    out_image = cv2.cvtColor(out_image, cv2.COLOR_GRAY2RGB)
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2    
        center=(x_center,y_center)
        cv2.circle(out_image, center, radius,(111,17,108),thickness=2)

    plt.figure()
    plt.imshow(out_image)
    plt.autoscale(False)
    return out_image
# display results
def overlay_mask2image(img,mask,title=None):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    #ax = axes.ravel()
    ax = axes
    ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax.imshow(mask, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
    if title:
        ax.set_title(title)
    
    fig.tight_layout()
    plt.show()

def cv_overlay_mask2image(mask,img,color="green"):
    """Overlay a mask to an image using opencv"""
    transparency=0.2
    if mask.dtype==np.int:
        mask = mask>0
        mask = mask.astype(np.uint8)*255
    if mask.dtype==bool:
        mask = mask.astype(np.uint8)*255
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    if len(img.shape)==2:
        image=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    else:
        image=img.copy()
    if color=="green":
        mask[:,:,0]=0
        mask[:,:,2]=0
    else:
        mask[:,:,0]=0
        mask[:,:,1]=0
    cv2.addWeighted(mask,transparency,image,1-transparency,0,image)
    return image


#-------------------------Tracking----------------------------------------------
def centroids(img_segm):
    """Computes the positions of centroids in a segmented image.
    Returns the centroids numbers in order"""
    m = int(np.amax(img_segm))
    xs = np.zeros(m)
    ys = np.zeros(m)
    
    for i in range(0,m):
        pos_list = np.where(img_segm==i+1)
        xs[i] = np.mean(pos_list[0])
        ys[i] = np.mean(pos_list[1])
    return xs,ys

def fillCostMatrix(xs0,ys0,xs1,ys1):
    """Computes a cost matrix for the different distances between centroids
    The rows represent the centroids in the previous frame
    The columns the centroids in the next frame
    Returns the cost matrix"""
    M = int ( max(len(xs0),len(xs1)) ) #Number of centroids.
    costMatrix = np.ones((M,M))*-1
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

def w_hungarian(prev_centroids,next_centroids,max_distance=50):
    """Using the hungarian algorithm, looks for corresponding cells between 
    prev_image and next_image. If a distance is greater than max_distance, it is considered
    as two different cells.
    When a cell is associated with -1, it means that it is not associated with anything yet"""
    xs0,ys0 = prev_centroids
    xs1,ys1 = next_centroids

    cost_matrix = fillCostMatrix(xs0,ys0,xs1,ys1)
    #make it disatvantageous to select any distance >max_distance
    cost_matrix[cost_matrix>(max_distance**2)]=np.max(cost_matrix)
    cost_matrix[cost_matrix==-1]=np.max(cost_matrix)
    xs,ys = linear_sum_assignment(cost_matrix)
    
    correspondance_list=[]
    for i in range(xs.size):
        correspondance_list.append( (xs[i],ys[i]) )
    apparition_list = [] 
    elements_to_remove = []  
    for i, coords in enumerate(correspondance_list):
        if cost_matrix[coords]>max_distance**2:
            if coords[0]<len(xs0):   #the left element exists
                correspondance_list[i] = (coords[0],-1)
            else:   #the left element does not exists
                elements_to_remove.append(i)
            if coords[1]<len(xs1):   #Add the right element only if it exists.
                apparition_list.append((-1,coords[1]))
    for j in range(len(elements_to_remove)):
        correspondance_list.pop(elements_to_remove[-(j+1)])
    correspondance_list.extend(apparition_list)
    return correspondance_list

class FrameInfo(object):
    """Class remembering the information from a labeled frame.
    All indexes should start with 0"""
    def __init__(self,frame_nr,frame):
        self.nr = frame_nr
        self.n_objects = int(np.max(frame))
        self.objects_size = []
        self.centroids = centroids(frame)
    def __str__(self):
        return "Frame N:"+str(self.nr)+" Nr objects: "+str(self.n_objects)
#-------Just to wait---
def find_cells():
    """Just to wait"""
    pass        
def openImage():
    pass
#-----End just to wait----

class Tracker(object):
    """Object iterating through a set of frames contiguous in time.
    Segments each frame and looks for correspondances in the previous frame
    usin the hungarian algorithm. Stores the main results in info_list and
    correspondance_lists
    
    We use here a quite heavily filtered by size image, as we are not interested 
    in small pieces here for main tracking"""
    def __init__(self,path=None,n_frames=241):
        self.info_list=[]
        self.correspondance_lists=[]
        self.first_frame = 0
        self.last_frame = 240
        self.n_frames=n_frames
        
        if path:
            self.path = path
        else:
            self.path = os.path.join("..",'data','microglia','RFP1_cropped')
        
    def preprocess(self):
        prev_centroids = self.info_list[0].centroids
        
        for i in range(1,241):
            print "Tracking iteration ",i
            next_centroids = self.info_list[i].centroids
            match_list = w_hungarian(prev_centroids,next_centroids)
            prev_centroids = next_centroids[:]
            self.correspondance_lists.append(match_list)
            
    def next_cell(self,frame,label):
        """Finds for the cell label in frame, the label of the same cell in the next cell"""
        elements = [y for x,y in self.correspondance_lists[frame] if x==label]
        if frame>len(self.correspondance_lists) or frame<0:
            raise ValueError('Trying to access an element outside the video range')
        if len(elements)>1:
            raise IndexError('Tracker found more than one match for label '+str(label)+' in frame '+str(frame))
        if len(elements)==0:
            raise IndexError('Tracker could not find any match for label '+str(label)+' in frame '+str(frame))
        return elements[0]
    
    def prev_cell(self,frame,label):
        """Finds for the cell label in frame, the label of the same cell in the previous cell"""
        elements = [x for x,y in self.correspondance_lists[frame-1] if y==label]
        
        if frame>len(self.correspondance_lists)+1 or frame<1:
            raise ValueError('Trying to access an element outside the video range')
        if len(elements)>1:
            print elements
            raise IndexError('Tracker found more than one match for label '+str(label)+' in frame '+str(frame))
        if len(elements)==0:
            raise IndexError('Tracker could not find any match for label '+str(label)+' in frame '+str(frame))
        return elements[0]
    
    def segment(self,path=None):
        """Uses the information extracted fron the preprocessing step to segment
        the images"""
        if path==None:
            path = path = os.path.join("..",'data','microglia','RFP1_cropped_segmented')
        
        first_labels = find_cells(self.first_frame)
        
        labels_buffer = np.zeros((first_labels.shape[0],first_labels.shape[1],2))
        labels_buffer[:,:,self.first_frame%2]=first_labels
        
        current_index=0   #Index in the referential of Tracker (equal to 0 at self.first_frame)
        for i in range(self.first_frame+1,self.last_frame+1):
            
            labels=find_cells(i)
            labels_buffer[:,:,i%2]= labels
            correspondance = self.correspondance_lists[current_index]
            disappear=[]    # Elements disappearing in the frame before
            appear = []   #Elements appearing in the current frame
            for bef,aft in correspondance:
                print bef,aft
                if bef==-1:
                    appear.append(aft)
                if aft==-1:
                    disappear.append(bef)
            current_index+=1
            # If there are cells disappearing:
            for index in disappear:
                elts_in_contact = labels[labels_buffer[:,:,(i-1)%2]==(index+1)]
                candidates = np.unique(elts_in_contact)
                candidates = [x for x in candidates if x!=0]
                if len(candidates)==1:
                    print "found who it disappeared for"
                else:
                    print "error several candidates index ",i
                
    def find_indices(self,nr_frame,label,forward=True):
        """For a given cell in a given frame, gets the corresponding indices
        in the next (if forward=true) or the previous frames"""
        if forward:
            index = nr_frame-self.first_frame
            label_list=[label]
            #Fetches the 10 first frames. 10 is arbitrary
            n_iterations = min(10,len(self.correspondance_lists)-index-1 )
            for i in range(n_iterations):
                corresp_list = self.correspondance_lists[index+i]
                match = [v for u,v in corresp_list if u==label_list[index+i]]
                match = match[0]
                if match==-1:
                    break
                
                label_list.append(match)
            return label_list
        
        else:
            index = nr_frame-self.first_frame
            label_list=[label]
            #Fetches the 10 first frames. 10 is arbitrary
            n_iterations = min(10,index )
            for i in range(n_iterations):
                corresp_list = self.correspondance_lists[index-i]
                match = [u for u,v in corresp_list if v==label_list[index-i]]
                match = match[0]
                if match==-1:
                    break
                label_list.append(match)
            return label_list
                
    def showTrajectory(self,cell_of_interest=5,overlay=False,plot=False,wait=50):
        #Print evolution of different parameters for one sinle cell
        
        sizes = []
        sizes.append(self.info_list[0].objects_size[cell_of_interest])
        info_index=1
        cell_list=[cell_of_interest]
        for elements in self.correspondance_lists:
            corresp = -1
            for (u,v) in elements:
                if u==cell_of_interest:
                    corresp = v
            if corresp==-1:
                print "Correspondace lost."
                print "Helloooooo"
                break
            cell_of_interest = corresp
            if cell_of_interest>=len(self.info_list[info_index].objects_size):
                print "Target lost"
                break
            cell_list.append(cell_of_interest)
            sizes.append(self.info_list[info_index].objects_size[cell_of_interest])
            info_index+=1
            
        if plot:
            plt.figure()
            plt.plot(sizes)
            plt.title("Evolution of the size of a cell")
        
        speeds = []
        index=0
        xs=0
        ys=0
        speed=0
        for i in cell_list:
            lab = find_cells(self.first_frame+index)
            #Get centroid
            pos_list = np.where(lab==i+1)
            if index>0:
                speed = np.sqrt( (xs-np.mean(pos_list[0]))**2 + (ys-np.mean(pos_list[1]))**2 )
                speeds.append(speed)
            xs = np.mean(pos_list[0])
            ys = np.mean(pos_list[1])
            if overlay:
                im=openImage(self.first_frame+index)
                im.setflags(write=1)
                im[lab!=i+1]=0
                cv2.imshow("frame",im)
            else:
                cv2.imshow("frame",(lab==i+1).astype(np.float))
            cv2.waitKey(wait)
            index+=1
        plt.figure()
        plt.plot(speeds)
        plt.title("Motion of the detected cell in pixels")
            
    def showMovie(self,first_frame = -1, last_frame = -1,wait=50):
        if first_frame<0:
            first_frame = self.first_frame
        if last_frame<0:
            last_frame = self.last_frame
        for i in range(first_frame,last_frame+1):
            cv2.imshow("Not processed movie",openImage(i))
            cv2.waitKey(wait)
            
    def showLabels(self,first_frame = -1, last_frame = -1,wait=50):
        if first_frame<0:
            first_frame = self.first_frame
        if last_frame<0:
            last_frame = self.last_frame
        for i in range(first_frame,last_frame+1):
            cv2.imshow("Not processed movie",find_cells(i))
            cv2.waitKey(wait)