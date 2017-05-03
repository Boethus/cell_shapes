# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 12:44:46 2014

@author: Michael
"""
import random
import matplotlib.pyplot as plt

import numpy as np
from scipy import misc
from scipy import sparse
import time

print "Hello. Are you still there?"
def neighbourhood24_of_pixel(imagedata, x, y):
    w = imagedata.shape[0]    
    h = imagedata.shape[1] 
    
    neighbours = []
    for i in xrange(x-2,x+2):
        for j in xrange(y-2,y+2):
            if not (i == x and j == y) and i >= 0 and j >= 0 and i < w and j < h:
                neighbours.append((i,j))    
    return neighbours

def neighbourhood8_of_pixel(imagedata, x, y):
    w = imagedata.shape[0]    
    h = imagedata.shape[1] 
    
    if x == 0 or y == 0 or x == w - 1 or y == h-1:
        neighbours2 = ((x-1,y),(x,y-1),(x+1,y),(x,y+1),(x-1,y-1),(x-1,y+1),(x+1,y-1),(x+1,y+1))
        neighbours = []
        for n in neighbours2:
           if n[0] >= 0 and n[0] < w and n[1] >= 0 and n[1] < h:
               neighbours.append(n)
        return neighbours
    else:
        return  ((x-1,y),(x,y-1),(x+1,y),(x,y+1),(x-1,y-1),(x-1,y+1),(x+1,y-1),(x+1,y+1))

def neighbourhood4_of_pixel(imagedata, x, y):
    w = imagedata.shape[0]    
    h = imagedata.shape[1] 
    
    if x == 0 or y == 0 or x == w - 1 or y == h-1:
        neighbours2 = ((x-1,y),(x,y-1),(x+1,y),(x,y+1))
        neighbours = []
        for n in neighbours2:
           if n[0] >= 0 and n[0] < w and n[1] >= 0 and n[1] < h:
               neighbours.append(n)
        return neighbours
    else:
        return  ((x-1,y),(x,y-1),(x+1,y),(x,y+1))

        
def get_adjacency_matrix(imagedata, neighbourfunction):
    shape = (imagedata.shape[0]*imagedata.shape[1], imagedata.shape[0]*imagedata.shape[1])
    row=[]    
    col=[]
    data=[]    
    for x in xrange(imagedata.shape[0]):
        for y in xrange(imagedata.shape[1]):
            index = x*imagedata.shape[1] + y                    
            for neighbour in neighbourfunction(imagedata,x,y):
                row.append(index)
                col.append(neighbour[0]*imagedata.shape[1]+neighbour[1])
                data.append(1)
    return sparse.csr_matrix( (data,(row,col)),shape=shape,dtype=np.dtype("int8"))

#-----------------------------My Scripts------------------------

def computePixelLikelihood2(y,x):
	"""This function computes the pixel-by pixel likelihood of the noisy image y with the
	hidden state x"""
	middle_value = 125.5
	high_proba = 0.85
	low_proba = 0.25

	x_boolean = x.astype(np.bool)

	likelihood = np.zeros(y.shape) #Likelihood will contain the likelihood map

	#filling the map in the case x is high
	likelihood[x_boolean] = (y[x_boolean]>middle_value)*high_proba + ( y[x_boolean]<middle_value )*low_proba
	
	x_boolean = ~x_boolean
	
	#filling the map in the case x is low
	likelihood[x_boolean] = (y[x_boolean]>middle_value)*low_proba + ( y[x_boolean]<middle_value )*high_proba
	return likelihood
 
def computePixelLikelihood3(y,x,factor = 1):
    """This function computes the pixel-by pixel likelihood of the noisy image y with the
    hidden state x, using arctan"""
    x_boolean = x.astype(np.bool)
    
    likelihood = np.zeros(y.shape) #Likelihood will contain the likelihood map
    im=y.astype(np.float64)
    im = im/np.max(im)
    im=im-0.5
    #filling the map in the case x is high
    likelihood[x_boolean] = np.arctan(im[x_boolean]*factor)/np.pi*0.7+0.5
    likelihood[~x_boolean] = np.arctan((-im)[~x_boolean]*factor)/np.pi*0.7+0.5
    """plt.figure()
    lik=likelihood.reshape(-1)
    lik=lik.astype(np.float64)
    plt.hist(lik,bins=20)
    plt.show()"""
    return likelihood 
        
def gibbsUpdate(imageData, hiddenStates, neighbourMatrix,beta=0.01):
    """Computes an iteration of the Gibbs update Algorithm. imageData and hidden 
    states are 2 dimensional arrays with the same shape. neighbourMatrix is also 
    2-dimensional and is sparse. This method works for 4 neighbors and the number of 
    neighbors has to be manually changed in case of need."""

    states = hiddenStates.reshape(-1)          

    #We obtain the number of neighbours of eqch pixel by summing the neighbour matrix along
    #its rows.
    nb_neighbours = neighbourMatrix.sum(axis=1)       
    nb_neighbours = nb_neighbours.getA()        #Conversion into a numpy array      
    nb_neighbours = nb_neighbours.reshape(-1)
                                                                                                                                                                                                                                                                                                                                     
	#E0 and E1 contain the number of neigbors similar
    #to each pixel if these pixels are equal to                                                             respectively 0 or one, multiplied by beta.
    E0 = beta * (nb_neighbours-neighbourMatrix*states)
    E1 = beta * (neighbourMatrix*states)              
    
	#lnF0 and lnF1 contain the pixel likelihood if each pixel is respectively equal to 0 and 1.
    lnF0 = computePixelLikelihood3(imageData,np.zeros(imageData.shape))
    lnF0 = lnF0.reshape(-1)
    lnF1 = computePixelLikelihood3(imageData,np.ones(imageData.shape))
    lnF1 = lnF1.reshape(-1)

	#p0 and p1 are the arrays containing the probability that each pixel is respectively equal to 0 or 1
    p0 = lnF0*np.exp(E0)
    p1 = lnF1*np.exp(E1)

	#Selecting the more likely value for each pixel
    solution = np.zeros(imageData.shape[0]*imageData.shape[1])
    """
    solution[p0>p1]=0
    solution[p0<=p1]=1"""
    proba_map = p1/(p0+p1)
    variable = np.random.rand(imageData.shape[0]*imageData.shape[1])
    solution[variable<proba_map]=1
    
    solution.reshape(imageData.shape)
            
    return solution;
    
    
def gibbs_iterative(imageData,hiddenStates,neighbourMatrix,iterations,beta=0.01):
    solution = np.copy(hiddenStates)
    solution2 = np.zeros(solution.shape)
    for i in range(iterations):
        solution = gibbsUpdate(imageData, solution, neighbourMatrix,beta)
        if i>0.2*iterations:
            solution2+=solution/iterations
    return solution2
    
def normrand(pos,step):
    """returns a new position in a random walk with step step."""
    return pos+(np.random.rand()*2-1)*step
    
def metropolis(hiddenStates, neighbourMatrix,n_iter):
    """
    for i = 2:N
    newLoc = normrnd(samples(i-1),step);
    if rand < realDistribution(newLoc)/realDistribution(samples(i-1))
        samples(i) = newLoc;
    else
        samples(i) = samples(i-1);"""
       
    beta = 0.7
    step = 0.05
    states = hiddenStates.reshape(-1)          

    #We obtain the number of neighbours of eqch pixel by summing the neighbour matrix along
    #its rows.
    nb_neighbours = neighbourMatrix.sum(axis=1)       
    nb_neighbours = nb_neighbours.getA()        #Conversion into a numpy array      
    nb_neighbours = nb_neighbours.reshape(-1)
                                                                                                                                                                                                                                                                                                                                     
	#E0 and E1 contain the number of neigbors similar
    #to each pixel if these pixels are equal to 
    # respectively 0 or one, multiplied by beta.
    betas = np.zeros((n_iter,1))        
    betas[0]=beta  
    probas = np.zeros((n_iter,1))
    
    #Initialization: calculation of the probability in the first step
    E0 = beta * (nb_neighbours-neighbourMatrix*states)
    E1 = beta * (neighbourMatrix*states)
    p0 = np.exp(E0)
    p1 = np.exp(E1)

    proba_matrix = np.copy(p1)
    proba_matrix[states==0]=p0[states==0]
    proba_matrix = proba_matrix.astype(np.float)/(p0+p1)
    """
    plt.figure()
    plt.subplot(131)
    plt.imshow(p0.reshape(imageData.shape))
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(p1.reshape(imageData.shape))
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(proba_matrix.reshape(imageData.shape))
    plt.colorbar()"""
    prev_matrix = proba_matrix.astype(np.float128)
    p_tild=np.product(proba_matrix) 
    #/factor**(states.shape[0])
    print p_tild
    probas[0] = p_tild
                                                                                                                                                                                                                                                                
    for i in range(1,n_iter): 
        
        newBeta = normrand(betas[i-1],step)
        E0 = newBeta * (nb_neighbours-neighbourMatrix*states)
        E1 = newBeta * (neighbourMatrix*states)

    	#lnF0 and lnF1 contain the pixel likelihood if each pixel is respectively equal to 0 and 1.
    
    	#p0 and p1 are the arrays containing the probability that each pixel is respectively equal to 0 or 1
        p0 = np.exp(E0)
        p1 = np.exp(E1)
    
        proba_matrix = np.copy(p1)
        proba_matrix[states==0]=p0[states==0]
        proba_matrix = proba_matrix/(p0+p1)
        p_tild=np.product(proba_matrix.astype(np.float128)/prev_matrix)
        probas[i] = p_tild
        print p_tild
        if np.random.rand()<p_tild:
            betas[i] = newBeta
            prev_matrix = proba_matrix
        else:
            betas[i] = betas[i-1]
        
            
    return betas,probas
    
    
def metropolis_optimisation(imageData,hiddenStates, neighbourMatrix,n_iter):
       
    step = 0.05
    states = hiddenStates.reshape(-1)          
    solution2 = np.zeros(states.shape)
    #We obtain the number of neighbours of eqch pixel by summing the neighbour matrix along
    #its rows.
    nb_neighbours = neighbourMatrix.sum(axis=1)       
    nb_neighbours = nb_neighbours.getA()        #Conversion into a numpy array      
    nb_neighbours = nb_neighbours.reshape(-1)
         
    n_iter_beta = 400                                                                                                                                                                                                                                                                                                                            

    betas = np.zeros((n_iter,1))        
    betas[0]= np.random.rand()
    probas = np.zeros((n_iter,1))
    total_betas=[]
    
    for i in range(1,n_iter): 
        betas_inside = np.zeros((n_iter_beta,1))
        betas_inside[0] = np.random.rand()*1.4
        
        E0 = betas_inside[0] * (nb_neighbours-neighbourMatrix*states)
        E1 = betas_inside[0] * (neighbourMatrix*states)
        p0 = np.exp(E0)
        p1 = np.exp(E1)
    
        proba_matrix = np.copy(p1)
        proba_matrix[states==0]=p0[states==0]
        proba_matrix = proba_matrix.astype(np.float)/(p0+p1)
    
        prev_matrix = proba_matrix.astype(np.float128)
        p_tild=np.product(proba_matrix)         
        
        for j in range(1,n_iter_beta):
            """Looking for values of Beta"""
            newBeta = normrand(betas_inside[j-1],step)
            E0 = newBeta * (nb_neighbours-neighbourMatrix*states)
            E1 = newBeta * (neighbourMatrix*states)

            p0 = np.exp(E0)
            p1 = np.exp(E1)
        
            proba_matrix = np.copy(p1)
            proba_matrix[states==0]=p0[states==0]
            proba_matrix = proba_matrix/(p0+p1)
            p_tild=np.product(proba_matrix.astype(np.float128)/prev_matrix)
            probas[i] = p_tild
            if np.random.rand()<p_tild:
                betas_inside[j] = newBeta
                prev_matrix = proba_matrix
            else:
                betas_inside[j] = betas_inside[j-1]
                
        print "iteration",i,"Beta value:",betas_inside[n_iter_beta-1],"mean",np.mean(betas_inside)
        total_betas = np.append(total_betas,betas_inside)
        betas[i]=betas_inside[-1]
        E0 = betas[i] * (nb_neighbours-neighbourMatrix*states)
        E1 = betas[i] * (neighbourMatrix*states)              
    
        #lnF0 and lnF1 contain the pixel likelihood if each pixel is respectively equal to 0 and 1.
        """--------------------------------------------------"""
        lnF0 = computePixelLikelihood3(imageData,np.zeros(imageData.shape))
        lnF0 = lnF0.reshape(-1)
        lnF1 = computePixelLikelihood3(imageData,np.ones(imageData.shape))
        lnF1 = lnF1.reshape(-1)
    
    	#p0 and p1 are the arrays containing the probability that each pixel is respectively equal to 0 or 1
        p0 = lnF0*np.exp(E0)
        p1 = lnF1*np.exp(E1)
    
    	#Selecting the more likely value for each pixel
        solution = np.zeros(imageData.shape[0]*imageData.shape[1])

        proba_map = p1/(p0+p1)
        variable = np.random.rand(imageData.shape[0]*imageData.shape[1])
        solution[variable<proba_map]=1
        states = solution
        solution2+=states/n_iter
    return total_betas,probas,solution2
"""


test = False
# Initialise
if test:
    imagedata = np.array([[0, 0, 0, 0, 0],[0, 0, 1, 0, 0],[0, 1, 1, 1, 0],[0, 0, 1, 0, 0],[0, 0, 0, 0, 0]]);
    hiddenstates = np.array([0, 0, 0.5, 0, 1, 0, 0, 0, 0, 1, 0.5, 0, 1, 1, 0, 1, 1, 0.5, 0, 1, 0, 0, 0, 0, 1])
else:
    imagedata = misc.imread('large.png')
    hiddenstates = np.array([random.randint(0, 1) for i in xrange(imagedata.shape[0]*imagedata.shape[1])], dtype=np.dtype("int8"))        
"""

""" returns an adjacency matrix where each row i indexes a pixel and each non-zero index j indicates that j is a neighbour of pixel i """
#neighbourmatrix = get_adjacency_matrix(imagedata, neighbourhood4_of_pixel)
#E = computeLogPosterior(imagedata, hiddenstates, neighbourmatrix)


#logPos = computeLogPosterior(imagedata, hiddenstates, neighbourmatrix)


""" plot the result """
"""
def showGibbs():
    plt.figure()
    plt.gray()
    plt.subplot(2,2,1),plt.imshow(imagedata,'gray')
    plt.subplot(2,2,2),plt.imshow(hiddenstates.reshape(imagedata.shape),'gray')
    hiddenstates2 = gibbsUpdate(imagedata, np.copy(hiddenstates), neighbourmatrix)
    plt.subplot(2,2,3),plt.imshow(hiddenstates2.reshape(imagedata.shape),'gray')
    hiddenstates3 = gibbs_iterative(imagedata, np.copy(hiddenstates2), neighbourmatrix,20)
    plt.subplot(2,2,4),plt.imshow(hiddenstates3.reshape(imagedata.shape),'gray')
    plt.show()
    return hiddenstates3

hiddenstates3 = showGibbs()

hiddenstates = np.array([random.randint(0, 1) for i in xrange(imagedata.shape[0]*imagedata.shape[1])], dtype=np.dtype("int8"))        
bet,pro,states= metropolis_optimisation(imagedata,hiddenstates,neighbourmatrix,24)

plt.figure()
plt.subplot(121)
plt.hist(bet,bins=20)
plt.subplot(122)
plt.imshow(states.reshape(imagedata.shape))"""
