#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import cv2 as cv
import math
#import matplotlib.pyplot as plt
#import scipy.stats as stats
#import statistics
import time

from numpy.linalg import norm, inv
from scipy.stats import multivariate_normal as mv_norm
from sklearn.cluster import KMeans

video = cv.VideoCapture('umcp.mpg')


number_of_frames_to_start = 100
number_of_cluster = 4
epochs = 10
#t1 = time.time()
data_for_initilization = read_starting_frame_rgb(number_of_frames_to_start,video)
tr_mean, tr_cov, tr_pi = train_with_rgb_values1(data_for_initilization,number_of_cluster)


# In[7]:


'''
this function takes video and return a matrix of dimension (rgb_values * number_of_pixel_in_one_frame * number_of_frames)
'''
def read_starting_frame_rgb(number,video):
    _, first_frame = video.read()
    first_frame = first_frame.reshape(-1,first_frame.shape[2])

    a = first_frame.shape[0]          #no. of pixels
    b = first_frame.shape[1]          #rgb components
    c = number                        #number of frames to train
 
    pixels_data = [[ [0 for col in range(b)] for col in range(a)] for row in range(number)]
 
    for i in range(number):
        ret, frame = video.read()        
        if frame is None:
            break            
        frame = frame.reshape(-1,frame.shape[2])        
        pixels_data[i] = frame
    return np.array(pixels_data)


# In[8]:


'''
use gaussian to train on starting values and return (mean, cov, pi) 
'''
def train_with_rgb_values1(data_for_initilization,number_of_cluster):
    data_for_initilization = cv.cvtColor(data_for_initilization, cv.COLOR_BGR2GRAY)

    a = len(data_for_initilization)# no of frames
    b = len(data_for_initilization[0]) #no of pxiels
 
    for i in range(b):
        print("pixel no",i)
        temp = data_for_initilization[:,i]
        temp = temp.reshape(-1, 1)
        
        #replace this with our own code

        gm_model = Gaussian_Mixture(data_for_initilization, number_of_cluster)
        means = gm_model.means_
        covs = gm_model.covariances_
        covs = covs.reshape(-1,1)
        pies = gm_model.weights_
        pies = pies.reshape(-1,1)
        #'''
        
        if i == 0:
            mean = means
            cov = covs
            pi = pies            
        else:
            mean = np.append(mean, means, axis=1)
            cov = np.append(cov, covs, axis=1)
            pi = np.append(pi, pies, axis=1)
    return mean, cov, pi


# In[9]:


'''
Adaptive Algorithm implementation
'''
def match3(video, tr_mean, tr_cov, tr_pi,number_of_cluster):
    import math
    from scipy.stats import multivariate_normal as mv_norm    

    
    count = 0
    ret = True
    while(ret == True):
        ret, frame = video.read()
        count = count+1
        print("Frame", count)
        alp = 0.008
        #if count >24:
        
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame = gray_frame.reshape(-1,1)
        new_frame = np.zeros((84480,1))


        #Calculating maximum maching gussian
           
        for i in  range (gray_frame.shape[0]):
            guss = []
            rep = []
            for j in range(number_of_cluster):
                guss.append(abs(gray_frame[i] - tr_mean[j][i])/ math.sqrt(tr_cov[j][i]))
                rep.append((tr_pi[j][i])/math.sqrt(tr_cov[j][i]))
    
            min_ind = guss.index(min(guss))
            rep_ind = rep.index(min(rep))
            
            
            if (guss[min_ind] <= 2.5):
                
                
                d = (2*math.pi*tr_cov[min_ind][i])**0.5
                n = math.exp(-(gray_frame[i] - tr_mean[min_ind][i])**2/(2*tr_cov[min_ind][i]))
                rho = n/d
                rho = rho*alp
                if rho > 1:
                    print("###################")
                   
                
                #Mean & Covariance update
                
                    
                tr_mean[min_ind][i] = abs((1-rho)*tr_mean[min_ind][i] + rho*gray_frame[i])
                tr_cov[min_ind][i] = abs((1-rho)*tr_cov[min_ind][i] + rho*(gray_frame[i] - tr_mean[min_ind][i])*(gray_frame[i] - tr_mean[min_ind][i]))    
                                                                                                                                       
                #Pies update
                pi_sum = 0
                for l in range (number_of_cluster):
                    if l == min_ind:
                        tr_pi[l][i] = abs((1-alp)*tr_pi[l][i] + alp)
                        pi_sum = tr_pi[l][i] + pi_sum
                    else: 
                        tr_pi[l][i] = abs((1-alp)*tr_pi[l][i])
                        pi_sum = tr_pi[l][i] + pi_sum
                        
                #normalization
                if pi_sum > 1:
                    for k in range(number_of_cluster):
                        tr_pi[k][i] = tr_pi[k][i]/pi_sum
                
                
                #check background and foregroud
                ratio = []
                for j in range(number_of_cluster):
                    ratio.append((tr_pi[j][i])/math.sqrt(tr_cov[j][i]))
               
                des_ord = np.argsort(ratio)[::-1]
                
                threshold = 0.75
                # 255 - background
                # 0 - foreground       
                total_sum = 0
                B = 0

                for j in des_ord:
                    B = B + 1
                    total_sum = total_sum + tr_pi[j][i]
                    if total_sum > threshold:
                        break
                new_pix = 255
                
                for k in range(B):
                    if abs(gray_frame[i] - tr_mean[des_ord[k]][i]) < 2.5*math.sqrt(tr_cov[des_ord[k]][i]):
                        new_pix = 0
                        break
                        
                         
            else: 
                new_pix = 255
                tr_mean[rep_ind][i] = gray_frame[i]
                tr_cov[rep_ind][i] = 400
                tr_pi[rep_ind][i] = tr_pi[rep_ind][i]/6
                
                for l in range(number_of_cluster):
                    pi_sum = pi_sum + tr_pi[l][i]
                
                #normalization
                if pi_sum > 1:
                    for k in range(number_of_cluster):
                        tr_pi[k][i] = tr_pi[k][i]/pi_sum
        
            new_frame[i] = new_pix
            
        new_frame = new_frame.reshape(1,84480,1)
            
        if count == 1:
            comb_frame = new_frame
        else:
            comb_frame = np.append (comb_frame, new_frame, axis = 0)
            
        if count == 100:
            break          
    print ("comb_frame", comb_frame.shape)
    return comb_frame


# In[10]:


'''
this is for foreground
'''
def display_frame(process_data):
    num = len(process_data)
    component = len(process_data[0][0])
    #fourcc = cv.VideoWriter_fourcc(*'XVID')
    #out = cv.VideoWriter("Gaussian4.avi", fourcc, 24.0 , (352,240),0)
    for i in range(num):

        first_frame = process_data[i].reshape(240, 352, component)
        #first_frame = np.uint8(first_frame)
        #out.write(first_frame)
        cv.imshow("frame", first_frame)

        if cv.waitKey(20) == ord('q'):
            break

    cv.destroyAllWindows()


# In[51]:


video = cv.VideoCapture('umcp.mpg')
for_grd = match3(video, tr_mean, tr_cov, tr_pi,number_of_cluster)
print(for_grd.shape)


# In[4]:


#Running by putting formula inside
display_frame(for_grd)


# In[ ]:


'''
this is for background
'''
def display_frame3():
    video = cv.VideoCapture("umcp.mpg")
    video1 = cv.VideoCapture("Gaussian4.avi")
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter("BG1.avi", fourcc, 24.0 , (352,240))
    for k in range(950):
        _, frame = video.read()
        _, frame1 = video1.read()
        frame = frame.reshape(-1,3)
        frame1 = frame1.reshape(-1,3)
        for i in range(frame.shape[0]):
                if frame1[i].all() == 0:
                    frame[i] = frame[i]
                else:
                    frame[i] = frame[i - 20]
        frame = frame.reshape(240,352,3)

        frame = np.uint8(frame)
        out.write(frame)
        cv.imshow("frame", frame)
        

        if cv.waitKey(20) == ord('q'):
            break

    cv.destroyAllWindows()
display_frame3()


# In[55]:


def initailizeCluster(X, n_cluster):
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    mu_k = kmeans.cluster_centers_
    clusters = []
    for i in range(n_cluster):
        clusters.append({'mu' : mu_k[i],
                         'cov': 255,
                         'pi': tuple(kmeans.labels_).count(i)/X.shape[0]})
    return clusters


# In[7]:


def ExpectationStep(X, clusters):
    cluster_prob =[]
    for cluster in clusters:
        prob =[]
        for i in range(X.shape[0]):
            cluster['cov'] = cluster['cov']*np.eye(3)
            prob.append(cluster['pi'] * mv_norm.pdf(X[i],cluster['mu'], cluster['cov'],allow_singular=True))#gaussian(X[i],cluster['mu'], cluster['cov']))
        cluster_prob.append(prob)   
    sum_prob = np.sum(cluster_prob, axis=0)

    return cluster_prob/sum_prob


# In[8]:


def MaximizationStep(X,clusters, epochs):    
    for _ in range(epochs):
        gama = ExpectationStep(X, clusters)       
        for i,cluster in enumerate(clusters):
            
            cluster['pi'] = np.sum(gama[i]) / gama.shape[1]
            cov_k = np.zeros((X.shape[1], X.shape[1]))

            for j in range(X.shape[0]):
                diff = (X[j] - cluster['mu']).reshape(-1, 1)
                cov_k += gama[i][j] * np.dot(diff, diff.T)
            cluster['cov'] = cov_k/ np.sum(gama[i])
            cluster['mu'] = np.matmul(gama[i],X) / np.sum(gama[i])        
    return clusters


# In[9]:


def Gaussian_Mixture(X,n_cluster,epochs):
    clusters = initailizeCluster(X,n_cluster)
    return MaximizationStep(X,clusters,epochs)

