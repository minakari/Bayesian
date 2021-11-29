#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fenics import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


mesh = RectangleMesh(Point(0,0), Point(8, 4), 35,25 )
mesh_points=mesh.coordinates()

mesh_points_x = mesh.coordinates()[:,0].T
mesh_points_y = mesh.coordinates()[:,1].T

nn = np.shape(mesh_points_x)[0]

points = np.zeros((nn,2))
for i in range (nn):
    points[i,:] = (mesh_points_x[i], mesh_points_y[i])


# In[3]:


import csv
reader = csv.reader(open("H.CSV"), delimiter=",")
x = list(reader)
Hessian = np.array(x).astype("float")
cov = np.linalg.inv(Hessian)

reader = csv.reader(open("kappa.CSV"), delimiter=",")
x = list(reader)
mu = np.array(x).astype("float")


# In[4]:


std  = (np.diag(np.linalg.inv(Hessian)))**0.5  
std = std.reshape((nn,1))


# In[5]:


def p_y (mu, Hessian,nn, y):  
    return 0.5*np.transpose(y-mu) @ Hessian @ (y-mu) 


# In[6]:


def grad (mu, Hessian,nn, y):
    return  Hessian@(y-mu) 


# In[7]:


def kinetic_MCMC (q_new,q, Hessian, invH,g):
    return .5 * np.transpose(q - q_new + invH @ g) @ Hessian @ (q - q_new + invH @ g)


# In[8]:


def kinetic_HMC (p, Hessian):
    return .5 * np.transpose(p) @ Hessian @ p 


# In[11]:


# Hessian - Markov Chain Monte Carlo

dim = nn
n = 10000  # number of generated sample
q = mu 
proposals = 0
accepted = 0

observations = np.zeros( (dim,n) )
iterations = np.zeros( (n,1) )

dt = 1

for i in range(n):
    accept = False
    # random p
    p = (np.random.normal(0.0,1.0,nn)) 
    p = p.reshape((nn,1)) 
    
    J = p_y (mu, Hessian,nn, q)
    g = grad (mu, Hessian,nn, q)
    
####### local Hessian
    H = Hessian
    invH = np.linalg.inv(H)
    
    
    L = np.linalg.cholesky(invH)
    
    q_new = q - invH @ g   + (dt * L @ p)
    J_new = p_y (mu, Hessian,nn, q_new)
    
    g_new = grad (mu, Hessian,nn, q_new)
    
    q_y_k = kinetic_MCMC(q_new, q ,H , invH, g_new);
    q_k_y = kinetic_MCMC(q, q_new ,H , invH, g);
        
    a = min(1,np.exp((J + q_k_y - J_new - q_y_k)))
        
    u1 = np.random.uniform()    
    if a>u1:
        proposals += 1
        accept = True
        q = q_new
        accepted +=1
        
    observations[:,i] = q.reshape((nn))
    iterations[i,0] = i+1
    print('iteration', i)
    print('##############################################')


# In[48]:


# Markov Chain Monte Carlo

dim = nn
n = 10000  # number of generated sample
q = mu 
proposals = 0
accepted = 0

observations = np.zeros( (dim,n) )
iterations = np.zeros( (n,1) )

dt = 0.01

for i in range(n):
    accept = False
    tally = 0
    # random p
    p = (np.random.normal(0.0,1.0,nn)) 
    p = p.reshape((nn,1)) 
    
    J = p_y (mu, Hessian,nn, q)        
    
    q_new = q + (dt * p)

    J_new = p_y (mu, Hessian,nn, q_new)
        
    a = min(1,np.exp((J - J_new )))
        
    u1 = np.random.uniform()    
    if a>u1:
        proposals += 1
        accept = True
        q = q_new
        accepted +=1
    
    observations[:,i] = q.reshape((nn))
    iterations[i,0] = i+1
    print('iteration', i)
    print('##############################################')


# In[51]:


# Hessian - Hamiltonian Monte Carlo

dim = nn
n = 5000  # number of generated sample
q = mu 
proposals = 0
accepted = 0

observations = np.zeros( (dim,n) )
acceptance_rate = np.zeros( (n,1) )
iterations = np.zeros( (n,1) )

dt = 0.3

for i in range(n):
    accept = False
    # random p
    p = (np.random.normal(0.0,1.0,nn)) 
    p = p.reshape((nn,1)) 
    
    J = p_y (mu, Hessian,nn, q)
    g = grad (mu, Hessian,nn, q)
    
    ######### local H
    H = Hessian
    invH = np.linalg.inv(H)
    
    L = np.linalg.cholesky(H)
    
    # update q  
    p1 = L@p
    p_new = p1 - (dt/2 * g)
    q_new = q +  (dt * invH @ p_new)
    
    g_new = grad (mu, Hessian,nn, q_new)
    
    p_new = p_new - dt/2 * g_new
    
    J_new = p_y (mu, Hessian,nn, q_new)
        
    a = min(1,np.exp((J + kinetic_HMC(p1, invH) - J_new - kinetic_HMC(p_new, invH))))
       
    acce = np.exp((J + kinetic_HMC(p1, invH) - J_new - kinetic_HMC(p_new, invH)))
    
    pot = J
    
    u1 = np.random.uniform()    
    if a>u1:
        proposals += 1
        accept = True
        q = q_new
        p = p_new
        pot = J_new
        accepted +=1
        
    acceptance_rate[i,0] = acce
    observations[:,i] = q.reshape((nn))
    iterations[i,0] = i+1
    print('iteration', i)
    print('##############################################')


# In[76]:


# Hamiltonian Monte Carlo

dim = nn
n = 10000  # number of generated sample
q = mu 
proposals = 0
accepted = 0

observations = np.zeros( (dim,n) )
acceptance_rate = np.zeros( (n,1) )
iterations = np.zeros( (n,1) )

dt = 0.1

for i in range(n):
    accept = False
    # random p
    p = (np.random.normal(0.0,1.0,nn)) 
    p = p.reshape((nn,1)) 
    
    J = p_y (mu, Hessian,nn, q)
    g = grad (mu, Hessian,nn, q)
    
    ######### M
    M = np.eye((nn))
    
    # update q   
    p_new = p - (dt/2 * g)
    q_new = q +  (dt * p_new)
    
    g_new = grad (mu, Hessian,nn, q_new)
    
    p_new = p_new - dt/2 * g_new
    
    J_new = p_y (mu, Hessian,nn, q_new)
        
    a = min(1,np.exp((J + kinetic_HMC(p, M) - J_new - kinetic_HMC(p_new, M))))
        
    acce = np.exp((J + kinetic_HMC(p, M) - J_new - kinetic_HMC(p_new, M)))
    
    pot = J
    
    u1 = np.random.uniform()    
    if a>u1:
        proposals += 1
        accept = True
        q = q_new
        p = p_new
        pot = J_new
        accepted +=1
        
    acceptance_rate[i,0] = acce
    observations[:,i] = q.reshape((nn))
    iterations[i,0] = i+1
    print('iteration', i)
    print('##############################################')


# In[ ]:





# In[ ]:




