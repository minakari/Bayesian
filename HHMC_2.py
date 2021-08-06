#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fenics import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from dolfin_adjoint import *


# In[2]:


parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,                "eliminate_zeros": True,                "precompute_basis_const": True,                "precompute_ip_const": True}


# In[3]:


mesh = RectangleMesh(Point(0,0), Point(8, 4), 35,25 )
mesh_points=mesh.coordinates()
mesh_points_x = mesh.coordinates()[:,0].T
mesh_points_y = mesh.coordinates()[:,1].T

nn = np.shape(mesh_points_x)[0]

points = np.zeros((nn,2))
for i in range (nn):
    points[i,:] = (mesh_points_x[i], mesh_points_y[i])


# In[4]:


E = 10**(8) * 1000
nu = 0.25
lmbda = Constant(E*nu/((1+nu)*(1-2*nu)))
mu = Constant(E/2/(1+nu))


# In[5]:


d = 1 # interpolation degree
Vue = VectorElement('CG', mesh.ufl_cell(), d) # displacement finite element
Vpe = FiniteElement('CG', mesh.ufl_cell(), d) # concentration finite element
V = FunctionSpace(mesh, MixedElement([Vue, Vpe]))

# Boundary conditions
def bottom(x, on_boundary):
    return near(x[1], 0) and on_boundary
def left(x, on_boundary):
    return near(x[0], 0) and on_boundary
def right(x, on_boundary):
    return near(x[0], 8) and on_boundary
def top(x, on_boundary):
    return near(x[1], 4) and on_boundary
bc1 = DirichletBC(V.sub(1), Constant(0.), left)
bc2 = DirichletBC(V.sub(1), Constant(0.), right)
bc3 = DirichletBC(V.sub(1), Constant(0.), bottom)
bc4 = DirichletBC(V.sub(1), Constant(500000000000.), top)

bc7 = DirichletBC(V.sub(0).sub(0), Constant(0.), left)
bc8 = DirichletBC(V.sub(0).sub(0), Constant(0.), right)
bc9 = DirichletBC(V.sub(0).sub(0), Constant(0.), bottom)
bc10 = DirichletBC(V.sub(0).sub(1), Constant(0.), left)
bc11 = DirichletBC(V.sub(0).sub(1), Constant(0.), right)
bc12 = DirichletBC(V.sub(0).sub(1), Constant(0.), bottom)

bcs = [bc4,bc12,bc7,bc8, bc9]


# In[6]:


# Reading k
def read_data(file_name):

    X = []
    y = []
    with open(file_name, "r") as f:
        for line in f:
            num_list = list(map(str, line[:-1].split(",")))
            X.append(num_list[:2])
            y.append([num_list[-1]])

    return np.array(X), np.array(y)
rand ,true= read_data("kappa_noise1_obs60.csv") #HMCP
true = np.array(true)

k_save = np.zeros((np.shape(true)))

for i in range (np.shape(true)[0]):
    k_save[i,0] = float(true[i])

Om = FunctionSpace(mesh, "Lagrange", 1)
k_true = Function(Om)

dofs = dof_to_vertex_map(Om)
var_np = k_save[dofs]
k_true.vector().set_local(var_np)

kappa = project ( k_true, V.sub(1).collapse())


# In[7]:


# Defining multiple Neumann boundary conditions 
mf = MeshFunction("size_t", mesh, 1)
mf.set_all(0) # initialize the function to zero
class right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 8) and on_boundary
right = right() # instantiate it
right.mark(mf, 1)
ds = ds(subdomain_data = mf)

class left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary
left = left() # instantiate it
left.mark(mf, 2)
ds = ds(subdomain_data = mf)

class bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0) and on_boundary
bottom = bottom() # instantiate it
bottom.mark(mf, 3)
ds = ds(subdomain_data = mf)

class top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 4) and on_boundary
top = top() # instantiate it
top.mark(mf, 4)
ds = ds(subdomain_data = mf)


# In[8]:


U_ = TestFunction(V)  # P is the concentration or \rho
(u_, P_) = split(U_)
dU = TrialFunction(V)
(du, dP) = split(dU) 
U = Function(V)
(u, P) = split(U)


# In[9]:


# Reading last q
def read_data(file_name):

    X = []
    y = []
    with open(file_name, "r") as f:
        for line in f:
            num_list = list(map(str, line[:-1].split(",")))
            X.append(num_list[:2])
            y.append([num_list[-1]])

    return np.array(X), np.array(y)
rand ,true= read_data("q_HHMC_1_60_num28.CSV") #HMCP 
true = np.array(true)

q_last = np.zeros((np.shape(true)))

for i in range (np.shape(true)[0]):
    q_last[i,0] = float(true[i])
    
Om = FunctionSpace(mesh, "Lagrange", 1)
q = Function(Om)

dofs = dof_to_vertex_map(Om)
var_np = q_last[dofs]
q.vector().set_local(var_np)


# In[10]:


#########################################
# Reading u, observations
#########################################

def read_data(file_name):

    X = []
    y = []
    with open(file_name, "r") as f:
        for line in f:
            num_list = list(map(str, line[:-1].split(",")))
            X.append(num_list[:2])
            y.append([num_list[-1]])

    return np.array(X), np.array(y)
rand ,true= read_data("obs.csv")
true = np.array(true)

true_param2 = np.zeros((np.shape(true)))

for i in range (np.shape(true)[0]):
    true_param2[i,0] = float(true[i])
Om = FunctionSpace(mesh, "Lagrange", 1)
var = Function(Om)

dofs = dof_to_vertex_map(Om)
var_np = true_param2[dofs]
var.vector().set_local(var_np)


# In[11]:


# noise
def read_data(file_name):

    X = []
    y = []
    with open(file_name, "r") as f:
        for line in f:
            num_list = list(map(str, line[:-1].split(",")))
            X.append(num_list[:2])
            y.append([num_list[-1]])

    return np.array(X), np.array(y)
rand ,true= read_data("noise_1mpa.csv")
true = np.array(true)

noise = np.zeros((np.shape(true)))

for i in range (np.shape(true)[0]):
    noise[i,0] = float(true[i])
Om = FunctionSpace(mesh, "Lagrange", 1)
var2 = Function(Om)

dofs = dof_to_vertex_map(Om)
var2_np = noise[dofs]
var2.vector().set_local(var2_np)


# In[14]:


# location_obs
def read_data(file_name):

    X = []
    y = []
    with open(file_name, "r") as f:
        for line in f:
            num_list = list(map(str, line[:-1].split(",")))
            X.append(num_list[:2])
            y.append([num_list[-1]])

    return np.array(X), np.array(y)
rand ,true= read_data("loc_obsnum_60.csv")
true = np.array(true)

location = np.zeros((np.shape(true)))

for i in range (np.shape(true)[0]):
    location[i,0] = float(true[i])
Om = FunctionSpace(mesh, "Lagrange", 1)
n2 = Function(Om)

dofs = dof_to_vertex_map(Om)
n2_np = location[dofs]
n2.vector().set_local(n2_np)


# In[15]:


import csv
reader = csv.reader(open("H_noise1_obs60.CSV"), delimiter=",")
x = list(reader)
Hessian = np.array(x).astype("float")
invH = np.linalg.inv(Hessian)
L = np.linalg.cholesky(Hessian)


# In[16]:


def kinetic (p, Hessian):
    return .5 * np.transpose(p) @ Hessian @ p 


# In[17]:


# prior k

k_pr = Constant(33)     # Permeability of soil   
k_pr = project ( k_pr, V.sub(1).collapse())

k_pr1 = np.zeros((np.shape(mesh_points_x)[0],1))
for i in range (np.shape(mesh_points_x)[0]):
    k_pr1[i] = k_pr(points[i,:])


# In[18]:


def J_dJdm (points,nn,true_param,var,U_,dU,U,u_,P_,u,P,k_pr1,var2,n2):
#########################################
# Reading k from last file
#########################################

    U_ = TestFunction(V)  # P is the concentration or \rho
    (u_, P_) = split(U_)
    dU = TrialFunction(V)
    (du, dP) = split(dU) 
    U = Function(V)
    (u, P) = split(U)

    Om = FunctionSpace(mesh, "Lagrange", 1)
    kappa = Function(Om)

    dofs = dof_to_vertex_map(Om)
    var_np = true_param[dofs]
    kappa.vector().set_local(var_np)

#########################################
# Runing the forward poisson model
#########################################

    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor

    F = grad(u)             # Deformation gradient

    alpha = 1         
    phi = 0.2

    rho = 1000.0* 10**(9)
    vis = 0.001* 10**(3)

    KK = E/(3*(1-2*nu))
    G = E/(2*(1+nu))

    S = phi * 4.4*10**(-10)* 10**(-3)  # n*beta

    J1 = det(I + grad(u))

    # Invariants of deformation tensors
    Ic = tr((F + F.T)/2) 

    F1 = (1-phi)*(-lmbda*Ic * I -mu*(F + F.T)) + phi*alpha*P*I 


    gravity = Expression(("0.0", "-0.0098"),degree = 0)
    rho_f = rho*exp(4.4*10**(-10)* 10**(-3) * (P))

    Flux = (exp(-kappa)*10**(-6) *rho/vis)*(grad(P) + (1 - (1-phi)/J1)*rho_f*gravity) #  q or flux   

    # Define time things.
    T, num_steps = 86400*10 , 1                    
    dt1 = T / (1.0*num_steps)

    f = Expression(("0.0", "-0.0098"), degree = 0) 
    f1 = Expression(("0.0", "0.0"), degree = 0)

    mech_form = inner(F1, grad(u_))*dx - inner(f1,u_)*ds(4) + (1 - (1-phi)/J1)*rho_f*inner(f,u_)*dx  
    phi_form = (1/rho)*inner(Flux, grad(P_))*dx  + alpha*( Ic  )/dt1 * P_ * dx + S*( P  )/dt1 * P_ * dx  
    F = mech_form + phi_form  
    J2 = derivative(F, U, dU) 


    problem = NonlinearVariationalProblem(F, U, bcs, J2)
    solver  = NonlinearVariationalSolver(problem)

    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1E-8
    prm['newton_solver']['relative_tolerance'] = 1E-5
    prm['newton_solver']['maximum_iterations'] = 25
    prm['newton_solver']['relaxation_parameter'] = 1.0
    
    t = 0
    for nr in range(num_steps):
        t += dt1
        solver.solve()   
        (u1, P1) = U.split() 
    
#########################################
# Calculating objective function, J
#########################################
    # tensor
    theta1 = 1.0
    theta2 = 0.7
    ten_alpha = np.pi*(40/90)
    unp = np.array([theta1*(np.sin(ten_alpha))**2, (theta1-theta2)*np.sin(ten_alpha)*np.cos(ten_alpha), (theta1-theta2)*np.sin(ten_alpha)*np.cos(ten_alpha), theta2*(np.cos(ten_alpha))**2])
    Ten_theta = as_tensor(unp.reshape((2,2)).tolist())

    sigma = 1.0 * 10**(6) *10**(3)
    JJ = assemble( 1*(0.5*(1/sigma**2)*n2*(P-var-var2)**2 *dx + 0.5 * ( (0.5)*inner(ten_alpha*grad(kappa-k_pr), grad(kappa-k_pr)) + (kappa-k_pr)*0.005*(kappa-k_pr)) * dx))
    control = Control(kappa)

#########################################
# Calculating Gradient
#########################################

    Gradient = compute_gradient(JJ, control)

#########################################
# Save partial_p
#########################################

    Partial_p = np.zeros((nn,1))
    for i in range (nn):
        Partial_p[i] = Gradient(points[i,:])

    return JJ, Partial_p


# In[19]:


M = Hessian
invM = np.linalg.inv(M)
L = np.linalg.cholesky(M)


# In[21]:


# Hamiltonian Monte Carlo

dim = nn
n = 200  # number of generated sample
q = k_save  #  np.zeros((nn,1)) + 0.5 #
proposals = 0
accepted = 0

observations = np.zeros( (dim,n) )
potential = np.zeros( (n,1) )
iterations = np.zeros( (n,1) )
rf = 1
dt = 0.4

for i in range(n):
    accept = False
    tally = 0
    # random p
    p = (np.random.normal(0.0,1.0,nn)) # st.norm(0, 1).rvs(size=2) 
    p = p.reshape((nn,1)) 
    
    J, partial_p = J_dJdm (points,nn,q,var,U_,dU,U,u_,P_,u,P,k_pr1,var2,n2)
    J = J * 1
    
    print('J=',J)
    # update q 
    p1 = L@p
    p_new = p1 - (dt/2 * partial_p)
    q_new = q +  (dt * invM@p_new)
    
    J_new, partial_p_new = J_dJdm (points,nn,q_new,var,U_,dU,U,u_,P_,u,P,k_pr1,var2,n2)
    
    p_new = p_new - dt/2 * partial_p_new
    a = min(1,np.exp((J + kinetic(p1, invM) - J_new - kinetic(p_new, invM))*1))    
    pot = J
    
    u11 = np.random.uniform()    
    if a>u11:
        proposals += 1
        accept = True
        q = q_new
        p = p_new
        pot = J_new
        accepted +=1
        
    tally+=1
    if tally%150==0:
        print("not mixing well")
    
    potential[i,0] = pot
    observations[:,i] = q.reshape((nn))
    iterations[i,0] = i+1

