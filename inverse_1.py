#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
from fenics import *
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import pylab  
import random

import moola
from dolfin_adjoint import *


# In[2]:


parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,                "eliminate_zeros": True,                "precompute_basis_const": True,                "precompute_ip_const": True}


# In[3]:


mesh = RectangleMesh(Point(0,0), Point(8000000, 4000000), 35,25 )
plot ( mesh, title = 'mesh' )
mesh_points=mesh.coordinates()

mesh_points_x = mesh.coordinates()[:,0].T
mesh_points_y = mesh.coordinates()[:,1].T

nn = np.shape(mesh_points_x)[0]

points = np.zeros((nn,2))
for i in range (nn):
    points[i,:] = (mesh_points_x[i], mesh_points_y[i])


# In[4]:


E = 10**(8) * 0.001
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
    return near(x[0], 8000000) and on_boundary
def top(x, on_boundary):
    return near(x[1], 4000000) and on_boundary
bc1 = DirichletBC(V.sub(1), Constant(0.), left)
bc2 = DirichletBC(V.sub(1), Constant(0.), right)
bc3 = DirichletBC(V.sub(1), Constant(0.), bottom)
bc4 = DirichletBC(V.sub(1), Constant(500000.), top)

bc7 = DirichletBC(V.sub(0).sub(0), Constant(0.), left)
bc8 = DirichletBC(V.sub(0).sub(0), Constant(0.), right)
bc9 = DirichletBC(V.sub(0).sub(0), Constant(0.), bottom)
bc10 = DirichletBC(V.sub(0).sub(1), Constant(0.), left)
bc11 = DirichletBC(V.sub(0).sub(1), Constant(0.), right)
bc12 = DirichletBC(V.sub(0).sub(1), Constant(0.), bottom)

bcs = [bc4,bc12,bc7,bc8, bc9]


# In[6]:


# # random s
# z = np.random.normal(0.0,1,nn)
# z = z.reshape((nn,1))

# Om = FunctionSpace(mesh, "Lagrange", 1)
# s = Function(Om)

# dofs = dof_to_vertex_map(Om)
# var_np = z[dofs]
# s.vector().set_local(var_np)

# m2 = plot(s, title= 's ')
# plt.colorbar(m2)
# plt.show()


# In[8]:


# # functions for prior
V1 = FunctionSpace(mesh, "CG", 1)
W1 = FunctionSpace(mesh, "DG", 0)

u1 = Function(V1, name='State')
v1 = TestFunction(V1)

# tensor
theta1 = 1.0
theta2 = 0.7
ten_alpha = np.pi*(40/90)
unp = np.array([theta1*(np.sin(ten_alpha))**2, (theta1-theta2)*np.sin(ten_alpha)*np.cos(ten_alpha), (theta1-theta2)*np.sin(ten_alpha)*np.cos(ten_alpha), theta2*(np.cos(ten_alpha))**2])
Ten_theta = as_tensor(unp.reshape((2,2)).tolist())

# finding k
gamma = Constant(500000000000)
gamma = interpolate(gamma, W1)

delta = Constant(.005)
delta = interpolate(delta, W1)

# u_pr = Constant(33.0)
# u_pr = interpolate(u_pr, W1)
# g = Constant(0.0)
# g = interpolate(g, W1)

# F = (gamma*inner(Ten_theta*grad(u1-u_pr), grad(v1)) + delta*(u1-u_pr)*v1 - s * v1) * dx - g*v1* ds
# solve(F == 0, u1)


# In[9]:


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
rand ,true= read_data("kappa_true.csv") #HMCP
true = np.array(true)

k_save = np.zeros((np.shape(true)))

for i in range (np.shape(true)[0]):
    k_save[i,0] = float(true[i])*(1)

Om = FunctionSpace(mesh, "Lagrange", 1)
k_true = Function(Om)

dofs = dof_to_vertex_map(Om)
var_np = k_save[dofs]
k_true.vector().set_local(var_np)


# In[10]:


# Defining multiple Neumann boundary conditions 
mf = MeshFunction("size_t", mesh, 1)
mf.set_all(0) # initialize the function to zero
class right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 8000000) and on_boundary
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
        return near(x[1], 4000000) and on_boundary
top = top() # instantiate it
top.mark(mf, 4)
ds = ds(subdomain_data = mf)


# In[11]:


U_ = TestFunction(V)  # P is the concentration or \rho
(u_, P_) = split(U_)
dU = TrialFunction(V)
(du, dP) = split(dU) 
U = Function(V)
(u, P) = split(U)

Un = Function(V)
(un, Pn) = split(Un)


# In[12]:


d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = grad(u)             # Deformation gradient

alpha = 1         
phi = 0.2
rho = 1000.0* 10**(-9)
vis = 0.001* 10**(-3)

kappa = Expression ( "24", degree = 0 )
kappa = project ( kappa, V.sub(1).collapse())

KK = E/(3*(1-2*nu))
G = E/(2*(1+nu))

S = phi * 4.4*10**(-10)* 10**(3)  # n*beta

J = det(I + grad(u))

# Invariants of deformation tensors
Ic = tr((F + F.T)/2) 
F1 = (1-phi)*(-lmbda*Ic * I -mu*(F + F.T)) + phi*alpha*P*I 


gravity = Expression(("0.0", "-9800"),degree = 0)
rho_f = rho*exp(4.4*10**(-10)* 10**(3) * (P))

Flux = (exp(-kappa)*10**(6)*rho/vis)*(grad(P) + (1 - (1-phi)/J)*rho_f*gravity) #  q or flux   

# Define time things.
T, num_steps = 86400*10 , 1          
dt = T / (1.0*num_steps)

f = Expression(("0.0", "-9800"), degree = 0)
f1 = Expression(("0.0", "0.0"), degree = 0)
g = Expression(("0.0"), degree = 0)

mech_form = inner(F1, grad(u_))*dx - inner(f1,u_)*ds(4) + (1 - (1-phi)/J)*rho_f*inner(f,u_)*dx  
phi_form = (1/rho)*inner(Flux, grad(P_))*dx  + alpha*( Ic - tr((grad(un) + (grad(un)).T)/2) )/dt * P_ * dx + S*( P - Pn )/dt * P_ * dx    
F = mech_form + phi_form  
J = derivative(F, U, dU) 


# In[13]:


problem = NonlinearVariationalProblem(F, U, bcs, J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-8
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['relaxation_parameter'] = 1.0


# In[14]:


t = 0
for n in range(num_steps):
    t += dt
    solver.solve()      
    (u1, P1) = U.split() 
    assign(Un,U)


# In[18]:


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
rand ,true= read_data("obs_mm.csv")
true = np.array(true)

true_param2 = np.zeros((np.shape(true)))

for i in range (np.shape(true)[0]):
    true_param2[i,0] = float(true[i])
    
Om = FunctionSpace(mesh, "Lagrange", 1)
var = Function(Om)

dofs = dof_to_vertex_map(Om)
var_np = true_param2[dofs]
var.vector().set_local(var_np)


# In[19]:


# noise
true_param = np.zeros((np.shape(true)))

sigma = 1 * 10**(6)*10**(-3)
true_param = (np.random.normal(0.0,sigma,nn))

Om = FunctionSpace(mesh, "Lagrange", 1)
var2 = Function(Om)

dofs = dof_to_vertex_map(Om)
var_np = true_param[dofs]
var2.vector().set_local(var_np) 


# In[20]:


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


# In[21]:


k_pr = Expression ( " 33 ", degree = 1)     # Permeability of soil   
k_pr = project ( k_pr, V.sub(1).collapse())


# In[22]:


JJ = assemble( ((0.5*(1/sigma**2)*n2*(P-var-var2)**2 *dx + 0.5 * ( (gamma)*inner(ten_alpha*grad(kappa-k_pr), grad(kappa-k_pr)) + (kappa-k_pr)*delta*(kappa-k_pr))* dx )))
control = Control(kappa)


# In[23]:


Gradient = compute_gradient(JJ, control)
R1 = plot(Gradient, title= 'Gradient ')
plt.colorbar(R1)
plt.show()

V1 = FunctionSpace(mesh, "CG", 2)
m_dot = interpolate ( Constant ( 1 ) , V1 )
H = compute_hessian (JJ , control, m_dot, options=None, tape=None )
R2 = plot(H, title= 'Hessian')
plt.colorbar(R2)
plt.show()


# In[24]:


rf = ReducedFunctional(JJ, control)


# In[25]:


import time as time
t1 = time.time()
problem = MoolaOptimizationProblem(rf)
f_moola = moola.DolfinPrimalVector(kappa)
solver = moola.BFGS(problem, f_moola, options={'jtol': 0.0,
                                               'gtol': 1e-9,
                                               'Hinit': "default",
                                               'maxiter': 100,
                                               'mem_lim': 10})
sol = solver.solve()
f_opt = sol['control'].data
t2 = time.time()


# In[ ]:




