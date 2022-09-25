# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:58:48 2022

@author: halvorak
"""


import numpy as np
# import scipy.stats
import matplotlib.pyplot as plt
# import casadi as cd
# import matplotlib

# Did some modification to these packages
# from myFilter import UKF
# from myFilter import UKF_constrained
from myFilter import kalmanFilter
# from myFilter import UKF
# from myFilter import UKF2
# from myFilter import UKF3
# from myFilter import unscented_transform as ut

# #Self-written modules
# import sigma_points_classes as spc
# # import unscented_transformation as ut
# # import utils_bioreactor_tuveri as utils_br

#%%Define functions

def model_resistor(dt):
    R = 3
    L = 1
    C = .5
    
    #continous time
    F = np.array([[0., 1/C],
                  [-1/L, -R/L]])
    G = np.array([[0],
                  [1/L]])
    
    #discretized
    F_d = np.eye(F.shape[0]) - F*dt
    G_d = G*dt
    
    H = np.array([[0, 1]])
    return F_d, G_d, H

def noise_statistics():
    Q = 1.
    R = 1.
    return Q, R

def model_1(dt):
    F = np.array([[1., dt],
                  [0., 1]])
    G = np.array([[0.],
                  [0.]])
    H = np.array([[1, 0]])
    return F, G, H
dt = 1.
F, G, H = model_1(dt)

dim_x = F.shape[0]
dim_y = H.shape[0]

# Q = np.eye(dim_x)
Q = np.array([[1., 1.],
              [1., 1.]])
R = np.eye(dim_y)

x0 = np.array([[0.],
               [10.]])
P0 = np.eye(dim_x)*10

kf = kalmanFilter.KalmanFilter(F, G, H, Q, R,name="orig")
kf.x_post = x0 + np.random.multivariate_normal(np.zeros(dim_x), P0).reshape(-1,1)

#%%Simulate the system
N =  5
t = np.arange(0, N, dt)
dim_t = t.shape[0]
x_true = np.zeros((dim_x, dim_t))
x_prior = np.zeros((dim_x, dim_t))
x_post = np.zeros((dim_x, dim_t))
y = np.zeros((dim_y, dim_t))

x_true[:, 0] = x0.flatten()
for i in range(1, dim_t):
    x_true_next = F @ x_true[:, i-1].reshape(-1,1) + np.random.multivariate_normal(np.zeros(dim_x), Q).reshape(-1,1)
    
    yk = H @ x_true_next + np.random.multivariate_normal(np.zeros(dim_y), R).reshape(-1,1)
    y[:, i] = yk
    
    x_true[:, i] = x_true_next.flatten()
    
    #Kalman filter
    kf.predict(np.zeros((G.shape[1], 1)))
    kf.update(yk)
    x_prior[:, i] = kf.x_prior.flatten()
    x_post[:, i] = kf.x_post.flatten()

    
#%%Plotting

fig_x, ax_x = plt.subplots(2, 1)
for j in range(dim_x):
    ax_x[j].plot(t, x_true[j,:], label = "true")
    ax_x[j].plot(t, x_prior[j,:], label = "prior")
    ax_x[j].plot(t, x_post[j,:], label = "post")
    ax_x[j].set_ylabel(f"x{j+1}")
    
ax_x[j].set_xlabel("t")

ax_x[0].scatter(t, y, label = "y")
ax_x[0].legend()

