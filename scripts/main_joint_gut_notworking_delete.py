# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:55:44 2022

@author: halvorak
"""

import numpy as np
import scipy.stats
import scipy.io
# import scipy.integrate
import casadi as cd
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import os
import random
import sklearn.preprocessing
# import scipy.linalg
# import matplotlib.patches as plt_patches
# import matplotlib.transforms as plt_transforms
import copy
import time
import pandas as pd
import seaborn as sns
# import matlab.engine

# Did some modification to these packages
# from myFilter import UKF
# from myFilter import UKF_constrained
from myFilter import sigma_points as ukf_sp
from myFilter import UKF
from myFilter import unscented_transform as ut_ukf
import sigma_points_classes as spc

#Self-written modules
# import sigma_points_classes as spc
import unscented_transformation as ut
import utils_bioreactor_tuveri as utils_br
font = {'size': 14}
matplotlib.rc('font', **font)

#%% Import parameters
par_samples_fx, par_samples_hx, par_names_fx, par_names_hx, par_det_fx, par_det_hx, Q_nom, R_nom, plt_output, par_dist_fx = utils_br.get_literature_values(N_samples = int(5e3), plot_par = False)

par_cov_fx = np.cov(par_samples_fx)
par_cov_uncorrelated_fx = np.diag(np.diag(par_cov_fx).copy())

df_par = pd.DataFrame(data = par_samples_fx.T, 
                      columns = par_names_fx)

mode = {}
for key, dist in par_dist_fx.items():
    mode[key] = scipy.optimize.minimize(lambda theta: -dist.pdf(theta),
                                   dist.mean(), #use mean as theta0
                                   tol = 1e-10).x[0]

#%% fx parameters
par_true_fx = par_det_fx.copy()
par_kf_fx = par_det_fx.copy()

par_mean_fx = np.mean(par_samples_fx, axis = 1)
par_from_mean_fx = utils_br.select_points_from_mean_multivar_mahalanobis(
    par_samples_fx.T,
    .95, 1.05) #multivariate way of selecting point mean + 1std (or .95, 1.05 from mean)

for i in range(len(par_names_fx)):
    if par_names_fx[i] == "S_in":
        # par_true_fx[par_names_fx[i]] = par_mean_fx[i] + np.sqrt(par_cov_fx[i, i])
        par_true_fx[par_names_fx[i]] = mode["S_in"]
    else:
        par_true_fx[par_names_fx[i]] = par_mean_fx[i]
    # par_true_fx[par_names_fx[i]] = par_from_mean[i]
    par_kf_fx[par_names_fx[i]] = par_mean_fx[i]
    

#%% hx parameters
par_true_hx = par_det_hx.copy()
par_kf_hx = par_det_hx.copy()
for i in range(len(par_names_hx)):
    # par_true_hx[par_names_hx[i]] = par_samples_hx.mean()+par_samples_hx.std() #true system uses the mean - std_dev
    par_true_hx[par_names_hx[i]] = par_samples_hx.mean() #true system uses the mean - std_dev
    par_kf_hx[par_names_hx[i]] = par_samples_hx.mean() #the ukf uses mean values reported in the literaure

#%% Define dimensions and initialize arrays
        
x0 = utils_br.get_x0_literature()
u0 = utils_br.get_u0_literature()

dim_x = x0.shape[0]
dim_par_fx = len(par_true_fx)
dim_par_hx = len(par_true_hx)
dim_xa = dim_par_fx + dim_x
dim_u = u0.shape[0]
dt_y = 1/60 # [h] Measurement frequency

# t_end = 48 # [h]
# t_end = 14 # [h]
t_end = 15 # [h]
t = np.linspace(0, t_end, int(t_end/dt_y))
dim_t = t.shape[0]

y0 = utils_br.hx(x0, par_true_hx)
dim_y = y0.shape[0]
y = np.zeros((dim_y, dim_t))
y[:, 0] = y0*np.nan

x_true = np.zeros((dim_x, dim_t)) 
par_history_fx = np.zeros((dim_par_fx, dim_t))
par_history_fx[:, 0] = list(par_true_fx.values())

#Make control law
u = np.tile(u0.reshape(-1,1), dim_t)
t_low_sugar_ol = t[-1] #for control law 2, see end of the for loop
# idx_u_increase = np.searchsorted(t, 21, side = "right") #after 21h, we have inflow
# u[0, idx_u_increase] = (.5* #L/min
#                         60) #min/h ==> L/h

#UKF initial states
P0 = utils_br.get_P0_literature()
Pa0 = scipy.linalg.block_diag(par_cov_fx, P0)
x0_kf = utils_br.get_x0_kf_literature(sigma_multiplier=1.) # x0+ the standard deviation*multiplier
xa0 = np.hstack((par_mean_fx, x0_kf))

#Arrays where values are stored
x_true = np.zeros((dim_x, dim_t)) #[[] for _ in range(dim_t-1)] #make a list of list
x_ol = np.zeros((dim_x, dim_t)) #Open loop simulation - same starting point and param as UKF

x_post_jgut = np.zeros((dim_xa, dim_t))


P_diag_post_jgut = np.zeros((dim_xa, dim_t))


x_true[:, 0] = x0

x_post_jgut[:, 0] = xa0
x0_ol = x0_kf.copy()
x_ol[:, 0] = x0_ol

P_diag_post_jgut[:, 0] = np.diag(Pa0)

t_span = (t[0],t[1])

solver_tol_default_mode = 1e-5 #for maximizing the pdf \approx kde(w_stoch).

# #%% Define UKF with adaptive Q, R from UT
# # alpha = 1
# # beta = 0
# # kappa = 3-dim_x
# alpha = 1e-3
# beta = 2.
# kappa = 0.#3-dim_x
# points_gut = ukf_sp.MerweScaledSigmaPoints(dim_xa,
#                                         alpha,
#                                         beta,
#                                         kappa)

# fx_ukf_gut = None #updated later in the simulation
# hx_ukf_gut = None #updated later in the simulation

# kfc_gut = UKF.UnscentedKalmanFilter(dim_x = dim_xa, 
#                                                         dim_z = dim_y, 
#                                                         dt = 100, 
#                                                         hx = hx_ukf_gut, 
#                                                         fx = fx_ukf_gut,
#                                                         points = points_gut
#                                                         )
# kfc_gut.x = x_post_jgut[:, 0]
# kfc_gut.P = P0.copy()
# kfc_gut.Q = None #to be updated in a loop
# kfc_gut.R = R_nom 

# #Juliers sigma points
# points_ut = ukf_sp.JulierSigmaPoints(dim_xa)
# sigmas_julier = points_ut.sigma_points(xa0, Pa0)
# points_ut._compute_weights()
# w_julier = points_ut.Wm

# #genUt sigma points
# points_gut = spc.GenUTSigmaPoints(dim_xa)
# samples_x = scipy.stats.multivariate_normal(mean = x0, cov = P0).rvs(size = par_samples_fx.shape[1])
# samples_x = scipy.stats.multivariate_normal(mean = x0, cov = P0).rvs(size = int(5e3))

# #3rd,4th moment
# # cm3_par = scipy.stats.moment(par_samples_fx, moment=3, axis = 1)
# cm3_par = np.zeros((dim_par_fx,)) #always zero for normal distribution
# cm4_par = scipy.stats.moment(par_samples_fx, moment=4, axis = 1)
# cm3 = scipy.stats.moment(samples_x, moment=3)
# cm4 = scipy.stats.moment(samples_x, moment=4)
# cm3a = np.hstack((cm3_par, cm3))
# cm4a = np.hstack((cm4_par, cm4))

# def cm4_isserli_for_multivariate_normal(P):
#     cm4 = P @ (np.trace(P) * np.eye(P.shape[0]) + 2*P) #cm4.shape = P.shape
#     return np.diag(cm4)

# cm4_isserli = cm4_isserli_for_multivariate_normal(P0)

# # samples_xa = np.vstack((par_samples_fx, samples_x.T))
# # cm3a_check = scipy.stats.moment(samples_xa.T, moment=3)
# # cm4a_check = scipy.stats.moment(samples_xa.T, moment=4)

# s, w_gut = points_gut.compute_scaling_and_weights(Pa0, cm3a, cm4a, s1 = None)

# sigmas_gut, P_sqrt_gut = points_gut.compute_sigma_points(xa0, Pa0, s)

# points_jgut = ukf_sp.ParametricUncertaintyGenUTSigmaPoints(dim_par_fx, dim_x, cm3_par, cm4_par)
# sigmas_jgut = points_jgut.sigma_points(xa0, Pa0)
# w_jgut = points_jgut.Wm


# #make a model and see if the genut sigmapoints give the same result
# #First, try the unity function (y=x)
# x_hat_julier, P_hat_julier = ut_ukf.unscented_transform(sigmas_julier, w_julier, w_julier)
# norm_x_julier = np.linalg.norm(x_hat_julier - Pa0, ord = None)
# norm_P_julier = np.linalg.norm(P_hat_julier - Pa0, ord = "fro")
# print(f"norm_x_julier: {norm_x_julier}\n",
#       f"norm_P_julier: {norm_P_julier}")

# x_hat_gut, P_hat_gut = ut.unscented_transformation(sigmas_gut, w_gut)
# norm_x_gut = np.linalg.norm(x_hat_gut - xa0, ord = None)
# norm_P_gut = np.linalg.norm(P_hat_gut - Pa0, ord = "fro")
# print(f"norm_x_gut: {norm_x_gut}\n",
#       f"norm_P_gut: {norm_P_gut}")

# x_hat_jgut, P_hat_jgut = ut_ukf.unscented_transform(sigmas_jgut, w_jgut, w_jgut)
# norm_x_jgut = np.linalg.norm(x_hat_jgut - xa0, ord = None)
# norm_P_jgut = np.linalg.norm(P_hat_jgut - Pa0, ord = "fro")
# print(f"norm_x_jgut: {norm_x_jgut}\n",
#       f"norm_P_jgut: {norm_P_jgut}")


#%%Block Cholesky algorithm

# corr = np.corrcoef(par_cov_fx)
corr = np.corrcoef(par_samples_fx)
corr[-2:, :] = 0
corr[:, -2:] = 0
corr[-2, -2] = 1
corr[-1, -1] = 1

cond_corr = np.linalg.cond(corr)
cond_cov = np.linalg.cond(par_cov_fx)



sigmas = np.diag(np.sqrt(np.diag(par_cov_fx)))
cov2 = sigmas @ corr @ sigmas

norm_cov = np.linalg.norm(cov2 - par_cov_fx)

L_A = scipy.linalg.cholesky(par_cov_fx)
A_inv = scipy.linalg.inv(par_cov_fx)
L_A_inv = scipy.linalg.inv(L_A)
B = Pa0[dim_par_fx:, :dim_par_fx]
D = Pa0[-dim_x:, -dim_x:]
S = D - B @ A_inv @ B.T
L_S = scipy.linalg.cholesky(S)
L_C_upper = np.hstack((L_A, np.zeros((L_A.shape[0], L_S.shape[1]))))
L_C_lower = np.hstack((B @ L_A_inv.T, L_S))

L_C = np.vstack((L_C_upper, L_C_lower))

C = L_C.T @ L_C

norm_par = np.linalg.norm(Pa0 - C)

def block_cholesky(U_A, U_A_inv, A_inv, B, D):
    """
    Cholesky decomposition of full matrix C, by using available cholesky decomposition of it's block matrix A. See https://scicomp.stackexchange.com/questions/5050/cholesky-factorization-of-block-matrices.
    
    Have 
    
    C = [[A, B.T]
         [B, D]]
    Want to find U_C such that U_C.T @ U_C = C (upper Cholesky root) by using available information about A (it's cholesky decomposition and inverses)

    Parameters
    ----------
    U_A : TYPE np.array((dim_a, dim_a))
        DESCRIPTION. Cholesky decomposition of A. Have A = U_A.T @ U_A
    U_A_inv : TYPE np.array((dim_a, dim_a))
        DESCRIPTION. Inverse of U_A
    A_inv : TYPE np.array((dim_a, dim_a))
        DESCRIPTION. Inverse of A
    B : TYPE np.array((dim_d, dim_a))
        DESCRIPTION. Block matrix
    D : TYPE np.array((dim_d, dim_d))
        DESCRIPTION. Block matrix. Need to do Cholesky decomposition of this dimensionality to compute U_C. Note that dim_d < dim_c, so this is more efficienct than computing U_C directly.

    Returns
    -------
    U_C : TYPE np.array((dim_c, dim_c))
        DESCRIPTION. Upper Cholesky decomposition of the matrix C

    """
    S = D - B @ A_inv @ B.T
    U_S = scipy.linalg.cholesky(S, lower = False) #upper is also default choice, so this is redundant
    U_C_upper = np.hstack((U_A, np.zeros((U_A.shape[0], U_S.shape[1]))))
    U_C_lower = np.hstack((B @ U_A_inv.T, U_S))
    U_C = np.vstack((U_C_upper, U_C_lower))
    return U_C

U_C = block_cholesky(L_A, L_A_inv, A_inv, B, D)
C2 = U_C.T@U_C
norm_par2 = np.linalg.norm(Pa0 - C2)


print(f"norm_par: {norm_par}")
print(f"norm_par2: {norm_par2}")



