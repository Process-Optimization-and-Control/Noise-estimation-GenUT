# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:58:48 2022

@author: halvorak
"""


import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
# import matplotlib
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
# import pathlib
import os
import copy
import seaborn as sns

# Did some modification to these packages
# from myFilter import UKF
# from myFilter import UKF_constrained
from myFilter import sigma_points as ukf_sp
from myFilter import UKF
from myFilter import UKF2

#Self-written modules
import sigma_points_classes as spc
import unscented_transformation as ut
import utils_bioreactor_tuveri as utils_br

#%%Define functions

def fx_ode(t, x, p):
    mu_max = p["mu_max"]
    K_S = p["K_S"]
    k_d = p["k_d"]
    Y_XS = p["Y_XS"]
    xdot = np.zeros(x.shape)
    
    #"unscale" the parameters
    scaler_biopar = 1e3
    mu_max_unsc = mu_max/scaler_biopar
    K_S_unsc = K_S/scaler_biopar
    Y_XS_unsc = Y_XS/scaler_biopar
    k_d_unsc = k_d/scaler_biopar
    
    X = x[0]
    S = x[1]
    
    xdot[0] = mu_max_unsc*S/(K_S_unsc + S)*X - k_d_unsc*X
    xdot[1] = mu_max_unsc*S/(K_S_unsc + S)*X/Y_XS_unsc
    return xdot

def fx_ukf_ode(ode_model, t_span, x0, args_ode = None, args_solver = {}):
    res = scipy.integrate.solve_ivp(ode_model,
                                    t_span,
                                    x0,
                                    args = args_ode,
                                    **args_solver)
    x_all = res.y
    x_final = x_all[:, -1]
    return x_final


#changing x0,P0 of x (determines where the uncertainty is estimated) greatly affects the uncertainty. Regions of low S,X are especially sensitive
x0 = np.array([1., .95001]) + 2
P0 = np.diag(np.array([.95**2., .95**2]))
P0 = np.array([[.95**2., 0.05**2],
               [0.05**2, .95**2]])

# x0 = np.array([1])
# P0 = np.array([[4]])
# x0 = np.array([1., 1.])
# # P0 = np.diag(np.array([1.95**2., .92**2]))

# x0 = np.array([10, 1e-1])
# P0 = np.diag(np.array([.095**2., (.95*1e-1)**2]))
# x0 = np.array([1., 1.])
# P0 = np.diag(np.array([1.95**2., .92**2]))

dim_x = x0.shape[0]
dim_y = 1 #dummy
scaler_var = .6
theta = {"mu": .19445,
         "K_s": .007,
         "Y_XS": .42042,
         "k_d": .006}

t_span = ((0, 1/60)) #integration time

#%% import parameters
N_samples = int(1e5)
par_samples_fx, par_samples_hx, par_names_fx, par_names_hx, par_det_fx, par_det_hx, Q_nom, R_nom, plt_output, par_dist_fx, par_scaling_fx = utils_br.get_literature_values(N_samples = N_samples, plot_par = False)

filter_indices = [0, 1, 2, 4]
par_samples_fx = np.take(par_samples_fx, filter_indices, axis = 0)
par_names_fx = np.take(par_names_fx, filter_indices)


par_mean_fx = np.mean(par_samples_fx, axis = 1)
par_cov_fx = np.cov(par_samples_fx)
dim_par_fx = par_cov_fx.shape[0]

par_true_fx = par_det_fx.copy()
par_kf_fx = par_det_fx.copy()


#true system has random parameter value, ukf uses the mean
for i in range(len(par_names_fx)):
    par_true_fx[par_names_fx[i]] = par_samples_fx[i, -1]
    par_kf_fx[par_names_fx[i]] = par_mean_fx[i]
    
#%% Def GenUT sigmapoints
cm3 = scipy.stats.moment(par_samples_fx.T, moment = 3) 
cm4 = scipy.stats.moment(par_samples_fx.T, moment = 4)


points_genut = spc.GenUTSigmaPoints(dim_par_fx)

s, w_gut_theta = points_genut.compute_scaling_and_weights(par_cov_fx,  #generate scaling and weights
                                                cm3, 
                                                cm4)
# sigmas_gut_theta, P_sqrt = points_genut.compute_sigma_points(par_mean_fx, #sigma points and P_sqrt
#                                                     par_cov_fx, 
#                                                     s)

#%% Make UKF
# kappa = 0
# kappa = 3-dim_x
# # kappa = 1-dim_x#3-dim_x
# points = ukf_sp.JulierSigmaPoints(dim_x, kappa)

# sigmas_x = points.sigma_points(x0, P0)
# points._compute_weights()
# w_x = points.Wm

sqrt_fn = scipy.linalg.sqrtm
# sqrt_fn = lambda P: scipy.linalg.cholesky(P, lower = True)
# sqrt_fn = scipy.linalg.cholesky

y=np.array([43])
dim_y = y.shape[0]
hx = lambda x: x[:dim_y]
fx_ukf = lambda x: fx_ukf_ode(fx_ode, t_span, x, args_ode = [par_true_fx])
# fx_ukf = lambda x: 3*x+x**3 #np.square(x)
points_gut= spc.GenUTSigmaPoints(dim_x, sqrt_method = sqrt_fn)
ukf_jgut = UKF2.UnscentedKalmanFilter(dim_x, dim_y, hx, fx_ukf, points_gut, sqrt_fn = sqrt_fn)
ukf_jgut.x_post = x0.copy()
ukf_jgut.P_post = P0.copy()
ukf_jgut.Q = np.eye(dim_x)
ukf_jgut.R = np.eye(dim_y)
ukf_jgut.predict()
ukf_jgut.update(y)


# #compare with other UKF
# fx_ukf_std = lambda x, dt: fx_ukf(x)
# ukf_std = UKF.UnscentedKalmanFilter(dim_x, dim_y, 100, hx, fx_ukf_std, points, name = "qf")
# ukf_std.x = x0.copy()
# ukf_std.P = P0.copy()
# ukf_std.Q = ukf_jgut.Q.copy()
# ukf_std.R = ukf_jgut.R.copy()
# ukf_std.predict()
# ukf_std.update(y)

# samples = np.random.multivariate_normal(x0, P0, size = int(1e6))
# x2 = fx_ukf(samples)
# x2_m = np.mean(x2, axis = 0)
# x2_cov = np.cov(x2.T)

# print(f"x0: {x0}\n",
#       f"P0: {P0}\n",
#       f"par_true_fx: {par_true_fx}\n",
#       f"x_std_prior: {ukf_std.x_prior}\n",
#       f"x_jgut_prior: {ukf_jgut.x_prior}\n",
#       f"x_sampled: {x2_m}\n",
#       f"P_std_prior: {ukf_std.P_prior}\n",
#       f"P_jgut_prior: {ukf_jgut.P_prior}\n",
#       f"P_sampled: {x2_cov}\n",
#       f"x_std_post: {ukf_std.x_post}\n",
#       f"x_jgut_post: {ukf_jgut.x_post}\n",
#       f"P_std_post: {ukf_std.P_post}\n",
#       f"P_jgut_post: {ukf_jgut.P_post}\n"
#       )

# #%% example from GenUT paper
# mu = np.array([1.5, 1])
# P = np.diag(np.array([1.5, 1]))
# S = mu.copy()
# K = 3*mu**2 + mu

# sigmas, Wm, Wc, P_sqrt = points_gut.compute_sigma_points(mu, P, S = S, K = K)
