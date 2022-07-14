# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:15:36 2022

@author: halvorak
"""

import numpy as np
import scipy.stats
# import scipy.io
# import scipy.integrate
# import casadi as cd
import matplotlib.pyplot as plt
# import matplotlib
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
# import pathlib
import os
# import random
# import sklearn.preprocessing
# import scipy.linalg
# import matplotlib.patches as plt_patches
# import matplotlib.transforms as plt_transforms
import copy
# import time
# import pandas as pd
import seaborn as sns
# import matlab.engine

# Did some modification to these packages
# from myFilter import UKF
# from myFilter import UKF_constrained
from myFilter import sigma_points as ukf_sp
# from myFilter import UKF
# from myFilter import UKF2

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
x0 = np.array([1., .95001])
P0 = np.diag(np.array([.95**2., .95**2]))
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
sigmas_gut_theta, P_sqrt = points_genut.compute_sigma_points(par_mean_fx, #sigma points and P_sqrt
                                                    par_cov_fx, 
                                                    s)

#%% Make UKF
kappa = 0#3-dim_x
kappa = 1-dim_x#3-dim_x
points = ukf_sp.JulierSigmaPoints(dim_x, kappa)

sigmas_x = points.sigma_points(x0, P0)
points._compute_weights()
w_x = points.Wm

# sigmas_x[-1, 1] = -1e-4
# sigmas_x[-1] = [10, 1e-1]

#%% Make w
w_dist = [[] for i in range(points.num_sigmas())]
for i in range(points.num_sigmas()):
    theta_nom = sigmas_gut_theta[:, 0]
    theta = {key: val for (key, val) in zip(par_names_fx, theta_nom)} #dict comprehension - re-create the dict theta with mean (nominal) values (first sigma-points) every time
    
    #point where w should be calcuated
    x_nom = sigmas_x[i]
    fx_nom = fx_ukf_ode(fx_ode, t_span, x_nom, args_ode = [theta])
    fx_get_w = lambda theta_val: (fx_ukf_ode(fx_ode, 
                                            t_span, 
                                            x_nom, 
                                            args_ode = [{key: val for (key, val) in zip(par_names_fx, theta_val)}] #make theta input to the function
                                            ) 
                                  - fx_nom)
    
    w_m, w_c = ut.unscented_transformation(sigmas_gut_theta, w_gut_theta, fx = fx_get_w) #iterates over the sigma points for theta and estimates w_mean/cov
    w_dist[i] = scipy.stats.multivariate_normal(mean = w_m, cov = w_c)

#%%Plot

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

#need samples to plot the confidence ellipse of x0,P0 in the confidence_ellipse function
x0_dist = scipy.stats.multivariate_normal(mean = x0, cov = P0)
x0_rvs = x0_dist.rvs(N_samples)


fig, ax = plt.subplots(1,1)
n_std_w = 500 #how many times the standard deviation of the samples should be increased
for i in range(points.num_sigmas()):
    w_samples = w_dist[i].rvs(size = N_samples) #samples from wi-distribution
    print(f"w_samples: {w_samples.shape}")
    print(f"sigmas_x: {sigmas_x.shape}")
    uncertainty_samples = sigmas_x[i, :] + w_samples
    print(f"uncertainty_samples: {uncertainty_samples.shape}")
    # print("----------")
    label_ellipse = r"$\chi^{(i)}_x + $" + f"{n_std_w}" + r"$\sigma_w$" if i == (points.num_sigmas()-1) else None
    confidence_ellipse(uncertainty_samples[:, 0], uncertainty_samples[:, 1], ax, edgecolor = "g", n_std = n_std_w, facecolor = "g", alpha = .2, label = label_ellipse)
    
confidence_ellipse(x0_rvs[:,0], x0_rvs[:,1], ax, edgecolor = "red", n_std = 1., label = r"$\hat{x}^{+} + 1\sigma_x$")
ax.scatter(sigmas_x[:, 0], sigmas_x[:, 1], label = r"$\chi^{(i)}_x$", s = 10)  
ax.set_xlabel(r"$X$ [g/L]")    
ax.set_ylabel(r"$S$ [g/L]")    


ax.legend()
plt.tight_layout()