# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:58:48 2022

@author: halvorak
"""


import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import casadi as cd
# import matplotlib

# Did some modification to these packages
# from myFilter import UKF
# from myFilter import UKF_constrained
from myFilter import sigma_points as ukf_sp
from myFilter import UKF
from myFilter import UKF2
from myFilter import UKF3
from myFilter import unscented_transform as ut

#Self-written modules
import sigma_points_classes as spc
# import unscented_transformation as ut
# import utils_bioreactor_tuveri as utils_br

#%%Define functions

# def fx_ode(t, x, p):
#     mu_max = p["mu_max"]
#     K_S = p["K_S"]
#     k_d = p["k_d"]
#     Y_XS = p["Y_XS"]
#     xdot = np.zeros(x.shape)
    
#     #"unscale" the parameters
#     scaler_biopar = 1e3
#     mu_max_unsc = mu_max/scaler_biopar
#     K_S_unsc = K_S/scaler_biopar
#     Y_XS_unsc = Y_XS/scaler_biopar
#     k_d_unsc = k_d/scaler_biopar
    
#     X = x[0]
#     S = x[1]
    
#     xdot[0] = mu_max_unsc*S/(K_S_unsc + S)*X - k_d_unsc*X
#     xdot[1] = mu_max_unsc*S/(K_S_unsc + S)*X/Y_XS_unsc
#     return xdot

# def fx_ukf_ode(ode_model, t_span, x0, args_ode = None, args_solver = {}):
#     res = scipy.integrate.solve_ivp(ode_model,
#                                     t_span,
#                                     x0,
#                                     args = args_ode,
#                                     **args_solver)
#     x_all = res.y
#     x_final = x_all[:, -1]
#     return x_final



def fx_ukf(x, p):
    # x_next = x*p**2
    x_next = (np.cos(x)*x**5)*p**2
    return x_next

#changing x0,P0 of x (determines where the uncertainty is estimated) greatly affects the uncertainty. Regions of low S,X are especially sensitive
x0 = np.array([1., .95001])
P0 = np.diag(np.array([.95**2., .95**2]))
P0 = np.array([[.95**2., 0.05**2],
               [0.05**2, .95**2]])

x0 = np.array([-1.])
P0 = np.array([[.5**2]])


dim_x = x0.shape[0]
dim_y = 1 #dummy


#%% import parameters
#assume normal distribution for now
par_mean_fx = np.array([.99, .97])
par_cov_fx = np.array([[.5**2., 0.05**2],
                       [0.05**2, .55**2]])
par_mean_fx = np.array([.97])
par_cov_fx = np.array([[.5**2]])
par_cm3 = None
par_cm4 = None

par_samples = np.random.gamma(shape=1., scale = 2., size = int(1e4) )
par_mean_fx = np.atleast_1d(par_samples.mean())
par_cov_fx = np.atleast_2d(par_samples.var())
par_cm3 = np.atleast_1d(scipy.stats.moment(par_samples, moment = 3))
par_cm4 = np.atleast_1d(scipy.stats.moment(par_samples, moment = 4))
# plt.hist(par_samples)

dim_par_fx = par_cov_fx.shape[0]
#%% Def GenUT sigmapoints for the parameters
sqrt_fn = scipy.linalg.sqrtm
# sqrt_fn = lambda P: scipy.linalg.cholesky(P, lower = True)
# sqrt_fn = scipy.linalg.cholesky

points_genut = spc.GenUTSigmaPoints(dim_par_fx, sqrt_method = sqrt_fn)

(sigmas_par, Wm_par, 
 Wc_par, 
 P_par_sqrt) = points_genut.compute_sigma_points(par_mean_fx,
                                                par_cov_fx,
                                                S = par_cm3, #normal
                                                K = par_cm4, #normal
                                                s1 = None, #min CM4
                                                sqrt_method = sqrt_fn
                                                )

#%% Make UKF

y=np.array([43])

dim_y = y.shape[0]
hx = lambda x: x[:dim_y]
fx = lambda x: fx_ukf(x[:dim_x], x[dim_x:]) #based on augmented states description

#%% UKF-joint

# fx_ukf = lambda x: 3*x+x**3 #np.square(x)
points_gut= spc.GenUTSigmaPoints(dim_x, sqrt_method = sqrt_fn)
# ukf_jgut = UKF3.UnscentedKalmanFilter(dim_x, dim_y, hx, fx, points_gut, sqrt_fn = sqrt_fn)
ukf_jgut = UKF3.UnscentedKalmanFilter(dim_x, dim_y, hx, fx, points_gut, dim_par_fx, sigmas_par, Wm_par, sqrt_fn=sqrt_fn, name = None)

ukf_jgut.x_post = x0.copy()
ukf_jgut.P_post = P0.copy()
ukf_jgut.Q = np.eye(dim_x)
ukf_jgut.R = np.eye(dim_y)
ukf_jgut.predict()

fx_test = lambda theta: fx_ukf(x0, theta) - fx_ukf(x0, par_mean_fx)
sigmas_test = ukf_jgut.compute_transformed_sigmas(sigmas_par, fx_test)
w_mean_gut, Q_gut = ut.unscented_transformation_gut(sigmas_test, Wm_par, Wc_par)

#%% UKF-EKF pred

# fx_ukf = lambda x: 3*x+x**3 #np.square(x)
points_gut2= spc.GenUTSigmaPoints(dim_x, sqrt_method = sqrt_fn)
# ukf_ekf = UKF3.UnscentedKalmanFilter(dim_x, dim_y, hx, fx, points_gut, sqrt_fn = sqrt_fn)
ukf_ekf = UKF3.UnscentedKalmanFilter_EKF_based(dim_x, dim_y, hx, fx, points_gut2, dim_par_fx, sigmas_par.copy(), Wm_par.copy(), sqrt_fn=sqrt_fn, name = None)

ukf_ekf.x_post = x0.copy()
ukf_ekf.P_post = P0.copy()
ukf_ekf.Q = np.eye(dim_x)
ukf_ekf.R = np.eye(dim_y)
ukf_ekf.predict()

fx_test = lambda theta: fx_ukf(x0, theta) - fx_ukf(x0, par_mean_fx)
sigmas_test = ukf_ekf.compute_transformed_sigmas(sigmas_par, fx_test)
w_mean_gut, Q_gut = ut.unscented_transformation_gut(sigmas_test, Wm_par, Wc_par)

# ukf_jgut.update(y)


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
