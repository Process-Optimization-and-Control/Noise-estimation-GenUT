# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 09:34:34 2022

@author: halvorak
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:47:32 2021

@author: halvorak
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 09:10:12 2021

@author: halvorak
"""

import numpy as np
import scipy.stats
# import scipy.io
# import scipy.integrate
import casadi as cd
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import os
import random
# import sklearn.preprocessing
# import scipy.linalg
# import matplotlib.patches as plt_patches
# import matplotlib.transforms as plt_transforms
import copy
import time
import timeit
import pandas as pd
import seaborn as sns

# Did some modification to these packages
# from myFilter import UKF
# from myFilter import UKF_constrained
from myFilter import sigma_points as ukf_sp
import sigma_points_classes as ukf_spc
from myFilter import UKF
from myFilter import UKF2
from myFilter import myExceptions

#Self-written modules
# import sigma_points_classes as spc
import unscented_transformation as ut
import utils_bioreactor_tuveri as utils_br
font = {'size': 14}
matplotlib.rc('font', **font)
# cmap = "tab10"
# plt.set_cmap(cmap)


#%% Set N simulation times
N_sim = 10 #this is how many times to repeat each iteration
x_var = ["V", "X", "S", "CO2"]
dim_x = len(x_var)

cost_func_type = "RMSE" #other valid option is "valappil"

filters_to_run = ["gut", 
                     "jgut", #joint gut
                     "jgutw", #joint gut based on w - equation
                         "lin", 
                   # #   "mc", 
                    "qf"
                  ]

j_valappil_gut = np.zeros((dim_x, N_sim)) #save values for cost function
j_valappil_jgut = np.zeros((dim_x, N_sim)) #with block factorization
j_valappil_jgutw = np.zeros((dim_x, N_sim)) #with block factorization
j_valappil_mc = np.zeros((dim_x, N_sim))
j_valappil_lin = np.zeros((dim_x, N_sim))
j_valappil_qf = np.zeros((dim_x, N_sim))

#See Barfoot page 95, he says we need both mean and rmse
j_mean_gut = np.zeros((dim_x, N_sim)) #save values for cost function
j_mean_jgut = np.zeros((dim_x, N_sim)) #with block factorization
j_mean_jgutw = np.zeros((dim_x, N_sim)) #with block factorization
j_mean_mc = np.zeros((dim_x, N_sim))
j_mean_lin = np.zeros((dim_x, N_sim))
j_mean_qf = np.zeros((dim_x, N_sim))

time_sim_gut = np.zeros(N_sim) #ave simulation time
time_sim_jgut = np.zeros(N_sim)
time_sim_jgutw = np.zeros(N_sim)
time_sim_mc = np.zeros(N_sim)
time_sim_lin = np.zeros(N_sim)
time_sim_qf = np.zeros(N_sim)

consistency_1s_jgut = np.zeros((dim_x, N_sim))
consistency_2s_jgut = np.zeros((dim_x, N_sim))
consistency_1s_gut = np.zeros((dim_x, N_sim))
consistency_2s_gut = np.zeros((dim_x, N_sim))
consistency_1s_jgutw = np.zeros((dim_x, N_sim))
consistency_2s_jgutw = np.zeros((dim_x, N_sim))
consistency_1s_lin = np.zeros((dim_x, N_sim))
consistency_2s_lin = np.zeros((dim_x, N_sim))
consistency_1s_qf = np.zeros((dim_x, N_sim))
consistency_2s_qf = np.zeros((dim_x, N_sim))

Ni = 0
# rand_seed = 1235
rand_seed = 6969
# rand_seed = 16535+5

ts = time.time()
ti = time.time()


print_subiter = True #print certain timesteps for a single case
num_exceptions = 0 #number of times we fail and start over
while Ni < N_sim:
    try:
        np.random.seed(rand_seed) #to get reproducible results. rand_seed updated in every iteration
        random.seed(rand_seed) #to get reproducible results. rand_seed updated in every iteration
        t_iter = time.time()
        #%% Import parameters
        par_samples_fx, par_samples_hx, par_names_fx, par_names_hx, par_det_fx, par_det_hx, Q_nom, R_nom, plt_output, par_dist_fx, par_scaling_fx = utils_br.get_literature_values(N_samples = int(5e3), plot_par = False)
        
        par_cov_fx = np.cov(par_samples_fx)
        dim_par_fx = par_cov_fx.shape[0]
        
        #For q_air, S_in set cov=0 (we know they are uncorrelated with biological system)
        uncorrelated_par = ["S_in", "q_air"]
        for p_name in par_dist_fx.keys():
            p_idx = par_names_fx.index(p_name)
            variance_par = par_cov_fx[p_idx, p_idx]
            par_cov_fx[p_idx, :] = np.zeros(dim_par_fx)
            par_cov_fx[:, p_idx] = np.zeros(dim_par_fx)
            par_cov_fx[p_idx, p_idx] = variance_par
        
        par_cov_uncorrelated_fx = np.diag(np.diag(par_cov_fx).copy())
        
        df_par = pd.DataFrame(data = par_samples_fx.T, 
                              columns = par_names_fx)
        
        # mode = {}
        # for key, dist in par_dist_fx.items():
        #     mode[key] = scipy.optimize.minimize(lambda theta: -dist.pdf(theta),
        #                                    dist.mean(), #use mean as theta0
        #                                    tol = 1e-10).x[0]
        
        #%% fx parameters
        par_true_fx = par_det_fx.copy()
        par_kf_fx = par_det_fx.copy()
        
        par_mean_fx = np.mean(par_samples_fx, axis = 1)
        
        #true system has random parameter value, ukf uses the mean
        for i in range(len(par_names_fx)):
            par_true_fx[par_names_fx[i]] = par_samples_fx[i, -1]
            par_kf_fx[par_names_fx[i]] = par_mean_fx[i]
            
        
        #%% hx parameters - not used in this case study
        par_true_hx = par_det_hx.copy()
        par_kf_hx = par_det_hx.copy()
        for i in range(len(par_names_hx)):
            par_true_hx[par_names_hx[i]] = par_samples_hx.mean() 
            par_kf_hx[par_names_hx[i]] = par_samples_hx.mean() 
        
        #%% Define dimensions and initialize arrays
                
        x0 = utils_br.get_x0_literature()
        u0 = utils_br.get_u0_literature()
        
        dim_x = x0.shape[0]
        dim_par_fx = len(par_true_fx)
        dim_par_hx = len(par_true_hx)
        dim_u = u0.shape[0]
        dt_y = 1/60 # [h] Measurement frequency
        
        # t_end = 15 # [h]
        t_end = 9 # [h]
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
        ## in case we want to specify inflow at a certain timepoint.
        # idx_u_increase = np.searchsorted(t, 21, side = "right") #after 21h, we have inflow
        # u[0, idx_u_increase] = (.5* #L/min
        #                         60) #min/h ==> L/h
        
        #UKF initial states
        P0 = utils_br.get_P0_literature()
        x0_kf = utils_br.get_x0_kf_random(eps=1e-4) # random value from x0_kf from the multivariate normal dist with (mean=x0_true, cov = P0) and x0_kf >= eps
        
        #arrays for the "joint" genUT
        Pa0 = scipy.linalg.block_diag(par_cov_fx, P0)
        xa0_kf = np.hstack((par_mean_fx, x0_kf))
        dim_xa = dim_par_fx + dim_x
        cm3_par = scipy.stats.moment(par_samples_fx, 
                                     moment = 3, 
                                     axis = 1)
        cm4_par = scipy.stats.moment(par_samples_fx, 
                                     moment = 4, 
                                     axis = 1)
        
        #Arrays where values are stored
        x_true = np.zeros((dim_x, dim_t)) 
        x_ol = np.zeros((dim_x, dim_t)) #Open loop simulation - same starting point and param as UKF
        
        #Arrays where posterior prediction is stored
        x_post_gut = np.zeros((dim_x, dim_t))
        x_post_jgut = np.zeros((dim_xa, dim_t))
        x_post_jgutw = np.zeros((dim_xa, dim_t))
        x_post_mc = np.zeros((dim_x, dim_t))
        x_post_lin = np.zeros((dim_x, dim_t))
        x_post_qf = np.zeros((dim_x, dim_t))
        
        #Track history of computed w_mean-s 
        w_jgutw_hist = np.zeros((dim_xa, dim_t))
        w_gut_hist = np.zeros((dim_x, dim_t))
        w_mc_hist = np.zeros((dim_x, dim_t))
        
        #Track history of computed Q-s (only diagonals)
        Q_gut_hist = np.zeros((dim_x, dim_t))
        Q_jgutw_hist = np.zeros((dim_xa, dim_t))
        Q_mc_hist = np.zeros((dim_x, dim_t))
        Q_lin_hist = np.zeros((dim_x, dim_t))
        # Q_qf_hist = np.zeros((dim_x, dim_t)) #not necessary, except for plot
        
        #diagnonal elements of covariance matrices
        P_diag_post_gut = np.zeros((dim_x, dim_t))
        P_diag_post_jgut = np.zeros((dim_xa, dim_t))
        P_diag_post_jgutw = np.zeros((dim_xa, dim_t))
        P_diag_post_mc = np.zeros((dim_x, dim_t))
        P_diag_post_lin = np.zeros((dim_x, dim_t))
        P_diag_post_qf = np.zeros((dim_x, dim_t))
        
        #save the starting points for the true system and the filters
        x_true[:, 0] = x0
        x_post_mc[:, 0] = x0_kf.copy()
        x_post_lin[:, 0] = x0_kf.copy()
        x_post_gut[:, 0] = x0_kf.copy()
        x_post_jgut[:, 0] = xa0_kf.copy()
        x_post_jgutw[:, 0] = xa0_kf.copy()
        x_post_qf[:, 0] = x0_kf.copy()
        x0_ol = x0_kf.copy()
        x_ol[:, 0] = x0_ol
        # x_prior[:, 0] = x0_kf
        
        #save starting points for covariance matrices
        P_diag_post_gut[:, 0] = np.diag(P0.copy())
        P_diag_post_jgut[:, 0] = np.diag(Pa0.copy())
        P_diag_post_jgutw[:, 0] = np.diag(Pa0.copy())
        P_diag_post_mc[:, 0] = np.diag(P0.copy())
        P_diag_post_lin[:, 0] = np.diag(P0.copy())
        P_diag_post_qf[:, 0] = np.diag(P0.copy())
        
        t_span = (t[0],t[1])
        solver_tol_default_mode = 1e-5 #for maximizing the pdf \approx kde(w_stoch).
        
        #%% Define UKF with adaptive Q, R from GenUT
        # alpha = 1
        # beta = 0
        # kappa = 3-dim_x
        alpha = 1e-2
        beta = 2.
        kappa = 0.#3-dim_x
        old_gut = False
        if old_gut:
            points_gut = ukf_sp.MerweScaledSigmaPoints(dim_x,
                                                    alpha,
                                                    beta,
                                                    kappa)
            
            fx_ukf_gut = None #updated later in the simulation
            hx_ukf_gut = None #updated later in the simulation
            
            kfc_gut = UKF.UnscentedKalmanFilter(dim_x = dim_x, 
                                                                    dim_z = dim_y, 
                                                                    dt = 100, 
                                                                    hx = hx_ukf_gut, 
                                                                    fx = fx_ukf_gut,
                                                                    points = points_gut
                                                                    )
            kfc_gut.x = x_post_gut[:, 0]
            kfc_gut.P = copy.deepcopy(P0)
        else:
            points_gut = ukf_spc.GenUTSigmaPoints(dim_x)
            
            fx_ukf_gut = None #updated later in the simulation
            hx_ukf_gut = None #updated later in the simulation
            
            kfc_gut = UKF2.UnscentedKalmanFilter(dim_x = dim_x, 
                                                                    dim_y = dim_y, 
                                                                    hx = hx_ukf_gut, 
                                                                    fx = fx_ukf_gut,
                                                                    points = points_gut
                                                                    )
            kfc_gut.x_post = x_post_gut[:, 0]
            kfc_gut.P_post = copy.deepcopy(P0)
        kfc_gut.Q = None #to be updated in a loop
        kfc_gut.R = R_nom 
        
        #%% Define UKF with adaptive Q, R from joint GenUT 
        points_jgut = ukf_sp.ParametricUncertaintyGenUTSigmaPoints(
            par_cov_fx, 
                                                                   dim_x,
                                                                   cm3_par,
                                                                   cm4_par)
        points_jgut.Wm, points_jgut.Wc = None, None #avoid printing error
        fx_ukf_jgut = None #updated later in the simulation
        hx_ukf_jgut = None #updated later in the simulation
        
        kfc_jgut = UKF.UnscentedKalmanFilterParametricUncertainty(dim_par = dim_par_fx,
                                                                  dim_x = dim_x, 
                                                                dim_z = dim_y, 
                                                                dt = 100, 
                                                                hx = hx_ukf_jgut, 
                                                                fx = fx_ukf_jgut,
                                                                name = "jgut",
                                                                points = points_jgut
                                                               
                                                                )
        kfc_jgut.x = x_post_jgut[:, 0]
        kfc_jgut.P = Pa0.copy()
        kfc_jgut.Q = None #to be updated in a loop
        kfc_jgut.R = R_nom #to be updated in a loop
        
        #%% Define UKF with adaptive Q, R from joint GenUT based on w 
        points_jgutw = ukf_sp.ParametricUncertaintyGenUTSigmaPoints(
            par_cov_fx, 
                                                                   dim_x,
                                                                   cm3_par,
                                                                   cm4_par, block_factorization = True)
        points_jgutw.Wm, points_jgutw.Wc = None, None #avoid printing error
        fx_ukf_jgutw = None #updated later in the simulation
        hx_ukf_jgutw = None #updated later in the simulation
        
        kfc_jgutw = UKF.JointGenUTFilter(dim_par = dim_par_fx,
                                                                  dim_x = dim_x, 
                                                                dim_z = dim_y, 
                                                                dt = 100, 
                                                                hx = hx_ukf_jgutw, 
                                                                fx = fx_ukf_jgutw,
                                                                name = "jgutw",
                                                                points = points_jgutw
                                                               
                                                                )
        kfc_jgutw.x = x_post_jgutw[:, 0]
        kfc_jgutw.P = Pa0.copy()
        kfc_jgutw.Q = None #to be updated in a loop
        kfc_jgutw.R = R_nom 
        
        
        #%% Define UKF with adaptive Q, R from MC
        alpha_mc = copy.copy(alpha)
        beta_mc = copy.copy(beta)
        kappa_mc = copy.copy(kappa)
        points_mc = ukf_sp.MerweScaledSigmaPoints(dim_x,
                                                alpha_mc,
                                                beta_mc,
                                                kappa_mc)
        fx_ukf_mc = None #updated later in the simulation
        hx_ukf_mc = None #updated later in the simulation
        
        kfc_mc = UKF.UnscentedKalmanFilter(dim_x = dim_x, 
                                                                dim_z = dim_y, 
                                                                dt = 100, 
                                                                hx = hx_ukf_mc, 
                                                                fx = fx_ukf_mc,
                                                                points = points_mc
                                                                )
        kfc_mc.x = x_post_mc[:, 0]
        kfc_mc.P = copy.deepcopy(P0)
        kfc_mc.Q = None #to be updated in a loop
        kfc_mc.R = R_nom #to be updated in a loop
        Q_mc = Q_nom.copy()
        
        
        #%% Define UKF with adaptive Q, R from linearized approach
        alpha_lin = copy.copy(alpha)
        beta_lin = copy.copy(beta)
        kappa_lin = copy.copy(kappa)
        points_lin = ukf_sp.MerweScaledSigmaPoints(dim_x,
                                                alpha_lin,
                                                beta_lin,
                                                kappa_lin)
        fx_ukf_lin = None
        hx_ukf_lin = None
        
        kfc_lin = UKF.UnscentedKalmanFilter(dim_x = dim_x, 
                                                                dim_z = dim_y, 
                                                                dt = 100, 
                                                                hx = hx_ukf_lin, 
                                                                fx = fx_ukf_lin,
                                                                name = "lin",
                                                                points = points_lin
                                                               )
        kfc_lin.x = x_post_lin[:, 0]
        kfc_lin.P = copy.deepcopy(P0)
        kfc_lin.Q = None #to be updated in a loop
        kfc_lin.R = R_nom #to be updated in a loop
        Q_lin = Q_nom.copy()
       
        
        #%% Define UKF with fixed Q, R (hand tuned)
        alpha_qf = copy.copy(alpha)
        beta_qf = copy.copy(beta)
        kappa_qf = copy.copy(kappa)
        points_qf = ukf_sp.MerweScaledSigmaPoints(dim_x,
                                                alpha_qf,
                                                beta_qf,
                                                kappa_qf)
        fx_ukf_qf = None
        hx_ukf_qf = None
        
        kfc_qf = UKF.UnscentedKalmanFilter(dim_x = dim_x, 
                                                                dim_z = dim_y, 
                                                                dt = 100, 
                                                                hx = hx_ukf_qf, 
                                                                fx = fx_ukf_qf,
                                                                name = "qf",
                                                                points = points_qf
                                                                )
        kfc_qf.x = x_post_qf[:, 0]
        kfc_qf.P = copy.deepcopy(P0)
        kfc_qf.Q = None #to be updated in a loop
        kfc_qf.R = R_nom #to be updated in a loop
        Q_qf = Q_nom.copy()
        
        
        #%% Get parametric uncertainty of fx by GenUT. Generate sigmapoints first ("offline")
        sigmas_fx_gut, w_fx_gut = utils_br.get_sigmapoints_and_weights(par_samples_fx, samples = True)
        list_dist_fx_keys = list(par_names_fx.copy()) # list of parameters with distribution. This variable can be deleted in this case study actually
        
        #%% N_MC samples, random sampling
        # N_mc_dist = int(50)
        # N_mc_dist = int(100)
        N_mc_dist = int(500)
        
        #par_mc_fx is a np.array((dim_par, N_mc_dist)) with random amples from par_samples_fx
        par_mc_fx, sns_grid_mc = utils_br.get_mc_points(
            par_samples_fx, 
            N_mc_dist = N_mc_dist, 
            plot_mc_samples = False,
            labels = list(par_kf_fx.keys())
            )
        
        
        #%% Q_fixed, robustness
        Q_diag_min = np.eye(dim_x)*1e-10
        # Q_diag_min = np.eye(dim_x)*1e-6
        # Q_diag_min = np.zeros((dim_x, dim_x))
        
        #add Q_diag_min to all filters
        Q_qf = Q_qf + Q_diag_min #not necessary, but making it the same for all tuning approaches
        kfc_qf.Q = Q_qf
        
        #jgut sees this as a fixed Q-matrix for the x-part and 0 for the parameter part
        kfc_jgut.Q = scipy.linalg.block_diag(np.zeros((dim_par_fx, dim_par_fx)),
                                             Q_diag_min)
        kfc_jgutw_Q_min = kfc_jgut.Q.copy()
        
        
        #%% Casadi integrator, jacobian df/dp and solvers for the mode
        F,jac_p_func,_,_,_,_,_,_,_,_,_,_,_= utils_br.ode_model_plant()
        opts_nlp_solver = {"print_time": 0, "ipopt": {"acceptable_tol": 1e-8, "print_level": 0}}
        
        #%% Simulate the plant and UKF
        
        par_samples_fx_temp = par_samples_fx[:, :-1].copy() #this new np.array() will be "popped". Already used last element in par_samples_fx (that's why we skip it here)
        for i in range(1, dim_t):
            t_span = (t[i-1], t[i])
            
            #Simulate the true plant
            x_true[:, i] = utils_br.integrate_ode(F, 
                                                  x_true[:,i-1],
                                                  t_span, 
                                                  u[:, i-1],
                                                  par_true_fx)
            
            # #if we obtain a negative x_true-value, it is unphysical and due to the numerical integration. If negative value is detected, set the value to 0 (or close to)
            # neg_xtrue_val = x_true[:, i] <= 0
            # x_true[neg_xtrue_val, i] = 1e-10
            
            #Simulate the open loop (kf parameters and starting point)
            x_ol[:, i] = utils_br.integrate_ode(F, 
                                                x_ol[:,i-1], 
                                                t_span, 
                                                u[:, i-1], 
                                                par_kf_fx)
            
            # neg_xtrue_val = x_ol[:, i] <= 0
            # x_ol[neg_xtrue_val, i] = 1e-10
            
            #Make a new measurement
            y[:, i] = (utils_br.hx(x_true[:, i], par_true_hx) + #deterministic
                        utils_br.vx(x_true[:, i], par_true_hx) #noise depends on the state
                        )
            
            #Control inputs for the plant. 
            control_law = 1 
            if control_law == 1:
                #Option 1) If less than 2 g/L of sugar, add more
                if x_true[2, i] < 2:
                    u[0, i] = (.5* #L/min
                               60) #min/h ==> L/h
            elif control_law == 2:
                #Option 2) If x_ol < 0,1 g/L for more than 0.5h, add more sugar
                if x_ol[2, i] <= .1:
                    if t[i] - t_low_sugar_ol >= .5:
                        u[0, i] = (.5* #L/min
                                   60) #min/h ==> L/h
                        t_low_sugar_ol = t[-1] #dummy
                    elif t_low_sugar_ol > t[i]: #low sugar predicted for the first time
                        t_low_sugar_ol = t[i]
                    else:
                        pass #low sugar has been predicted less than 30min ago
            
            # if print_subiter:
            #     if (i%100) == 0:
            #         print(f"Subiter {i}/{dim_t} complete. t_iter = {time.time()-ti: .2f} and t_tot = {time.time()-ts: .2f}. Main iter {Ni+1}/{N_sim}")
            #         ti = time.time()
                
            #Select new parameters for the plant. par_samples are already random, so select from the end of the array
            par_fx_new = par_samples_fx_temp[:, -1] 
            par_samples_fx_temp = par_samples_fx_temp[:, :-1] #popping the par_samples_fx_temp array
            for idx_p in range(len(par_names_fx)):
                a_par_name = par_names_fx[idx_p]
                if not a_par_name == "S_in": #S_in does not change at every time point
                    par_true_fx[a_par_name] = par_fx_new[idx_p]
            par_history_fx[:, i] = list(par_true_fx.values())    
        
        #end of for i in range(dim_t): (or end of simulation)
        
        y[:, 0] = np.nan #the 1st measurement is not real, just for programming convenience
            
        #%% Run state estimators
        #Get i) process noise statistics and ii) prior estimates for the different UKFs
        if "gut" in filters_to_run:
            #Adaptive Q by GenUT - no mean adjustment
            ts_gut = timeit.default_timer()
            for i in range(1, dim_t):
                t_span = (t[i-1], t[i])
                
                x_nom_gut = utils_br.integrate_ode(F, x_post_gut[:,i-1], t_span, u[:, i-1], par_kf_fx)
                
                fx_gen_Q_gut = lambda si: utils_br.fx_for_UT_gen_Q(si, list_dist_fx_keys, F, x_post_gut[:, i-1], t_span, u[:, i-1], par_kf_fx.copy()) - x_nom_gut
                
                
                w_mean_gut, Q_gut = ut.unscented_transformation(sigmas_fx_gut, w_fx_gut, fx = fx_gen_Q_gut)
                
                Q_gut = Q_gut + Q_diag_min #robustness/non-zero on diagonals
                kfc_gut.Q = Q_gut #assign to filter
                Q_gut_hist[:, i] = np.diag(Q_gut) #Save Q history
                if old_gut:
                    fx_ukf_gut = lambda x, dt_kf: utils_br.integrate_ode(F, x, t_span, u[:, i-1], par_kf_fx.copy()) + w_mean_gut #prediction function
                    kfc_gut.predict(fx = fx_ukf_gut)
                    
                    hx_gut = lambda x_in: utils_br.hx(x_in, par_kf_hx.copy()) 
                    kfc_gut.update(y[:, i], hx = hx_gut)
                
                    x_post_gut[:, i] = kfc_gut.x
                    P_diag_post_gut[:, i] = np.diag(kfc_gut.P)
                else:
                    fx_ukf_gut = lambda x: utils_br.integrate_ode(F, x, t_span, u[:, i-1], par_kf_fx.copy()) + w_mean_gut #prediction function
                    kfc_gut.predict(fx = fx_ukf_gut)
                    
                    hx_gut = lambda x_in: utils_br.hx(x_in, par_kf_hx.copy()) 
                    kfc_gut.update(y[:, i], hx = hx_gut)
                    x_post_gut[:, i] = kfc_gut.x_post
                    P_diag_post_gut[:, i] = np.diag(kfc_gut.P_post)
                
            tf_gut = timeit.default_timer()
            time_sim_gut[Ni] = tf_gut - ts_gut
        
        if "jgut" in filters_to_run:
            #Joint estimation of states and "estimation" of parameters (we only want P_x/par terms, see the paper)
            ts_jgut = timeit.default_timer()
            for i in range(1, dim_t):
                t_span = (t[i-1], t[i])
                fx_ukf_jgut = lambda x, dt_kf: utils_br.integrate_ode_parametric_uncertainty(F, x, t_span, u[:, i-1], par_kf_fx.copy(), dim_par_fx) #prediction function
                kfc_jgut.predict(fx = fx_ukf_jgut)
    
                hx_jgut = lambda x_in: utils_br.hx(x_in[-dim_x:], par_kf_hx.copy()) #only send in x, not xa
                kfc_jgut.update(y[:, i], hx = hx_jgut)
                
                x_post_jgut[:, i] = kfc_jgut.x
                P_diag_post_jgut[:, i] = np.diag(kfc_jgut.P)
            tf_jgut = timeit.default_timer()
            time_sim_jgut[Ni] = tf_jgut - ts_jgut
            # #check joint gut is doing as it should
            # norm_x_jgut = np.linalg.norm(kfc_jgut.x[:dim_par_fx] - par_mean_fx, ord = None)
            # norm_P_jgut = np.linalg.norm(kfc_jgut.P[:dim_par_fx, :dim_par_fx] - par_cov_fx, ord = "fro")
            
            # norm_diff_P = np.linalg.norm(kfc_jgut.P[-dim_x:, -dim_x:] - kfc_gutw.P, ord = "fro"
            #                              ) #difference jgut and gutw
            # print(f"norm_x_jgut: {norm_x_jgut}\n",
            #       f"norm_P_jgut: {norm_P_jgut}\n",
            #       f"norm_diff jgut and gut: {norm_diff_P}")
        
        if "jgutw" in filters_to_run:
            #Joint estimation of states and "estimation" of parameters (we only want P_x/par terms, see the paper)
            ts_jgutw = timeit.default_timer()
            kfc_jgutw.Q = np.zeros((dim_xa, dim_xa))
            kappa_Q = 1.
            for i in range(1, dim_t):
                t_span = (t[i-1], t[i])
                
                
                fx_ukf_jgutw = lambda x, dt_kf: utils_br.integrate_ode_parametric_uncertainty(F, x, t_span, u[:, i-1], par_kf_fx.copy(), dim_par_fx) #prediction function
                
                jgutw_x_nom = fx_ukf_jgutw(x_post_jgutw[:, i-1], None)
                
                fx_jgutw = lambda x, dt_kf: fx_ukf_jgutw(x, dt_kf) - jgutw_x_nom
                
                #test setting cross correlation terms to zero
                kfc_jgutw.P[dim_par_fx:, :dim_par_fx] = np.zeros((dim_x, dim_par_fx))
                kfc_jgutw.P[:dim_par_fx, dim_par_fx:] = np.zeros((dim_par_fx, dim_x))
                
                jgutw_w_mean, jgutw_Qk = kfc_jgutw.predict(
                    fx = fx_jgutw)
                # jgutw_w_mean[:dim_par_fx] = 0
                # jgutw_Qk[:dim_par_fx, :dim_par_fx] = 0
                jgutw_x_prior = jgutw_x_nom + jgutw_w_mean
                w_jgutw_hist[:, i] = jgutw_w_mean #Save w_mean history
                Q_jgutw_hist[:, i] = np.diag(jgutw_Qk + kfc_jgutw_Q_min) #Save Q history
                # jgutw_x_prior[:dim_par_fx] = par_mean_fx
                jgutw_x_nom2 = jgutw_x_nom.reshape(-1, 1)
                jgutw_x_prior2 = jgutw_x_prior.reshape(-1, 1)
                # raise KeyError
                jgutw_Pxx = (jgutw_Qk 
                              + 2*jgutw_x_prior2 @ jgutw_x_nom2.T
                              - jgutw_x_prior2 @ jgutw_x_prior2.T
                              # + jgutw_x_nom2 @ jgutw_x_prior2.T
                               - jgutw_x_nom2 @ jgutw_x_nom2.T
                              ) + kfc_jgutw_Q_min
                # jgutw_Pxx = (jgutw_Qk 
                #               + jgutw_x_prior2 @ jgutw_x_nom2.T
                #               - jgutw_x_prior2 @ jgutw_x_prior2.T
                #               # + jgutw_x_nom2 @ jgutw_x_prior2.T
                #               # - jgutw_x_nom2 @ jgutw_x_nom2.T
                #               )
                # jgutw_Pxx = jgutw_Qk
                # print(jgutw_Pxx)
                jgutw_Pxx = .5*(jgutw_Pxx + jgutw_Pxx.T) #ensure symmetry
                # jgutw_Pxx += jgutw_Qk #add process noise
                jgutw_Pxx[:dim_par_fx, :dim_par_fx] = par_cov_fx
                # print("after symmetry")
                # print(jgutw_Pxx)
                
                kfc_jgutw.x_prior = jgutw_x_prior
                kfc_jgutw.x = jgutw_x_prior
                kfc_jgutw.P_prior = jgutw_Pxx
                kfc_jgutw.P = jgutw_Pxx
                
                kfc_jgutw.sigmas_f = kfc_jgutw.points_fn.sigma_points(kfc_jgutw.x, kfc_jgutw.P) #these sigma points have not gone through any transformation - it is just the pure sigma points (not f(sigma points))
                kfc_jgutw.Wm, kfc_jgutw.Wc = kfc_jgutw.points_fn.Wm, kfc_jgutw.points_fn.Wc #with the genUT the weights can  change since they are dependent on the scaling parameters (which are dependent of self.P)
                
    
                hx_jgutw = lambda x_in: utils_br.hx(x_in[-dim_x:], par_kf_hx.copy()) #only send in x, not xa
                kfc_jgutw.update(y[:, i], hx = hx_jgutw)
                
                x_post_jgutw[:, i] = kfc_jgutw.x
                P_diag_post_jgutw[:, i] = np.diag(kfc_jgutw.P)
            tf_jgutw = timeit.default_timer()
            time_sim_jgutw[Ni] = tf_jgutw - ts_jgutw
            # #check joint gut is doing as it should
            # norm_x_jgutw = np.linalg.norm(kfc_jgutw.x[:dim_par_fx] - par_mean_fx, ord = None)
            # norm_P_jgutw = np.linalg.norm(kfc_jgutw.P[:dim_par_fx, :dim_par_fx] - par_cov_fx, ord = "fro")
            
            # norm_diff_P = np.linalg.norm(kfc_jgutw.P[-dim_x:, -dim_x:] - kfc_gutw.P, ord = "fro"
            #                              ) #difference jgutw and gutw
            # print(f"norm_x_jgutw: {norm_x_jgutw}\n",
            #       f"norm_P_jgutw: {norm_P_jgutw}\n",
            #       f"norm_diff jgutw and gut: {norm_diff_P}")
        
       
        if "mc" in filters_to_run:
            #Adaptive Q by MC random
            ts_mc = timeit.default_timer()
            for i in range(1, dim_t):
                t_span = (t[i-1], t[i])
                
                w_mean_mc, Q_mc = utils_br.get_wmean_Q_from_mc(par_mc_fx, #get_wmeanXXX or get_wmodeXXX
                                                                F,
                                                                x_post_mc[:, i-1], 
                                                                t_span, 
                                                                u[:, i-1],
                                                                par_kf_fx.copy())
                Q_mc = Q_mc + Q_diag_min #robustness/non-zero on diagonals
                kfc_mc.Q = Q_mc #assign to filter
                Q_mc_hist[:, i] = np.diag(Q_mc) #Save Q history
                fx_ukf_mc = lambda x, dt_kf: utils_br.integrate_ode(F, x, t_span, u[:, i-1], par_kf_fx)
                kfc_mc.predict(fx = fx_ukf_mc)
        
                hx_mc = lambda x_in: utils_br.hx(x_in, par_kf_hx.copy()) 
                kfc_mc.update(y[:, i], hx = hx_mc)
        
                x_post_mc[:, i] = kfc_mc.x
                P_diag_post_mc[:, i] = np.diag(kfc_mc.P)
                
            tf_mc = timeit.default_timer()
            time_sim_mc[Ni] = tf_mc - ts_mc

       
        if "lin" in filters_to_run:    
            #Adaptive Q by linearization
            ts_lin = timeit.default_timer()
            for i in range(1, dim_t):
                t_span = (t[i-1], t[i])
                
                Q_lin = utils_br.get_Q_from_linearization(jac_p_func, 
                                                          x_post_lin[:, i-1], t_span, u[:, i-1], par_kf_fx.copy(), par_cov_fx)
                Q_lin = Q_lin + Q_diag_min #robustness/non-zero on diagonals
                kfc_lin.Q = Q_lin #assign to filter
                Q_lin_hist[:, i] = np.diag(Q_lin) #Save Q history
                fx_ukf_lin = lambda x, dt_kf: utils_br.integrate_ode(F, x, t_span, u[:, i-1], par_kf_fx)
                kfc_lin.predict(fx = fx_ukf_lin)
                
                hx_lin = lambda x_in: utils_br.hx(x_in, par_kf_hx.copy()) 
                kfc_lin.update(y[:, i], hx = hx_lin)
        
                x_post_lin[:, i] = kfc_lin.x
                P_diag_post_lin[:, i] = np.diag(kfc_lin.P)
                
            tf_lin = timeit.default_timer()
            time_sim_lin[Ni] = tf_lin - ts_lin
        
        
        if "qf" in filters_to_run:
            ts_qf = timeit.default_timer()
            for i in range(1, dim_t):
                t_span = (t[i-1], t[i])
                
                fx_ukf_qf = lambda x, dt_kf: utils_br.integrate_ode(F, x, t_span, u[:, i-1], par_kf_fx)
                kfc_qf.predict(fx = fx_ukf_qf)
                    
                hx_qf = lambda x_in: utils_br.hx(x_in, par_kf_hx.copy()) 
                kfc_qf.update(y[:, i], hx = hx_qf)
        
                x_post_qf[:, i] = kfc_qf.x
                P_diag_post_qf[:, i] = np.diag(kfc_qf.P)
            
            tf_qf = timeit.default_timer()
            time_sim_qf[Ni] = tf_qf - ts_qf

        
        #%% Compute performance index
        value_filter_not_run = 1 #same cost as OL response
        if "gut" in filters_to_run:
            j_valappil_gut[:, Ni] = utils_br.compute_performance_index_valappil(x_post_gut, 
                                                                             x_ol, 
                                                                             x_true, cost_func = cost_func_type)
        else:
            j_valappil_gut[:, Ni] = value_filter_not_run
        
        if "jgut" in filters_to_run:
            j_valappil_jgut[:, Ni] = utils_br.compute_performance_index_valappil(x_post_jgut[-dim_x:], 
                                                                             x_ol, 
                                                                             x_true, cost_func = cost_func_type)
            within_band_1s_jgut = utils_br.truth_within_estimate(x_true, 
                                                                 x_post_jgut[-dim_x:,:], 
                                                                 P_diag_post_jgut[-dim_x:, :], 
                                                                 sigma_multiplier = 1) #get a boolean vector
            consistency_1s_jgut[:, Ni] = np.sum(within_band_1s_jgut, axis = 1)/dim_t #% time within band
            within_band_2s_jgut = utils_br.truth_within_estimate(x_true, 
                                                                 x_post_jgut[-dim_x:,:], 
                                                                 P_diag_post_jgut[-dim_x:, :], 
                                                                 sigma_multiplier = 2) #get a boolean vector
            consistency_2s_jgut[:, Ni] = np.sum(within_band_2s_jgut, axis = 1)/dim_t #% time within band
            j_mean_jgut[:, Ni] = np.mean(x_post_jgut[-dim_x:, :] - x_true, axis = 1)
        else:
            j_valappil_jgut[:, Ni] = value_filter_not_run
        
        if "jgutw" in filters_to_run:
            j_valappil_jgutw[:, Ni] = utils_br.compute_performance_index_valappil(x_post_jgutw[-dim_x:], 
                                                                             x_ol, 
                                                                             x_true, cost_func = cost_func_type)
            within_band_1s_jgutw = utils_br.truth_within_estimate(x_true, 
                                                                 x_post_jgutw[-dim_x:,:], 
                                                                 P_diag_post_jgutw[-dim_x:, :], 
                                                                 sigma_multiplier = 1) #get a boolean vector
            consistency_1s_jgutw[:, Ni] = np.sum(within_band_1s_jgutw, axis = 1)/dim_t #% time within band
            within_band_2s_jgutw = utils_br.truth_within_estimate(x_true, 
                                                                 x_post_jgutw[-dim_x:,:], 
                                                                 P_diag_post_jgutw[-dim_x:, :], 
                                                                 sigma_multiplier = 2) #get a boolean vector
            consistency_2s_jgutw[:, Ni] = np.sum(within_band_2s_jgutw, axis = 1)/dim_t #% time within band
            j_mean_jgutw[:, Ni] = np.mean(x_post_jgutw[-dim_x:, :] - x_true, axis = 1)
        else:
            j_valappil_jgutw[:, Ni] = value_filter_not_run
        
        
        if "mc" in filters_to_run:
            j_valappil_mc[:, Ni] = utils_br.compute_performance_index_valappil(x_post_mc, 
                                                                             x_ol, 
                                                                             x_true, cost_func = cost_func_type)
        else:
            j_valappil_mc[:, Ni] = value_filter_not_run
        
        if "lin" in filters_to_run:
            j_valappil_lin[:, Ni] = utils_br.compute_performance_index_valappil(x_post_lin, 
                                                                             x_ol, 
                                                                             x_true, cost_func = cost_func_type)
            within_band_1s_lin = utils_br.truth_within_estimate(x_true, 
                                                                 x_post_lin, 
                                                                 P_diag_post_lin, 
                                                                 sigma_multiplier = 1) #get a boolean vector
            consistency_1s_lin[:, Ni] = np.sum(within_band_1s_lin, axis = 1)/dim_t #% time within band
            within_band_2s_lin = utils_br.truth_within_estimate(x_true, 
                                                                 x_post_lin, 
                                                                 P_diag_post_lin, 
                                                                 sigma_multiplier = 2) #get a boolean vector
            consistency_2s_lin[:, Ni] = np.sum(within_band_2s_lin, axis = 1)/dim_t #% time within band
            j_mean_lin[:, Ni] = np.mean(x_post_lin - x_true, axis = 1)
        else:
            j_valappil_lin[:, Ni] = value_filter_not_run
        
        if "qf" in filters_to_run:
            j_valappil_qf[:, Ni] = utils_br.compute_performance_index_valappil(x_post_qf, 
                                                                            x_ol, 
                                                                            x_true, cost_func = cost_func_type)
            within_band_1s_qf = utils_br.truth_within_estimate(x_true, 
                                                                 x_post_qf, 
                                                                 P_diag_post_qf, 
                                                                 sigma_multiplier = 1) #get a boolean vector
            consistency_1s_qf[:, Ni] = np.sum(within_band_1s_qf, axis = 1)/dim_t #% time within band
            within_band_2s_qf = utils_br.truth_within_estimate(x_true, 
                                                                 x_post_qf, 
                                                                 P_diag_post_qf, 
                                                                 sigma_multiplier = 2) #get a boolean vector
            consistency_2s_qf[:, Ni] = np.sum(within_band_2s_qf, axis = 1)/dim_t #% time within band
            j_mean_qf[:, Ni] = np.mean(x_post_qf - x_true, axis = 1)
        else:
            j_valappil_qf[:, Ni] = value_filter_not_run
        
        j_valappil_i = np.vstack((j_valappil_gut[:, Ni],
                                  j_valappil_jgut[:, Ni],
                                  j_valappil_jgutw[:, Ni],
                                  j_valappil_mc[:, Ni],
                                  j_valappil_lin[:, Ni],
                                  j_valappil_qf[:, Ni],
                                  )).T
        cost_S_lim = 1.5
        if cost_func_type == "valappil":
            if not all(j_valappil_i[2, :] < cost_S_lim):
                N_sim += 1 #all filters were not ok. Want to have "orginially specified" N_sim good simulations. The bad ones are kept and filtered out later
                print(f"One filter had cost index for the sugar > {cost_S_lim}. Adding one extra simulation")
                #Need to add more columns to the pre-allocated cost arrays if this happens
                j_valappil_gut = np.hstack((j_valappil_gut, np.zeros((dim_x,1))))
                j_valappil_jgut = np.hstack((j_valappil_jgut, np.zeros((dim_x,1))))
                j_valappil_jgutw = np.hstack((j_valappil_jgutw, np.zeros((dim_x,1))))
                j_valappil_mc = np.hstack((j_valappil_mc, np.zeros((dim_x,1))))
                j_valappil_lin = np.hstack((j_valappil_lin, np.zeros((dim_x,1))))
                j_valappil_qf = np.hstack((j_valappil_qf, np.zeros((dim_x,1))))
        
        Ni += 1
        rand_seed += 1
        time_iteration = time.time() - t_iter
        # print(f"Sugar_par: {par_true_fx['S_in']}")
        if (Ni%1 == 0): #print every 5th iteration                                                               
            print(f"Iter {Ni}/{N_sim} done. t_iter = {time.time()-t_iter: .2f} s and t_tot = {(time.time()-ts)/60: .1f} min")
    except KeyboardInterrupt as user_error:
        raise user_error
    except myExceptions.NegativeSigmaPoint as neg_sigma:
        print(neg_sigma)
        print(f"Re-doing iteration Ni = {Ni}")
        num_exceptions += 1
        rand_seed += 1
        continue
    except BaseException as e: #this is literally taking all possible exceptions
        print(e)
        print(f"Iter: {i}: Time spent, t_iter = {time.time()-ti: .2f} s ")
        num_exceptions += 1
        rand_seed += 1
        raise e
        continue
                

     
#%% Plot x, x_pred, y
ylabels = [ r"$V$ [L]", r"$X [g/L]$", r"$S [g/L]$", r"$CO_2 [*]$"]
# print(f"Repeated {N_sim} time(s). In every iteration, the number of model evaluations for computing noise statistics:\n",
#       f"Q by UT: {sigmas_fx_gut.shape[1]}\n",
#        f"Q by LHS: {N_lhs_dist}\n",
#       f"Q by MC: {N_mc_dist}\n"
#       # f"Q by MCm: {N_mcm_dist}\n",
#       # f"R by UT: {sigmas_hx.shape[1]}\n",
#       # f"R by LHS: {N_lhs_dist}\n",
#       # f"R by MC: {N_mc_dist}\n",
#       # f"R by MCm: {N_mcm_dist}\n"
#       )
# print("Median value of cost function is\n")
# for i in range(dim_x):
#     print(f"{ylabels[i]}: Q-GUT = {np.median(j_valappil_gut[i]): .3f}, \ "
#            + f"Q-LHS-{N_lhs_dist} = {np.median(j_valappil_lhs[i]): .3f}, \ "
#            + f"Q-LHSw-{N_lhs_dist} = {np.median(j_valappil_lhsw[i]): .3f}, \ "
#            + f"Q-LHSm-{N_lhs_dist} = {np.median(j_valappil_lhsm[i]): .3f}, \ "
#           + f"Q-GUTw = {np.median(j_valappil_gutw[i]): .3f}, \ "
#           + f"jGUT = {np.median(j_valappil_jgut[i]): .3f}, \ "
#           + f"Q-UT = {np.median(j_valappil_ut[i]): .3f}, \ "
#           + f"Q-UTw = {np.median(j_valappil_utw[i]): .3f}, \ "
#           + f"Q-MC-{N_mc_dist} = {np.median(j_valappil_mc[i]): .3f}, \ "
#           + f"Q-MCw-{N_mc_dist} = {np.median(j_valappil_mcw[i]): .3f}, \ "
#           + f"Q-MCm-{N_mc_dist} = {np.median(j_valappil_mcm[i]): .3f}, \ "
#           + f"Q-Lin = {np.median(j_valappil_lin[i]): .3f}, \ "
#           + f"Q-Linu = {np.median(j_valappil_linu[i]): .3f}, \ "
#           + f"Q-Lin_n = {np.median(j_valappil_lin_n[i]): .3f}, \ "
#           # + f"Q-MCm-{N_mcm_dist} = {np.median(j_valappil_mcm[i]): .3f}, \ "
#           + f"and Q-fixed = {np.median(j_valappil_qf[i]): .3f}")
plot_it = True
if plot_it:
    alpha_fill = .2
    kwargs_pred = {"linestyle": "dashed"}
    kwargs_gut = {"alpha": alpha_fill}
    kwargs_jgut = {"alpha": alpha_fill}
    kwargs_jgutw = {"alpha": alpha_fill}
    kwargs_mc = {"alpha": alpha_fill}
    kwargs_lin = {"alpha": alpha_fill}
    kwargs_qf = {"alpha": alpha_fill}
    #
    meas_idx = np.array([0, 1, 3])
    idx_y = 0
    filters_to_plot = [
        "gut",
        # "jgut",
        # "jgutw",
        # "gutw", 
        # "mc",
        # "lin",
        # "qf",
        "ol"
        ]
    # cmap_selected = plt.cm.get_cmap(cmap)
    # colors = cmap_selected(np.linspace(0, 1, 
    #                                     5)#number of filters to plot
    #                         )
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors)
    font = {'size': 16}
    matplotlib.rc('font', **font)
    fig1, ax1 = plt.subplots(dim_x, 1, sharex = True)
    
    plt_std_dev = True #plots 1 and 2 std dev around mean with shading
    for i in range(dim_x): #plot true states and ukf's estimates
        #plot true state
        ax1[i].plot(t, x_true[i, :], label = r"$x_{true}$")
        # ax1[i].plot(t, x_true[i, :], label = r"$x_{true}$", color = 'b')
    
        #plot measurements
        if i in meas_idx:
            ax1[i].scatter(t, y[idx_y, :], 
                            color = "m", 
                            # color = l[0].get_color(), 
                            s = 2,
                            alpha = .2,
                            marker = "o",
                            label = r"$y$")
            idx_y += 1
    
        # ylim_orig = ax1[i].get_ylim()
        
        #plot state predictions
        #Q_gut
        if "gut" in filters_to_plot:
            l_gut = ax1[i].plot(t, x_post_gut[i, :], label = r"$\hat{x}^+_{GenUT}$", **kwargs_pred)
        
        #UKF_jgut
        if "jgut" in filters_to_plot:
            l_jgut = ax1[i].plot(t, x_post_jgut[dim_par_fx + i, :], label = r"$\hat{x}^+_{JointGenUT}$", **kwargs_pred)
        
        #UKF_jgutw
        if "jgutw" in filters_to_plot:
            l_jgutw = ax1[i].plot(t, x_post_jgutw[dim_par_fx + i, :], label = r"$\hat{x}^+_{JointGenUTw}$", **kwargs_pred)
        
        #Q_mc
        if "mc" in filters_to_plot:
            l_mc = ax1[i].plot(t, x_post_mc[i, :], label = r"$\hat{x}^+_{MC}$", **kwargs_pred)
        
        #Q_lin
        if "lin" in filters_to_plot:
            l_lin = ax1[i].plot(t, x_post_lin[i, :], label = r"$\hat{x}^+_{Lin}$", **kwargs_pred)
        
        
        # #Q_qf
        if "qf" in filters_to_plot:
            l_qf = ax1[i].plot(t, x_post_qf[i, :], label = r"$\hat{x}^+_{Fixed}$", **kwargs_pred)
        
        if plt_std_dev:
            #GenUT
            if "gut" in filters_to_plot:
                kwargs_gut.update({"color": l_gut[0].get_color()})
                ax1[i].fill_between(t, 
                                    x_post_gut[i, :] + 2*np.sqrt(P_diag_post_gut[i,:]),
                                    x_post_gut[i, :] - 2*np.sqrt(P_diag_post_gut[i,:]),
                                    **kwargs_gut)
                ax1[i].fill_between(t, 
                                    x_post_gut[i, :] + 1*np.sqrt(P_diag_post_gut[i,:]),
                                    x_post_gut[i, :] - 1*np.sqrt(P_diag_post_gut[i,:]),
                                    **kwargs_gut)
            #JointGenUT
            if "jgut" in filters_to_plot:
                kwargs_jgut.update({"color": l_jgut[0].get_color()})
                ax1[i].fill_between(t, 
                                    x_post_jgut[dim_par_fx + i, :] + 2*np.sqrt(P_diag_post_jgut[dim_par_fx + i,:]),
                                    x_post_jgut[dim_par_fx + i, :] - 2*np.sqrt(P_diag_post_jgut[dim_par_fx + i,:]),
                                    **kwargs_jgut)
                ax1[i].fill_between(t, 
                                    x_post_jgut[dim_par_fx + i, :] + 1*np.sqrt(P_diag_post_jgut[dim_par_fx + i,:]),
                                    x_post_jgut[dim_par_fx + i, :] - 1*np.sqrt(P_diag_post_jgut[dim_par_fx + i,:]),
                                    **kwargs_jgut)
            
            #JointGenUTw
            if "jgutw" in filters_to_plot:
                kwargs_jgutw.update({"color": l_jgutw[0].get_color()})
                ax1[i].fill_between(t, 
                                    x_post_jgutw[dim_par_fx + i, :] + 2*np.sqrt(P_diag_post_jgutw[dim_par_fx + i,:]),
                                    x_post_jgutw[dim_par_fx + i, :] - 2*np.sqrt(P_diag_post_jgutw[dim_par_fx + i,:]),
                                    **kwargs_jgutw)
                ax1[i].fill_between(t, 
                                    x_post_jgutw[dim_par_fx + i, :] + 1*np.sqrt(P_diag_post_jgutw[dim_par_fx + i,:]),
                                    x_post_jgutw[dim_par_fx + i, :] - 1*np.sqrt(P_diag_post_jgutw[dim_par_fx + i,:]),
                                    **kwargs_jgutw)
            
            
            #MC
            if "mc" in filters_to_plot:
                kwargs_mc.update({"color": l_mc[0].get_color()})
                ax1[i].fill_between(t, 
                                    x_post_mc[i, :] + 2*np.sqrt(P_diag_post_mc[i,:]),
                                    x_post_mc[i, :] - 2*np.sqrt(P_diag_post_mc[i,:]),
                                    **kwargs_mc)
                ax1[i].fill_between(t, 
                                    x_post_mc[i, :] + 1*np.sqrt(P_diag_post_mc[i,:]),
                                    x_post_mc[i, :] - 1*np.sqrt(P_diag_post_mc[i,:]),
                                    **kwargs_mc)
            
            #Linearized
            if "lin" in filters_to_plot:
                kwargs_lin.update({"color": l_lin[0].get_color()})
                ax1[i].fill_between(t, 
                                    x_post_lin[i, :] + 2*np.sqrt(P_diag_post_lin[i,:]),
                                    x_post_lin[i, :] - 2*np.sqrt(P_diag_post_lin[i,:]),
                                    **kwargs_lin)
                ax1[i].fill_between(t, 
                                    x_post_lin[i, :] + 1*np.sqrt(P_diag_post_lin[i,:]),
                                    x_post_lin[i, :] - 1*np.sqrt(P_diag_post_lin[i,:]),
                                    **kwargs_lin)
            
            #Fixed
            if "qf" in filters_to_plot:
                kwargs_qf.update({"color": l_qf[0].get_color()})
                ax1[i].fill_between(t, 
                                    x_post_qf[i, :] + 2*np.sqrt(P_diag_post_qf[i,:]),
                                    x_post_qf[i, :] - 2*np.sqrt(P_diag_post_qf[i,:]),
                                    **kwargs_qf)
                ax1[i].fill_between(t, 
                                    x_post_qf[i, :] + 1*np.sqrt(P_diag_post_qf[i,:]),
                                    x_post_qf[i, :] - 1*np.sqrt(P_diag_post_qf[i,:]),
                                    **kwargs_qf)
        
        ylim_scaled = ax1[i].get_ylim()
        
        if "ol" in filters_to_plot:
            ax1[i].plot(t, x_ol[i, :], label = "OL", **kwargs_pred)
        # if ylim_orig[0] < -5:
        #     ax1[i].set_ylim((-5, ylim_orig[1]))
        ax1[i].set_ylabel(ylabels[i])
        # ax1[i].legend(frameon = False, ncol = 3) 
    ax1[-1].set_xlabel("Time [h]")
    # ax1[1].set_ylim((-2,30))
    # ax1[2].set_ylim((-2,30))
    # ax1[3].set_ylim((-2,30))
    ax1[0].legend(ncol = 2, frameon = False)   
    # ax1[0].legend(ncol = 2, frameon = True)   
     
#%% Plot Co2
if False:
    font = {'size': 16}
    matplotlib.rc('font', **font)
    fig_co2, ax_co2 = plt.subplots(1,1)
    ax_co2 = [ax_co2]
    plt_std_dev = True #plots 1 and 2 std dev around mean with shading
    #plot true state
    ax_co2[-1].plot(t, x_true[-1, :], label = r"$x_{true}$")
    # ax_co2[-1].plot(t, x_true[-1, :], label = r"$x_{true}$", color = 'b')

    #plot measurements
    ax_co2[-1].scatter(t, y[-1, :], 
                    color = "m", 
                    # color = l[0].get_color(), 
                    s = 2,
                    alpha = .2,
                    marker = "o",
                    label = r"$y$")

    # ylim_orig = ax_co2[-1].get_ylim()
    
    #plot state predictions
    #Q_gut
    if "gut" in filters_to_plot:
        l_gut = ax_co2[-1].plot(t, x_post_gut[-1, :], label = r"$\hat{x}^+_{GenUT}$", **kwargs_pred)
    
    #UKF_jgut
    if "jgut" in filters_to_plot:
        l_jgut = ax_co2[-1].plot(t, x_post_jgut[-1, :], label = r"$\hat{x}^+_{JointGenUT}$", **kwargs_pred)
    
    #Q_mc
    if "mc" in filters_to_plot:
        l_mc = ax_co2[-1].plot(t, x_post_mc[-1, :], label = r"$\hat{x}^+_{MC}$", **kwargs_pred)

    #Q_lin
    if "lin" in filters_to_plot:
        l_lin = ax_co2[-1].plot(t, x_post_lin[-1, :], label = r"$\hat{x}^+_{Lin}$", **kwargs_pred)
    
    # #Q_qf
    if "qf" in filters_to_plot:
        l_qf = ax_co2[-1].plot(t, x_post_qf[-1, :], label = r"$\hat{x}^+_{Fixed}$", **kwargs_pred)
    
    if plt_std_dev:
        #GenUT
        if "gut" in filters_to_plot:
            kwargs_gut.update({"color": l_gut[0].get_color()})
            ax_co2[-1].fill_between(t, 
                                x_post_gut[-1, :] + 2*np.sqrt(P_diag_post_gut[-1,:]),
                                x_post_gut[-1, :] - 2*np.sqrt(P_diag_post_gut[-1,:]),
                                **kwargs_gut)
            ax_co2[-1].fill_between(t, 
                                x_post_gut[-1, :] + 1*np.sqrt(P_diag_post_gut[-1,:]),
                                x_post_gut[-1, :] - 1*np.sqrt(P_diag_post_gut[-1,:]),
                                **kwargs_gut)
        #JointGenUT
        if "jgut" in filters_to_plot:
            kwargs_jgut.update({"color": l_jgut[0].get_color()})
            ax_co2[-1].fill_between(t, 
                                x_post_jgut[-1, :] + 2*np.sqrt(P_diag_post_jgut[-1,:]),
                                x_post_jgut[-1, :] - 2*np.sqrt(P_diag_post_jgut[-1,:]),
                                **kwargs_jgut)
            ax_co2[-1].fill_between(t, 
                                x_post_jgut[-1, :] + 1*np.sqrt(P_diag_post_jgut[-1,:]),
                                x_post_jgut[-1, :] - 1*np.sqrt(P_diag_post_jgut[-1,:]),
                                **kwargs_jgut)
        
        
        #MC
        if "mc" in filters_to_plot:
            kwargs_mc.update({"color": l_mc[0].get_color()})
            ax_co2[-1].fill_between(t, 
                                x_post_mc[-1, :] + 2*np.sqrt(P_diag_post_mc[-1,:]),
                                x_post_mc[-1, :] - 2*np.sqrt(P_diag_post_mc[-1,:]),
                                **kwargs_mc)
            ax_co2[-1].fill_between(t, 
                                x_post_mc[-1, :] + 1*np.sqrt(P_diag_post_mc[-1,:]),
                                x_post_mc[-1, :] - 1*np.sqrt(P_diag_post_mc[-1,:]),
                                **kwargs_mc)
        
       
        #Linearized
        if "lin" in filters_to_plot:
            kwargs_lin.update({"color": l_lin[0].get_color()})
            ax_co2[-1].fill_between(t, 
                                x_post_lin[-1, :] + 2*np.sqrt(P_diag_post_lin[-1,:]),
                                x_post_lin[-1, :] - 2*np.sqrt(P_diag_post_lin[-1,:]),
                                **kwargs_lin)
            ax_co2[-1].fill_between(t, 
                                x_post_lin[-1, :] + 1*np.sqrt(P_diag_post_lin[-1,:]),
                                x_post_lin[-1, :] - 1*np.sqrt(P_diag_post_lin[-1,:]),
                                **kwargs_lin)
        
        #Fixed
        if "qf" in filters_to_plot:
            kwargs_qf.update({"color": l_qf[0].get_color()})
            ax_co2[-1].fill_between(t, 
                                x_post_qf[-1, :] + 2*np.sqrt(P_diag_post_qf[-1,:]),
                                x_post_qf[-1, :] - 2*np.sqrt(P_diag_post_qf[-1,:]),
                                **kwargs_qf)
            ax_co2[-1].fill_between(t, 
                                x_post_qf[-1, :] + 1*np.sqrt(P_diag_post_qf[-1,:]),
                                x_post_qf[-1, :] - 1*np.sqrt(P_diag_post_qf[-1,:]),
                                **kwargs_qf)
    
    ylim_scaled = ax_co2[-1].get_ylim()
    
    # if "ol" in filters_to_plot:
    #     ax_co2[-1].plot(t, x_ol[-1, :], label = "OL", **kwargs_pred)
    # if ylim_orig[0] < -5:
    #     ax_co2[-1].set_ylim((-5, ylim_orig[1]))
    ax_co2[-1].set_ylabel(ylabels[-1])
    # ax_co2[-1].legend(frameon = False, ncol = 3) 
    ax_co2[-1].set_xlabel("Time [h]")
    # ax_co2[1].set_ylim((-2,30))
    # ax_co2[2].set_ylim((-2,30))
    # ax_co2[3].set_ylim((-2,30))
    ax_co2[0].legend(ncol = 2, frameon = True) 
    plt.tight_layout()
    # ax_co2[0].legend(ncol = 2, frameon = True)   
     
    #%% Plot w_mean-history
    w_labels = [r"$w_{GUT}$", r"$w_{JGUT}$", r"$w_{MC}$"]#, r"$F_{in}$ [*]"]
    y_labels = [ r"$V$ [L]", r"$X [g/L]$", r"$S [g/L]$", r"$CO_2 [*]$"]#
    fig_w, ax_w = plt.subplots(dim_x, 1 ,sharex = True)
    if dim_x == 1:
        ax_w = [ax_w]
    for i in range(dim_x):
        ax_w[i].plot(t, w_gut_hist[i, :], label = w_labels[0])
        ax_w[i].plot(t, w_jgutw_hist[dim_par_fx + i, :], label = w_labels[1])
        ax_w[i].plot(t, w_mc_hist[i, :], label = w_labels[2])
        ax_w[i].set_ylabel(y_labels[i])
    ax_w[-1].set_xlabel("Time [h]")
    ax_w[0].legend()
    
    #%% Plot Q-history
    q_labels = [r"$Q_{GUT}$", r"$Q_{MC}$", r"$Q_{Lin}$", r"$Q_{JGUTw}$"]
    # y_labels = [ r"$V^2$ [L^2]", r"$X^2 [(g/L)^2]$", r"$S^2 [(g/L)^2]$", r"$(CO_2)^2 [*]$"]#
    y_labels = [ r"$V^2 [*]$", r"$X^2 [*]$", r"$S^2 [*]$", r"$(CO_2)^2 [*]$"]#
    kwargs_qplot = {"linestyle": "dashed"}
    matplotlib.rc('font', **font)
    fig_q, ax_q = plt.subplots(dim_x, 1 ,sharex = True)
    if dim_x == 1:
        ax_q = [ax_q]
    for i in range(dim_x):
        # ax_q[i].plot(t, Q_gut_hist[i, :], label = q_labels[0], **kwargs_qplot)
        ax_q[i].plot(t, Q_jgutw_hist[dim_par_fx + i, :], label = q_labels[3], **kwargs_qplot)
        # ax_q[i].plot(t, Q_mc_hist[i, :], label = q_labels[1], **kwargs_qplot)
        # ax_q[i].plot(t, Q_lin_hist[i, :], label = q_labels[2], **kwargs_qplot)
        # ax_q[i].plot(t, Q_qf_hist[i, :], label = q_labels[3])
        ax_q[i].set_ylabel(y_labels[i])
        ax_q[i].legend(ncol = 3)
    ax_q[-1].set_xlabel("Time [h]")
    # ax_q[0].legend(ncol = 3)
    plt.tight_layout()
    
    
    #%% Plot u
    # u_labels = [r"$F_{in}$ [L/h]"]#, r"$Gn_{in}$ []", r"$F_{in}$ [*]"]
    # fig_u, ax_u = plt.subplots(dim_u, 1 ,sharex = True)
    # if dim_u == 1:
    #     ax_u = [ax_u]
    # for i in range(dim_u):
    #     ax_u[i].step(t, u[i, :])
    #     ax_u[i].set_ylabel(u_labels[i])
    # ax_u[-1].set_xlabel("Time [h]")
    # # ax_u[0].legend()
    
    #%% Plot par_history_fx
    if False:
        kwargs_par_hist = {"corner": True}
        
        df_par_fx_hist = pd.DataFrame(data = par_history_fx.T, columns = list(par_true_fx.keys()))
        sns_grid_par_fx = sns.pairplot(df_par_fx_hist, **kwargs_par_hist)
        sns_grid_par_fx.fig.suptitle(r"History of $\theta_{fx}$") # y= some height>1
    
    
    
    # df_par_hx_hist = pd.DataFrame(data = par_history_hx.T, columns = list(par_true_hx.keys()))
    # sns_grid_par_hx = sns.pairplot(df_par_hx_hist, **kwargs_par_hist)
#%% Violin plot of cost function
if N_sim >= 5: #only plot this if we have done some iterations
    fig_v, ax_v = plt.subplots(dim_x,1, sharex = True)
    # labels_violin = ["GenUT", "Lin", f"MC-{N_mc_dist}", "Fixed"]#, "GenUTw", f"MCw-{N_mc_dist}"]
    labels_violin = ["GenUT", "Lin", f"MC-{N_mc_dist}", "Fixed", "JointGenUT", "JGenUTw"]
    
    # labels_violin = ["GenUT", "LHS", "MC", "Fixed"]
    def set_axis_style(ax, labels):
        ax.xaxis.set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        # ax.set_xlabel(r'Method for tuning $Q_k, R_k$')
    for i in range(dim_x):
        # data = np.vstack([j_valappil_gut[i], j_valappil_lin[i], j_valappil_mc[i], j_valappil_qf[i]]).T
        data = np.vstack([j_valappil_gut[i], j_valappil_lin[i], j_valappil_mc[i], j_valappil_qf[i], j_valappil_jgut[i], j_valappil_jgutw[i]]).T
        # print(f"---cost of x_{i}---\n",
        #       f"mean = {data.mean(axis = 0)}\n",
        #       f"std = {data.std(axis = 0)}")
        ax_v[i].violinplot(data)#, j_valappil_qf])
        ax_v[i].set_ylabel(fr"Cost $x_{i+1}$ [-]")
    set_axis_style(ax_v[i], labels_violin)
    ax_v[-1].set_xlabel(r'Method for tuning $Q_k, R_k$')
    fig_v.suptitle(f"Cost function distribution for N = {N_sim} iterations")

    #%% Violin plot of cost function, filtered
    df_cost_rmse_gut = pd.DataFrame(data = j_valappil_gut.T.copy(), columns = ["V", "X", "S", "CO2"])
    df_cost_rmse_jgut = pd.DataFrame(data = j_valappil_jgut.T.copy(), columns = ["V", "X", "S", "CO2"])
    df_cost_rmse_mc = pd.DataFrame(data = j_valappil_mc.T.copy(), columns = ["V", "X", "S", "CO2"])
    df_cost_rmse_jgutw = pd.DataFrame(data = j_valappil_jgutw.T.copy(), columns = ["V", "X", "S", "CO2"])
    df_cost_rmse_mc = pd.DataFrame(data = j_valappil_mc.T.copy(), columns = ["V", "X", "S", "CO2"])
    df_cost_rmse_lin = pd.DataFrame(data = j_valappil_lin.T.copy(), columns = ["V", "X", "S", "CO2"])
    df_cost_rmse_qf = pd.DataFrame(data = j_valappil_qf.T.copy(), columns = ["V", "X", "S", "CO2"])
    
    # df_cost_list = [df_cost_gut, df_cost_lin, df_cost_mc, df_cost_qf, df_cost_jgut, df_cost_gutw, df_cost_mcw, df_cost_lhs, df_cost_ut, df_cost_utw, df_cost_linu, df_cost_mcm, df_cost_lhsm, df_cost_lhsw]
    df_cost_rmse_list = [df_cost_rmse_gut, df_cost_rmse_lin, df_cost_rmse_qf, df_cost_rmse_jgut, df_cost_rmse_mc, df_cost_rmse_jgutw]
    labels_violin = ["gut", "lin", "qf", "jgut", "mc", "jgutw"]
    # df_cost_rmse_var = [[pd.concat(df_cost_rmse_list.iloc[:, i], axis = 1)]]
    df_cost_rmse_all = pd.concat(dict(gut=df_cost_rmse_gut, 
                                 lin = df_cost_rmse_lin,
                                  mc = df_cost_rmse_mc,
                                 qf = df_cost_rmse_qf,
                                 jgut = df_cost_rmse_jgut,
                                 jgutw = df_cost_rmse_jgutw
                                 ), axis = 1)
    # df_cost_all.iloc[:,df_cost_all.columns.get_level_values(1)=="V"] #selects all "V"
    
    cost_filtered = [[] for i in range(len(df_cost_rmse_list))]
    diverged_sim_list = [[] for i in range(len(df_cost_rmse_list))]
    
    if cost_func_type == "Valappil":
        i = 0
        for df_cost_rmse in df_cost_rmse_list:
            diverged_sim_list[i] = (df_cost_rmse["S"] > cost_S_lim).sum()
            cost_filtered[i] = df_cost_rmse[df_cost_rmse["S"] <= cost_S_lim]
            i += 1
    else:
        cost_filtered = [df_cost_rmse for df_cost_rmse in df_cost_rmse_list] #no filtering
    
    # #Plot number of diverged simulations
    # fig_div, ax_div = plt.subplots(1,1)
    # ax_div.bar(labels_violin, diverged_sim_list)
    # ax_div.set_ylabel(f"# divergences for {N_sim} simulations")
    
    #Plot violinplot of filtered cost function
    fig_h2, ax_h2 = plt.subplots(dim_x,1, sharex = True)
    for i in range(dim_x):
        d2 = [pd.DataFrame(data = cost_filtered[j].iloc[:, i].to_numpy(), columns = [labels_violin[j]]) for j in range(len(cost_filtered))]
        data = pd.concat(d2, join = "outer", axis = 1) #concat, fill with Nans if missing values
        # print(f"---cost of x_{i}---\n",
        #       f"mean:\n{data.mean(axis = 0)}\n\n",
        #       f"std_dev: \n{data.std(axis = 0)}")
        sns.violinplot(data = data, ax = ax_h2[i], bw = .2)#, j_h2alappil_qf])
        ax_h2[i].set_ylabel(r"$J_{\sigma}$" + fr" $(x_{i+1})$ [-]")
    ax_h2[-1].set_xlabel(r'Method for tuning $Q_k, R_k$')
    fig_h2.suptitle(f"RMSE for N = {N_sim} iterations")    
    
    #%% Mean cost, violinplot
    
    df_cost_mean_gut = pd.DataFrame(data = j_mean_gut.T.copy(), columns = ["V", "X", "S", "CO2"])
    df_cost_mean_jgut = pd.DataFrame(data = j_mean_jgut.T.copy(), columns = ["V", "X", "S", "CO2"])
    df_cost_mean_jgutw = pd.DataFrame(data = j_mean_jgutw.T.copy(), columns = ["V", "X", "S", "CO2"])
    df_cost_mean_mc = pd.DataFrame(data = j_mean_mc.T.copy(), columns = ["V", "X", "S", "CO2"])
    df_cost_mean_lin = pd.DataFrame(data = j_mean_lin.T.copy(), columns = ["V", "X", "S", "CO2"])
    df_cost_mean_qf = pd.DataFrame(data = j_mean_qf.T.copy(), columns = ["V", "X", "S", "CO2"])
    
    df_cost_mean_list = [df_cost_mean_lin, df_cost_mean_qf, df_cost_mean_jgut, df_cost_mean_jgutw, df_cost_mean_mc]
    labels_violin = ["lin", "qf", "jgut", "jgutw", "mc"]
    # df_cost_mean_var = [[pd.concat(df_cost_mean_list.iloc[:, i], axis = 1)]]
    df_cost_mean_all = pd.concat(dict(gut=df_cost_mean_gut, 
                                 lin = df_cost_mean_lin,
                                  mc = df_cost_mean_mc,
                                 qf = df_cost_mean_qf,
                                 jgut = df_cost_mean_jgut,
                                 jgutw = df_cost_mean_jgutw
                                 ), axis = 1)
    
    #Plot violinplot of filtered cost function
    fig_jm, ax_jm = plt.subplots(dim_x,1, sharex = True)
    for i in range(dim_x):
        d2 = [pd.DataFrame(data = df_cost_mean_list[j].iloc[:, i].to_numpy(), columns = [labels_violin[j]]) for j in range(len(df_cost_mean_list))]
        data = pd.concat(d2, join = "outer", axis = 1) #concat, fill with Nans if missing values
        # print(f"---cost of x_{i}---\n",
        #       f"mean:\n{data.mean(axis = 0)}\n\n",
        #       f"std_dev: \n{data.std(axis = 0)}")
        sns.violinplot(data = data, ax = ax_jm[i], 
                       # bw = .2,
                       inner = "box")#, j_h2alappil_qf])
        # sns.swarmplot(data = data, ax = ax_jm[i], color = "black")
        sns.stripplot(data = data, ax = ax_jm[i], color = "black")
        ax_jm[i].set_ylabel(r"$J_{\mathbb{E}}$" + fr" $(x_{i+1})$ [-]")
    ax_jm[-1].set_xlabel(r'Method for tuning $Q_k, R_k$')
    fig_jm.suptitle(r"$\hat{x}^{+} - x_{true}$ " + f" for N = {N_sim} iterations")
    
    #%%Barplot of mean
    fig_bm, ax_bm = plt.subplots(dim_x, 1, sharex=True)
    df_mean_all = df_cost_rmse_all.mean()
    for i in range(dim_x):
        var_name = df_mean_all.index.get_level_values(1).unique()[i]
        df_mean_i = df_mean_all[df_mean_all.index.get_level_values(1)==var_name]
        df_mean_i.index = df_mean_i.index.droplevel(1)
        df_mean_i = df_mean_i.drop("qf")
        # sns.barplot(ax = ax_bm[i], data = df_mean_i)
        sns.barplot(x=df_mean_i.index, y=df_mean_i.values, ax = ax_bm[i], dodge = False)
        ax_bm[i].set_ylabel(r"$\mu_{cost}$" + fr"$(x_{i})$")
        
    
    #%%Barplot of std_dev
    fig_bstd, ax_bstd = plt.subplots(dim_x, 1, sharex=True)
    df_std_all = df_cost_rmse_all.std()
    for i in range(dim_x):
        var_name = df_std_all.index.get_level_values(1).unique()[i]
        df_std_i = df_std_all[df_std_all.index.get_level_values(1)==var_name]
        df_std_i.index = df_std_i.index.droplevel(1)
        df_std_i = df_std_i.drop("qf")
        # sns.barplot(ax = ax_bstd[i], data = df_std_i)
        sns.barplot(x=df_std_i.index, y=df_std_i.values, ax = ax_bstd[i], dodge = False)
        ax_bstd[i].set_ylabel(r"$\sigma_{cost}$" + fr"$(x_{i})$")
        
    
    #%%Barplot of median
    fig_bmed, ax_bmed = plt.subplots(dim_x, 1, sharex=True)
    df_med_all = df_cost_rmse_all.std()
    for i in range(dim_x):
        var_name = df_med_all.index.get_level_values(1).unique()[i]
        df_med_i = df_med_all[df_med_all.index.get_level_values(1)==var_name]
        df_med_i.index = df_med_i.index.droplevel(1)
        df_med_i = df_med_i.drop("qf")
        # sns.barplot(ax = ax_bmed[i], data = df_med_i)
        sns.barplot(x=df_med_i.index, y=df_med_i.values, ax = ax_bmed[i], dodge = False)
        ax_bmed[i].set_ylabel(r"$median_{cost}$" + fr"$(x_{i})$")
        
    
    
    #%%Barplot for all simulations (filtered on all(S < S_lim))
    #Plot the sum of filtered cost function
    
    # #filter df first
    # df_S = df_cost_all.iloc[:,df_cost_all.columns.get_level_values(1)=="S"] #all S values in this df
    # df_cost_all = df_cost_all.iloc[(df_S < cost_S_lim).all(axis=1).values, :] #filter if all are true
    # # fig_csum_all, ax_csum_all = plt.subplots(dim_x,1, sharex = True)#create figure
    # # for i in range(dim_x):
    # #     df_xvar = df_cost_all.iloc[:,df_cost_all.columns.get_level_values(1)==df_cost_gut.columns[i]] #get variable (e.g. CO2 for all methods)
    # #     df_xvar.columns = df_xvar.columns.droplevel(1) #convert from multilevel columns to single level column
    # #     ax_csum_all[i].bar(list(df_xvar.columns), df_xvar.sum())
    # #     ax_csum_all[i].set_ylabel(fr"$\Sigma J(x_{i+1})$ [-]")
    # # ax_csum_all[-1].set_xlabel(r'Method for tuning $Q_k, R_k$')
    # # fig_csum_all.suptitle(f"Cost function sum for N = {df_cost_all.shape[0]} iterations")    
    
    
    # #Plot relative performance of the methods compared to GenUT tuning
    # matplotlib.rc('font', **font)
    # fig_csumrel_all, ax_csumrel_all = plt.subplots(dim_x,1, sharex = False)
    # palette ={"Positive": "C0", 
    #           "Negative": "C1"}
    # for i in range(dim_x):
    #     #get variable (e.g. CO2) for all methods and convert from multicolumn=>singlecolumn
    #     df_xvar = df_cost_all.iloc[:,df_cost_all.columns.get_level_values(1)==df_cost_gut.columns[i]]
    #     df_xvar.columns = df_xvar.columns.droplevel(1)
    #     #make values compared to GenUT and plot in %
    #     df_xvar_rel = (df_xvar.sum()/df_cost_all["gut"].iloc[:, i].sum()-1)*100
    #     #Make different colors if J_rel >0 and J_rel<0
    #     df_xvar_rel = pd.DataFrame(data=df_xvar_rel, columns=["Value"])
    #     df_xvar_rel["method"] = df_xvar_rel.index
    #     # df_xvar_rel = df_xvar_rel.drop(index = ["gut", "qf"])
    #     df_xvar_rel = df_xvar_rel.drop(index = ["gut"])
    #     df_xvar_rel["Increase"] = "Positive"
    #     df_xvar_rel.loc[df_xvar_rel["Value"] < 0, "Increase"] = "Negative"
    #     sns.barplot(x="method", y = "Value", data = df_xvar_rel, hue = "Increase", ax = ax_csumrel_all[i], dodge = False, palette = palette)
    #     ax_csumrel_all[i].set_ylabel(fr"$\Sigma J(x_{i+1})$/" + r"$\Sigma J^{GenUT}($" +fr"$x_{i+1})$[%]")
    #     ylim = ax_csumrel_all[i].get_ylim()
    #     # ax_csumrel_all[i].set_ylim((np.min([ylim[0], -5]), np.min([ylim[1], 5])))
    #     xlim = ax_csumrel_all[i].get_xlim()
    #     ax_csumrel_all[i].plot(list(xlim), [0, 0], 'k')
    #     ax_csumrel_all[i].set_xlim(xlim)
    #     if not i == 0:
    #         ax_csumrel_all[i].legend().remove()
    #     if not i == dim_x-1:
    #         ax_csumrel_all[i].set_xlabel("")
        
    # ax_csumrel_all[-1].set_xlabel(r'Method for tuning $Q_k, R_k$')
    # fig_csumrel_all.suptitle(f"Cost function sum/GenUT for N = {df_cost_all.shape[0]} iterations")    
    # plt.tight_layout()
    
    # print(df_cost_all.loc[:,["ut", "gut", "lin", "linu", "utw", "gutw"]].sum())

#%% Barplot baseline
    #Plot the sum of filtered cost function
    
    #filter df first
    if cost_func_type == "valappil":
        df_S = df_cost_rmse_all.iloc[:,df_cost_rmse_all.columns.get_level_values(1)=="S"] #all S values in this df
        df_cost_rmse_all = df_cost_rmse_all.iloc[(df_S < cost_S_lim).all(axis=1).values, :] #filter if all are true
        
    #select which filter to plot    
    df_cost_rmse_all2 = df_cost_rmse_all.iloc[:, df_cost_rmse_all.columns.get_level_values(0)=="jgut"]
    df_cost_rmse_all2 = pd.concat([df_cost_rmse_all2, 
                              df_cost_rmse_all.iloc[:, df_cost_rmse_all.columns.get_level_values(0)=="gut"],
                              df_cost_rmse_all.iloc[:, df_cost_rmse_all.columns.get_level_values(0)=="jgutw"],
                              df_cost_rmse_all.iloc[:, df_cost_rmse_all.columns.get_level_values(0)=="mc"],
                              # df_cost_rmse_all.iloc[:, df_cost_rmse_all.columns.get_level_values(0)=="qf"],
                              df_cost_rmse_all.iloc[:, df_cost_rmse_all.columns.get_level_values(0)=="lin"]
                              ], axis = 1)
    # df_cost_rmse_all = df_cost_rmse_all2
    # fig_csum_all, ax_csum_all = plt.subplots(dim_x,1, sharex = True)#create figure
    # for i in range(dim_x):
    #     df_xvar = df_cost_rmse_all.iloc[:,df_cost_rmse_all.columns.get_level_values(1)==df_cost_gut.columns[i]] #get variable (e.g. CO2 for all methods)
    #     df_xvar.columns = df_xvar.columns.droplevel(1) #convert from multilevel columns to single level column
    #     ax_csum_all[i].bar(list(df_xvar.columns), df_xvar.sum())
    #     ax_csum_all[i].set_ylabel(fr"$\Sigma J(x_{i+1})$ [-]")
    # ax_csum_all[-1].set_xlabel(r'Method for tuning $Q_k, R_k$')
    # fig_csum_all.suptitle(f"Cost function sum for N = {df_cost_rmse_all.shape[0]} iterations")    
    
    
    #Plot relative performance of the methods compared to GenUT tuning
    matplotlib.rc('font', **font)
    fig_csumrel_all, ax_csumrel_all = plt.subplots(dim_x,1, sharex = False)
    palette ={"Positive": "C0", 
              "Negative": "C1"}
    for i in range(dim_x):
        #get variable (e.g. CO2) for all methods and convert from multicolumn=>singlecolumn
        df_xvar = df_cost_rmse_all2.iloc[:,df_cost_rmse_all2.columns.get_level_values(1)==df_cost_rmse_lin.columns[i]]
        df_xvar.columns = df_xvar.columns.droplevel(1)
        #make values compared to GenUT and plot in %
        df_xvar_rel = (df_xvar.sum()/df_cost_rmse_all2["lin"].iloc[:, i].sum()-1)*100
        #Make different colors if J_rel >0 and J_rel<0
        df_xvar_rel = pd.DataFrame(data=df_xvar_rel, columns=["Value"])
        df_xvar_rel["method"] = df_xvar_rel.index
        # df_xvar_rel = df_xvar_rel.drop(index = ["gut", "qf"])
        df_xvar_rel = df_xvar_rel.drop(index = ["lin"])
        df_xvar_rel["Increase"] = "Positive"
        df_xvar_rel.loc[df_xvar_rel["Value"] < 0, "Increase"] = "Negative"
        sns.barplot(x="method", y = "Value", data = df_xvar_rel, hue = "Increase", ax = ax_csumrel_all[i], dodge = False, palette = palette)
        ax_csumrel_all[i].set_ylabel(fr"$\Sigma J(x_{i+1})$/" + r"$\Sigma J^{Lin}($" +fr"$x_{i+1})$[%]")
        ylim = ax_csumrel_all[i].get_ylim()
        # ax_csumrel_all[i].set_ylim((np.min([ylim[0], -5]), np.min([ylim[1], 5])))
        xlim = ax_csumrel_all[i].get_xlim()
        ax_csumrel_all[i].plot(list(xlim), [0, 0], 'k')
        ax_csumrel_all[i].set_xlim(xlim)
        if not i == 0:
            ax_csumrel_all[i].legend().remove()
        if not i == dim_x-1:
            ax_csumrel_all[i].set_xlabel("")
        
    ax_csumrel_all[-1].set_xlabel(r'Method for tuning $Q_k, R_k$')
    fig_csumrel_all.suptitle(f"Cost function sum/Lin for N = {df_cost_rmse_all2.shape[0]} iterations")    
    plt.tight_layout()
    
    # print(df_cost_all2.loc[:,["ut", "gut", "lin", "linu", "utw", "gutw"]].sum())

# #%%
# if N_sim >= 5: #only plot this if we have done some iterations
#     fig_v, ax_v = plt.subplots(dim_x,1, sharex = True)
#     labels_violin = ["GenUT", "Lin", "MC", "Fixed"]
#     # labels_violin = ["GenUT", "LHS", "MC", "Fixed"]
#     def set_axis_style(ax, labels):
#         ax.xaxis.set_tick_params(direction='out')
#         ax.xaxis.set_ticks_position('bottom')
#         ax.set_xticks(np.arange(1, len(labels) + 1))
#         ax.set_xticklabels(labels)
#         ax.set_xlim(0.25, len(labels) + 0.75)
#         # ax.set_xlabel(r'Method for tuning $Q_k, R_k$')
#     for i in range(dim_x):
#         data = np.vstack([j_valappil_gut[i], j_valappil_lin[i], j_valappil_mc[i], j_valappil_qf[i]]).T
#         # data = np.vstack([j_valappil[i], j_valappil_lhs[i], j_valappil_mc[i], j_valappil_qf[i]]).T
#         print("---cost of x_{i}---\n",
#               f"mean = {data.mean(axis = 0)}\n",
#               f"std = {data.std(axis = 0)}")
#         ax_v[i].violinplot(data)#, j_valappil_qf])
#         ax_v[i].set_ylabel(fr"Cost $x_{i+1}$ [-]")
#     set_axis_style(ax_v[i], labels_violin)
#     ax_v[-1].set_xlabel(r'Method for tuning $Q_k, R_k$')
#     fig_v.suptitle(f"Cost function distribution for N = {N_sim} iterations")

#%% Violin plot of cost function v2
# if N_sim >= 5: #only plot this if we have done some iterations
#     fig_v2, ax_v2 = plt.subplots(1,1)
#     import matplotlib.patches as patches
#     # labels_violin = ["UT", "LHS"]
#     labels_violin = ["GenUT", "Lin", "MC", "Fixed"]
#     legend_elements = []
#     for i in range(dim_x):
#         # data = np.vstack([j_valappil[i], j_valappil_lhs[i]]).T
#         data = np.vstack([j_valappil_gut[i], j_valappil_lin[i], j_valappil_mc[i], j_valappil_qf[i]]).T
#         print("---cost of x_{i}---\n",
#               f"mean = {data.mean(axis = 0)}\n",
#               f"std = {data.std(axis = 0)}")
#         l = ax_v2.violinplot(data)#, label = rf"$x_{i}$")#, j_valappil_qf])
#     #     patch= ax_v2.add_patch(patches.Rectangle((.5, .5), 0.5, 0.5,
#     # alpha=0.1,facecolor=l.get_fac,label='Label'))
#         legend_elements.append(patches.Patch(color=l["bodies"][0].get_facecolor(),label=rf"$x_{i}$"))
#     set_axis_style(ax_v2, labels_violin)
#     ax_v2.set_ylabel("Cost")
#     ax_v2.set_xlabel(r'Method for tuning $Q_k, R_k$')
#     ax_v2.legend(handles = legend_elements, frameon = False, loc = "center right")
#     fig_v2.suptitle(f"Cost function distribution for N = {N_sim} iterations")

# print(f"\n Failed {num_exceptions} times. Successful {N_sim} times. Simulation time: {time.time()-ts: .0f} s = {(time.time()-ts)/60: .1f} min")

#%% Consistency plot
    
    df_conc_jgut = pd.DataFrame(data = consistency_1s_jgut.T, columns = x_var)
    df_conc_jgut["sigma"] = r"$x \in (\hat{x} \pm \hat{\sigma})$"
    df_conc_jgut2 = pd.DataFrame(data = consistency_2s_jgut.T, columns = x_var)
    df_conc_jgut2["sigma"] = r"$x \in (\hat{x} \pm 2\hat{\sigma})$"
    df_conc_jgut = pd.concat([df_conc_jgut, df_conc_jgut2], ignore_index = True)
    df_conc_jgut["Filter"] = "jgut"
    del df_conc_jgut2
    
    df_conc_jgutw = pd.DataFrame(data = consistency_1s_jgutw.T, columns = x_var)
    df_conc_jgutw["sigma"] = r"$x \in (\hat{x} \pm \hat{\sigma})$"
    df_conc_jgutw2 = pd.DataFrame(data = consistency_2s_jgutw.T, columns = x_var)
    df_conc_jgutw2["sigma"] = r"$x \in (\hat{x} \pm 2\hat{\sigma})$"
    df_conc_jgutw = pd.concat([df_conc_jgutw, df_conc_jgutw2], ignore_index = True)
    df_conc_jgutw["Filter"] = "jgutw"
    del df_conc_jgutw2
    
    
    df_conc_gut = pd.DataFrame(data = consistency_1s_gut.T, columns = x_var)
    df_conc_gut["sigma"] = r"$x \in (\hat{x} \pm \hat{\sigma})$"
    df_conc_gut2 = pd.DataFrame(data = consistency_2s_gut.T, columns = x_var)
    df_conc_gut2["sigma"] = r"$x \in (\hat{x} \pm 2\hat{\sigma})$"
    df_conc_gut = pd.concat([df_conc_gut, df_conc_gut2], ignore_index = True)
    df_conc_gut["Filter"] = "gut"
    del df_conc_gut2
    
    df_conc_lin = pd.DataFrame(data = consistency_1s_lin.T, columns = x_var)
    df_conc_lin["sigma"] = r"$x \in (\hat{x} \pm \hat{\sigma})$"
    df_conc_lin2 = pd.DataFrame(data = consistency_2s_lin.T, columns = x_var)
    df_conc_lin2["sigma"] = r"$x \in (\hat{x} \pm 2\hat{\sigma})$"
    df_conc_lin = pd.concat([df_conc_lin, df_conc_lin2], ignore_index = True)
    df_conc_lin["Filter"] = "lin"
    del df_conc_lin2
    
    df_conc_qf = pd.DataFrame(data = consistency_1s_qf.T, columns = x_var)
    df_conc_qf["sigma"] = r"$x \in (\hat{x} \pm \hat{\sigma})$"
    df_conc_qf2 = pd.DataFrame(data = consistency_2s_qf.T, columns = x_var)
    df_conc_qf2["sigma"] = r"$x \in (\hat{x} \pm 2\hat{\sigma})$"
    df_conc_qf = pd.concat([df_conc_qf, df_conc_qf2], ignore_index = True)
    df_conc_qf["Filter"] = "Fixed"
    del df_conc_qf2
    
    df_conc = pd.concat([df_conc_jgut, 
                         df_conc_jgutw, 
                         df_conc_gut, 
                         df_conc_lin, 
                         df_conc_qf], ignore_index = True)
    
    del df_conc_jgut, df_conc_jgutw, df_conc_gut, df_conc_lin
    
    fig_con, ax_con = plt.subplots(dim_x, 1, sharex = True)
    fig_con.suptitle("Consistency of UKF")
    for i in range(dim_x):
        sns.violinplot(x = "Filter", y = x_var[i], data = df_conc, hue = "sigma", split = True, ax = ax_con[i], alpha = .2, legend = False, inner = "stick")
        plt.setp(ax_con[i].collections, alpha=.3)
        # ax_con[i].legend().remove()
        # sns.stripplot(x = "Filter", y = x_var[i], data = df_conc, hue = "sigma", dodge = True, ax = ax_con[i])
        # handles = ax_con[i].legend().legendHandles
        handles_leg, labels_leg = ax_con[i].get_legend_handles_labels()
        # h2 = handles[-2:]
        if i==0: 
            ax_con[i].legend(handles_leg[-2:], labels_leg[-2:], frameon = False) 
        else:
            ax_con[i].legend().remove()
        if not i == dim_x:
            ax_con[i].set_xlabel("")
        # ax_con[i].legend().remove()
        # sns.swarmplot(x = "Filter", y = x_var[i], data = df_conc, hue = "sigma", dodge = True, ax = ax_con[i])
    
    #%% Simultation time plot
    
    df_t_jgut = pd.DataFrame(data = time_sim_jgut.T, columns = ["Run time [s]"])
    df_t_jgut["Filter"] = "jgut"
    df_t_jgut["Option"] = "Option 1"
    
    df_t_jgutw = pd.DataFrame(data = time_sim_jgutw.T, columns = ["Run time [s]"])
    df_t_jgutw["Filter"] = "jgutw"
    df_t_jgutw["Option"] = "Option 1"
    df_t_jgutw2 = df_t_jgut.copy()
    df_t_jgutw2["Option"] = "Option 2"
    df_t_jgutw2["Filter"] = "jgutw"
    df_t_jgutw = pd.concat([df_t_jgutw, df_t_jgutw2], ignore_index = True)
    del df_t_jgutw2
    
    df_t_gut = pd.DataFrame(data = time_sim_gut.T, columns = ["Run time [s]"])
    df_t_gut["Filter"] = "gut"
    df_t_gut["Option"] = "Option 1"
    
    df_t_qf = pd.DataFrame(data = time_sim_qf.T, columns = ["Run time [s]"])
    df_t_qf["Filter"] = "Fixed"
    df_t_qf["Option"] = "Option 1"
    
    df_t_lin = pd.DataFrame(data = time_sim_lin.T, columns = ["Run time [s]"])
    df_t_lin["Filter"] = "lin"
    df_t_lin["Option"] = "Option 1"
    
    
    df_t = pd.concat([df_t_jgut, 
                      df_t_jgutw, 
                      df_t_lin, 
                      df_t_qf], ignore_index = True)
    
    del df_t_jgut, df_t_jgutw, df_t_lin, df_t_qf
    
    fig_rt, ax_rt = plt.subplots(1, 1)
    sns.violinplot(x = "Filter", y = "Run time [s]", data = df_t, hue = "Option", split = True, ax = ax_rt, alpha = .2, legend = False, inner = "stick")
    plt.setp(ax_rt.collections, alpha=.3)
    # ax_rt.legend().remove()
    # sns.stripplot(x = "Filter", y = "Run time [s]", data = df_t, hue = "Option", dodge = True, ax = ax_rt)
    
    handles_leg, labels_leg = ax_rt.get_legend_handles_labels()
    ax_rt.legend(handles_leg[-2:], labels_leg[-2:]) 
    
    # ax_rt.legend().remove()
    # sns.swarmplot(x = "Filter", y = "Run time [s]", data = df_t, hue = "sigma", dodge = True, ax = ax_rt)
    
    ax_rt.set_yscale('log')

#%% Save variables
    dir_project = pathlib.Path(__file__).parent.parent 
    dir_data = os.path.join(dir_project, "data")
    
    df_t.to_csv(os.path.join(dir_data, "sim_time.csv"))
    df_conc.to_csv(os.path.join(dir_data, "consistency.csv"))
    if "df_cost_rmse_all" in locals(): #check if variable exists
        df_cost_rmse_all.to_csv(os.path.join(dir_data, "df_cost_rmse_all.csv"))
        df_cost_mean_all.to_csv(os.path.join(dir_data, "df_cost_mean_all.csv"))

#example of reading back to a pandas file
# df_t2 = pd.read_csv(os.path.join(dir_data, "sim_time.csv"))

# class MyException(Exception):
#     pass
# raise MyException("negative value")