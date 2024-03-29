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

import matplotlib.pyplot as plt
import matplotlib
import pathlib
import os
import scipy.linalg
import scipy.stats
import copy
import time
import timeit
import pandas as pd
import seaborn as sns


#Self-written modules
from state_estimator import sigma_points_classes as spc
from state_estimator import UKF
from state_estimator import unscented_transform as ut
from state_estimator import myExceptions
import utils_batch_gasreactor as utils_gr
font = {'size': 14}
matplotlib.rc('font', **font)
# cmap = "tab10"
# plt.set_cmap(cmap)


#%% Set N simulation times
N_sim = 100 #this is how many times to repeat each iteration
overwrite_results = True #overwrites the saved results
points_x = "genut" #sigma-point method in the propagataion step
# points_x = "scaled"
sqrt_method = lambda P: scipy.linalg.cholesky(P, lower = True) #for sigma-points
x_var = ["A", "B", "C"]
dim_x = len(x_var)
power_par = 3.

cost_func_type = "RMSE" #other valid option is "valappil"

filters_to_run = ["gut", "lin", "mc"]

N_mc_dist = int(1e3) #Numer of MC samples to estimate w~(w^,Q) in Valappil's approach

#Cost function. Check both mean and rmse
j_valappil_gut = np.zeros((dim_x, N_sim))
j_valappil_mc = np.zeros((dim_x, N_sim))
j_valappil_lin = np.zeros((dim_x, N_sim))

j_mean_gut = np.zeros((dim_x, N_sim))
j_mean_mc = np.zeros((dim_x, N_sim))
j_mean_lin = np.zeros((dim_x, N_sim))

time_sim_gut = np.zeros(N_sim)
time_sim_mc = np.zeros(N_sim)
time_sim_lin = np.zeros(N_sim)

Ni = 0
rand_seed = 6969

ts = time.time() #total start time
ti = time.time() #iteration time for a MC run

print_subiter = True #print certain timesteps for a single case
num_exceptions = 0 #number of times we fail and start over
while Ni < N_sim:
    try:
        np.random.seed(rand_seed) #to get reproducible results. rand_seed updated in every iteration
        t_iter = time.time()
        
        #%% Import parameters
        dt_y = .25 # [-] Measurement frequency
        x0, P0, par_mean_fx, par_cov_fx, cm3_par, cm4_par, par_samples, Q_nom, R_nom = utils_gr.get_literature_values_points_dist(dt_y, N_samples = int(1e6), power_par = power_par)
            
        #%% Define dimensions and initialize arrays
        
        dim_x = x0.shape[0]
        dim_par_fx = par_cov_fx.shape[0]
        
        t_end = 30 # []
        t = np.linspace(0, t_end, int(t_end/dt_y))
        dim_t = t.shape[0]
        
        y0 = utils_gr.hx(x0)
        dim_y = y0.shape[0]
        y = np.zeros((dim_y, dim_t))
        y[:, 0] = y0*np.nan
        
        
        #%% Allocate initial arrays and set x0
        x0_dist = scipy.stats.multivariate_normal(mean = x0, cov = P0)
        try:
            x0_kf = utils_gr.get_positive_point(x0_dist, eps=1e-10) # random value from x0_kf from the multivariate normal dist with (mean=x0_true, cov = P0) and x0_kf >= eps
        except AssertionError:
            x0_kf = x0.copy() + 1e-10

        
        #Arrays where values are stored
        x_true = np.zeros((dim_x, dim_t)) 
        x_ol = np.zeros((dim_x, dim_t)) #Open loop simulation - same starting point and param as UKF
        
        #Arrays where posterior prediction is stored
        x_post_gut = np.zeros((dim_x, dim_t))
        x_post_mc = np.zeros((dim_x, dim_t))
        x_post_lin = np.zeros((dim_x, dim_t))
        
        #Track history of computed w_mean-s 
        w_gut_hist = np.zeros((dim_x, dim_t))
        w_mc_hist = np.zeros((dim_x, dim_t))
        
        #Track history of computed Q-s (only diagonals)
        Q_gut_hist = np.zeros((dim_x, dim_t))
        Q_mc_hist = np.zeros((dim_x, dim_t))
        Q_lin_hist = np.zeros((dim_x, dim_t))
        
        #diagnonal elements of covariance matrices
        P_diag_post_gut = np.zeros((dim_x, dim_t))
        P_diag_post_mc = np.zeros((dim_x, dim_t))
        P_diag_post_lin = np.zeros((dim_x, dim_t))
        
        #save the starting points for the true system and the filters
        x_true[:, 0] = x0
        x_post_gut[:, 0] = x0_kf.copy()
        x_post_mc[:, 0] = x0_kf.copy()
        x_post_lin[:, 0] = x0_kf.copy()
        x0_ol = x0_kf.copy()
        x_ol[:, 0] = x0_ol
        
        #save starting points for covariance matrices
        P_diag_post_gut[:, 0] = np.diag(P0.copy())
        P_diag_post_mc[:, 0] = np.diag(P0.copy())
        P_diag_post_lin[:, 0] = np.diag(P0.copy())
        
        t_span = (t[0],t[1])
        #%% Define UKF with adaptive Q, R from GenUT
        lbx = .0
        k_positive = 1 - 1e-8
        if points_x == "scaled":
            alpha = 1e-2
            beta = 2.
            kappa = 0.#3-dim_x
            points_gut = spc.ScaledSigmaPoints(dim_x,
                                                    alpha,
                                                    beta,
                                                    kappa, sqrt_method = sqrt_method)
        elif points_x == "genut":
            points_gut = spc.GenUTSigmaPoints(dim_x, sqrt_method = sqrt_method, theta = k_positive, lbx = lbx)
        
        fx_ukf_gut = None #updated later in the simulation
        kfc_gut = UKF.UKF_additive_noise(x0_kf.copy(), P0.copy(), None, utils_gr.hx, points_gut, Q_nom, R_nom, name="gut") 
        
        #%% Define UKF with adaptive Q, R from MC 
        if points_x == "scaled":
            alpha_mc = copy.copy(alpha)
            beta_mc = copy.copy(beta)
            kappa_mc = copy.copy(kappa)
            points_mc = spc.ScaledSigmaPoints(dim_x,
                                                    alpha_mc,
                                                    beta_mc,
                                                    kappa_mc, sqrt_method = sqrt_method)
        elif points_x == "genut":
            points_mc = spc.GenUTSigmaPoints(dim_x, sqrt_method = sqrt_method, theta = k_positive, lbx = lbx)
            
        kfc_mc = UKF.UKF_additive_noise(x0_kf.copy(), P0.copy(), None, utils_gr.hx, points_mc, Q_nom, R_nom, name="mc")
        
        #%% Define UKF with adaptive Q, R from linearized approach
        if points_x == "scaled":
            alpha_lin = copy.copy(alpha)
            beta_lin = copy.copy(beta)
            kappa_lin = copy.copy(kappa)
            points_lin = spc.ScaledSigmaPoints(dim_x,
                                                    alpha_lin,
                                                    beta_lin,
                                                    kappa_lin, sqrt_method = sqrt_method)
        elif points_x == "genut":
            points_lin = spc.GenUTSigmaPoints(dim_x, sqrt_method = sqrt_method, theta = k_positive, lbx = lbx)
        
        kfc_lin = UKF.UKF_additive_noise(x0_kf.copy(), P0.copy(), None, utils_gr.hx, points_lin, Q_nom, R_nom, name="lin")
       
        #%% Get parametric uncertainty of fx by GenUT. Generate sigmapoints first ("offline")
        positive_sigmas_par = True
        k_positive_par = k_positive
        points_par = spc.GenUTSigmaPoints(dim_par_fx, sqrt_method = sqrt_method, theta = k_positive_par, lbx = lbx)
        
        sigmas_fx_gut, w_fx_gut, _, _ = points_par.compute_sigma_points(np.array(list(par_mean_fx.values())), par_cov_fx, S = cm3_par, K = cm4_par)
        
        #%% N_MC samples, random sampling
        
        par_mc_fx = par_samples[np.random.randint(0, high = par_samples.shape[0], size = N_mc_dist), :]
        
        if False: #plot MC samples
            df_par_mc = pd.DataFrame(data = par_mc_fx, columns = [r"$k_" + str(i+1) + "$" for i in range(dim_par_fx)])
            sns.pairplot(df_par_mc, corner = True)
            
            std_dev_par = np.sqrt(np.diag(par_cov_fx))
            std_dev_inv = np.diag([1/si for si in std_dev_par])
            corr_par = std_dev_inv @ par_cov_fx @ std_dev_inv
            # raise ValueError
        par_mc_fx = par_mc_fx.T
        
        #%% Q_fixed, robustness
        Q_diag_min = np.eye(dim_x)*1e-10
        
        #%% Casadi integrator, jacobian df/dp
        F, jac_p_func, x_var_cd, p_var_cd,_,=  utils_gr.ode_model_plant(dt_y, power_par = power_par)
        assert dim_x == x_var_cd.shape[0], "Dimension mismatch between x0 and x_var_cd"
        
        #%% Simulate the plant
        
        par_true_val = par_samples[np.random.randint(0, high = par_samples.shape[0]), :] #same through entire simulation
        # par_true_val = utils_gr.get_points(par_dist_multivar, par_dist_univar, N = 1, constraint = 1e-10) #same through entire simulation
        par_true_fx = {key: val for key, val in zip(par_mean_fx.keys(), par_true_val)} #integrator function takes the parameters as a dict
        for i in range(1, dim_t):
            t_span = (t[i-1], t[i])
            
            #sample parameter values for the plant
            par_true_val = par_true_val = par_samples[np.random.randint(0, high = par_samples.shape[0]), :]
            assert (par_true_val > 0).all()
            par_true_fx = {key: val for key, val in zip(par_mean_fx.keys(), par_true_val)}
            
            #Simulate the true plant
            x_true[:, i] = utils_gr.integrate_ode(F, 
                                                  x_true[:,i-1],
                                                  par_true_fx)
            
            #if we obtain a negative x_true-value, it is unphysical and due to the numerical integration. If negative value is detected, set the value to 0 (or close to)
            neg_xtrue_val = x_true[:, i] <= 0
            x_true[neg_xtrue_val, i] = 1e-10
            
            #Simulate the open loop (kf parameters and starting point)
            x_ol[:, i] = utils_gr.integrate_ode(F, 
                                                x_ol[:,i-1], 
                                                par_mean_fx)
            #Make a new measurement
            vk = np.array([np.random.normal(0, sig_i) for sig_i in np.sqrt(np.diag(R_nom))])
            y[:, i] = utils_gr.hx(x_true[:, i]) + vk
            
        
        y[:, 0] = np.nan #the 1st measurement is not real, just for programming convenience
            
        #%% Run state estimators on the entire measurement set
        #Get i) process noise statistics and ii) prior estimates for the different UKFs
       
        if "gut" in filters_to_run:
            #Adaptive Q by GenUT
            ts_gut = timeit.default_timer()
            for i in range(1, dim_t):
                x_nom_gut = utils_gr.integrate_ode(F, x_post_gut[:,i-1], par_mean_fx)
                
                #function for calculating Qk
                fx_gen_Q_gut = lambda si: utils_gr.fx_for_UT_gen_Q(si, list(par_mean_fx.keys()).copy(), F, x_post_gut[:, i-1], par_mean_fx.copy()) - x_nom_gut
                
                w_mean_gut, Q_gut = ut.unscented_transform_w_function_eval(sigmas_fx_gut.copy(), w_fx_gut, w_fx_gut, fx_gen_Q_gut, first_yi = np.zeros(dim_x)) #calculate Qk. The first propagated sigma-point contains only zeros
               
                Q_gut = Q_gut + Q_diag_min #robustness/non-zero on diagonals
                kfc_gut.Q = Q_gut #assign to filter
                w_gut_hist[:, i] = w_mean_gut #Save w_mean history
                Q_gut_hist[:, i] = np.diag(Q_gut) #Save Q history
                
                #Normal UKF
                fx_ukf_gut = lambda x: utils_gr.integrate_ode(F, x, par_mean_fx)
                kfc_gut.predict(fx = fx_ukf_gut, w_mean = w_mean_gut)
                kfc_gut.update(y[:, i], hx = utils_gr.hx)
                
                #Save estimates
                x_post_gut[:, i] = kfc_gut.x_post
                P_diag_post_gut[:, i] = np.diag(kfc_gut.P_post)
            
            tf_gut = timeit.default_timer()
            time_sim_gut[Ni] = tf_gut - ts_gut
      
        if "lin" in filters_to_run:    
            #Adaptive Q by linearization
            ts_lin = timeit.default_timer()
            for i in range(1, dim_t):
                
                Q_lin = utils_gr.get_Q_from_linearization(jac_p_func, 
                                                          x_post_lin[:, i-1], par_mean_fx.copy(), par_cov_fx)
                Q_lin = Q_lin + Q_diag_min #robustness/non-zero on diagonals
                kfc_lin.Q = Q_lin #assign to filter
                Q_lin_hist[:, i] = np.diag(Q_lin) #Save Q history
                
                #Normal UKF
                fx_ukf_lin = lambda x: utils_gr.integrate_ode(F, x, par_mean_fx)
                kfc_lin.predict(fx = fx_ukf_lin)
                kfc_lin.update(y[:, i], hx = utils_gr.hx)
                
                #Save estimates
                x_post_lin[:, i] = kfc_lin.x_post
                P_diag_post_lin[:, i] = np.diag(kfc_lin.P_post)
                
            tf_lin = timeit.default_timer()
            time_sim_lin[Ni] = tf_lin - ts_lin
        
        if "mc" in filters_to_run:
            #Adaptive Q by MC random and w_mean
            ts_mc = timeit.default_timer()
            for i in range(1, dim_t):
                t_span = (t[i-1], t[i])
                    
                w_mean_mc, Q_mc = utils_gr.get_wmean_Q_from_mc(par_mc_fx.copy(),
                                                                F,
                                                                x_post_mc[:, i-1],
                                                                par_mean_fx.copy())
                Q_mc = Q_mc + Q_diag_min #robustness/non-zero on diagonals
                w_mc_hist[:, i] = w_mean_mc #Save w_mean history
                Q_mc_hist[:, i] = np.diag(Q_mc) #Save Q history
                kfc_mc.Q = Q_mc #assign to filter
                
                #Normal UKF
                fx_ukf_mc = lambda x: utils_gr.integrate_ode(F, x, par_mean_fx)
                kfc_mc.predict(fx = fx_ukf_mc, w_mean = w_mean_mc)
                kfc_mc.update(y[:, i])
                
                #Save estimates
                x_post_mc[:, i] = kfc_mc.x_post
                P_diag_post_mc[:, i] = np.diag(kfc_mc.P_post)
                # if i%10 == 0:
                #     print(f"Iter {i}/{dim_t} in MC tuning")
                
            tf_mc = timeit.default_timer()
            time_sim_mc[Ni] = tf_mc - ts_mc
        
        #%% Compute performance index
        value_filter_not_run = 1 #same cost as OL response
       
        if "gut" in filters_to_run:
            j_valappil_gut[:, Ni] = utils_gr.compute_performance_index_valappil(
                x_post_gut, 
                                                                             x_ol, 
                                                                             x_true, cost_func = cost_func_type)
            j_mean_gut[:, Ni] = np.mean(x_post_gut - x_true, axis = 1)
        else:
            j_valappil_gut[:, Ni] = value_filter_not_run
       
        if "mc" in filters_to_run:
            j_valappil_mc[:, Ni] = utils_gr.compute_performance_index_valappil(x_post_mc, 
                                                                             x_ol, 
                                                                             x_true, cost_func = cost_func_type)
            j_mean_mc[:, Ni] = np.mean(x_post_mc - x_true, axis = 1)
        else:
            j_valappil_mc[:, Ni] = value_filter_not_run
       
        if "lin" in filters_to_run:
            j_valappil_lin[:, Ni] = utils_gr.compute_performance_index_valappil(x_post_lin, 
                                                                             x_ol, 
                                                                             x_true, cost_func = cost_func_type)
            j_mean_lin[:, Ni] = np.mean(x_post_lin - x_true, axis = 1)
        else:
            j_valappil_lin[:, Ni] = value_filter_not_run
       
        j_valappil_i = np.vstack((j_valappil_gut[:, Ni],
                                  j_valappil_mc[:, Ni],
                                  j_valappil_lin[:, Ni],
                                  )).T
        
        Ni += 1
        rand_seed += 1
        time_iteration = time.time() - t_iter
        if (Ni%1 == 0): #print every Xth iteration                                                               
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
        num_exceptions += 1
        rand_seed += 1
        print(f"Iter: {i}: Time spent, t_iter = {time.time()-ti: .2f} s ")
        continue
                

     
#%% Plot x, x_pred, y
ylabels = [ r"$c_A$ [-]", r"$c_B [-]$", r"$c_C [-]$"]

print(f"Repeated {N_sim} time(s). In every iteration, the number of model evaluations for computing noise statistics:\n",
      f"Q by UT: {sigmas_fx_gut.shape[1]}\n",
      f"Q by MC: {N_mc_dist}\n"
      )

plot_it = True
if plot_it:
    alpha_fill = .2
    kwargs_pred = {"linestyle": "dashed"}
    kwargs_gut = {"alpha": alpha_fill}
    kwargs_mc = {"alpha": alpha_fill}
    kwargs_lin = {"alpha": alpha_fill}
    #
    meas_idx = np.array([])
    idx_y = 0
    filters_to_plot = [
        "gut",
        # "mc",
        "lin",
        # "ol"
        ]
    
    font = {'size': 16}
    matplotlib.rc('font', **font)
    fig1, ax1 = plt.subplots(dim_x, 1, sharex = True, layout = "constrained", figsize = (9,6))
    
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
        
        #plot state predictions
        #Q_gut
        if "gut" in filters_to_plot:
            l_gut = ax1[i].plot(t, x_post_gut[i, :], label = r"$\hat{x}^+_{GenUT}$", **kwargs_pred)
        
        #Q_mc
        if "mc" in filters_to_plot:
            l_mc = ax1[i].plot(t, x_post_mc[i, :], label = r"$\hat{x}^+_{mc}$", **kwargs_pred)

        #Q_lin
        if "lin" in filters_to_plot:
            l_lin = ax1[i].plot(t, x_post_lin[i, :], label = r"$\hat{x}^+_{Lin}$", **kwargs_pred)
        
        if plt_std_dev: #plot shading around mean trajectory
            #Genut
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
            
            #mc
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
        
        ylim_scaled = ax1[i].get_ylim()
        ax1[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        
        if "ol" in filters_to_plot:
            ax1[i].plot(t, x_ol[i, :], label = "OL", **kwargs_pred)
        ax1[i].set_ylabel(ylabels[i])
        # ax1[i].legend(frameon = False, ncol = 3) 
    ax1[-1].set_xlabel("Time [h]")
    
    if True: #inset additional zoomed axis on last axes
        ax_c = ax1[-1].inset_axes([0.5, 0.4, 0.35, 0.3])
        
        #replot in this new axes
        i = 2
        #plot true state
        ax_c.plot(t, x_true[i, :], label = r"$x_{true}$")
        # ax_c.plot(t, x_true[i, :], label = r"$x_{true}$", color = 'b')
    
        #plot measurements
        if i in meas_idx:
            ax_c.scatter(t, y[idx_y, :], 
                            color = "m", 
                            # color = l[0].get_color(), 
                            s = 2,
                            alpha = .2,
                            marker = "o",
                            label = r"$y$")
            idx_y += 1
        
        #plot state predictions
        #Q_gut
        if "gut" in filters_to_plot:
            l_gut = ax_c.plot(t, x_post_gut[i, :], label = r"$\hat{x}^+_{GenUT}$", **kwargs_pred)
        
        #Q_mc
        if "mc" in filters_to_plot:
            l_mc = ax_c.plot(t, x_post_mc[i, :], label = r"$\hat{x}^+_{mc}$", **kwargs_pred)

        #Q_lin
        if "lin" in filters_to_plot:
            l_lin = ax_c.plot(t, x_post_lin[i, :], label = r"$\hat{x}^+_{Lin}$", **kwargs_pred)
        
        if plt_std_dev: #plot shading around mean trajectory
            #Genut
            if "gut" in filters_to_plot:
                kwargs_gut.update({"color": l_gut[0].get_color()})
                ax_c.fill_between(t, 
                                    x_post_gut[i, :] + 2*np.sqrt(P_diag_post_gut[i,:]),
                                    x_post_gut[i, :] - 2*np.sqrt(P_diag_post_gut[i,:]),
                                    **kwargs_gut)
                ax_c.fill_between(t, 
                                    x_post_gut[i, :] + 1*np.sqrt(P_diag_post_gut[i,:]),
                                    x_post_gut[i, :] - 1*np.sqrt(P_diag_post_gut[i,:]),
                                    **kwargs_gut)
            
            #mc
            if "mc" in filters_to_plot:
                kwargs_mc.update({"color": l_mc[0].get_color()})
                ax_c.fill_between(t, 
                                    x_post_mc[i, :] + 2*np.sqrt(P_diag_post_mc[i,:]),
                                    x_post_mc[i, :] - 2*np.sqrt(P_diag_post_mc[i,:]),
                                    **kwargs_mc)
                ax_c.fill_between(t, 
                                    x_post_mc[i, :] + 1*np.sqrt(P_diag_post_mc[i,:]),
                                    x_post_mc[i, :] - 1*np.sqrt(P_diag_post_mc[i,:]),
                                    **kwargs_mc)
            
            #Linearized
            if "lin" in filters_to_plot:
                kwargs_lin.update({"color": l_lin[0].get_color()})
                ax_c.fill_between(t, 
                                    x_post_lin[i, :] + 2*np.sqrt(P_diag_post_lin[i,:]),
                                    x_post_lin[i, :] - 2*np.sqrt(P_diag_post_lin[i,:]),
                                    **kwargs_lin)
                ax_c.fill_between(t, 
                                    x_post_lin[i, :] + 1*np.sqrt(P_diag_post_lin[i,:]),
                                    x_post_lin[i, :] - 1*np.sqrt(P_diag_post_lin[i,:]),
                                    **kwargs_lin)
            
        
        #zoom in on desired subregion
        x1, x2, y1, y2 = 20, 25, 0.645, 0.67
        ax_c.set_xlim((x1, x2))
        ax_c.set_ylim((y1, y2))
        ax_c.set_yticklabels([])
        ax_c.set_xlabel(None)
        ax_c.set_ylabel(None)
        
        ax1[-1].indicate_inset_zoom(ax_c, edgecolor = "black")
    
    ax1[0].legend(ncol = 2, frameon = False)   
    
    
    #%% Plot trajectories in same plot
    fig_x, ax_x = plt.subplots(1,1, layout = "constrained")
    kwargs_x_true = {"linestyle": "dashed"}
    label_true = [r"$c_A$", r"$c_B$", r"$c_C$"]
    label_post = [r"$\hat{c}_A^+$", r"$\hat{c}_B^+$", r"$\hat{c}_C^+$"]
    kwargs_post = {"color": None, "alpha": .2}
    
    for i in range(dim_x):
        l_x = ax_x.plot(t, x_true[i, :], **kwargs_x_true, label = label_true[i])
        
        ax_x.plot(t, x_post_gut[i,:], l_x[0].get_color(), label = label_post[i])
        
        kwargs_post.update({"color": l_x[0].get_color()})
        ax_x.fill_between(t, 
                            x_post_gut[i, :] + 2*np.sqrt(P_diag_post_gut[i,:]),
                            x_post_gut[i, :] - 2*np.sqrt(P_diag_post_gut[i,:]),
                            **kwargs_post)
        ax_x.fill_between(t, 
                            x_post_gut[i, :] + 1*np.sqrt(P_diag_post_gut[i,:]),
                            x_post_gut[i, :] - 1*np.sqrt(P_diag_post_gut[i,:]),
                            **kwargs_post)
    
    ax_x.legend()
    ax_x.set_ylabel("Concentration")
    ax_x.set_xlabel("Time")
    
    #%% Plot w_mean-history
    w_labels = [r"$\hat{w}_{GenUT}$", r"$\hat{w}_{MC}$"]
    y_labels = [ r"$c_A$", r"$c_B$", r"$c_C$"]#
    fig_w, ax_w = plt.subplots(dim_x, 1 ,sharex = True, layout = "constrained")
    if dim_x == 1:
        ax_w = [ax_w]
    for i in range(dim_x):
        ax_w[i].plot(t, w_gut_hist[i, :], label = w_labels[0])
        ax_w[i].plot(t, w_mc_hist[i, :], label = w_labels[1])
        ax_w[i].set_ylabel(y_labels[i])
        # ax_w[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0e'))
        ax_w[i].ticklabel_format(scilimits=(-4, 8))
    ax_w[-1].set_xlabel("Time")
    ax_w[0].legend()
    
    #%% Plot Q-history
    q_labels = [r"$Q_{GenUT}$", r"$Q_{MC}$", r"$Q_{Lin}$"]
    y_labels = [ r"$c_A$", r"$c_B$", r"$c_C$"]#
    kwargs_qplot = {}#{"linestyle": "dashed"}
    matplotlib.rc('font', **font)
    fig_q, ax_q = plt.subplots(dim_x, 1 ,sharex = True, layout = "constrained")
    if dim_x == 1:
        ax_q = [ax_q]
    for i in range(dim_x):
        ax_q[i].plot(t, Q_gut_hist[i, :], label = q_labels[0], **kwargs_qplot)
        ax_q[i].plot(t, Q_mc_hist[i, :], label = q_labels[1], **kwargs_qplot)
        ax_q[i].plot(t, Q_lin_hist[i, :], label = q_labels[2], **kwargs_qplot)
        ax_q[i].set_ylabel(y_labels[i])
    ax_q[-1].set_xlabel("Time")
    ax_q[0].legend(ncol = 3)
    
    #%% Cost functions
if N_sim >= 5: #only compute this if we have done some iteration
    cols = ["A","B", "C"]
    df_cost_rmse_gut = pd.DataFrame(data = j_valappil_gut.T.copy(), columns = cols)
    df_cost_rmse_mc = pd.DataFrame(data = j_valappil_mc.T.copy(), columns = cols)
    df_cost_rmse_lin = pd.DataFrame(data = j_valappil_lin.T.copy(), columns = cols)
    
    df_cost_rmse_all = pd.concat(dict( 
                                  lin = df_cost_rmse_lin,
                                  gut = df_cost_rmse_gut,
                                  mc = df_cost_rmse_mc,
                                  ), axis = 1)
    
    
    df_cost_mean_gut = pd.DataFrame(data = j_mean_gut.T.copy(), columns = cols)
    df_cost_mean_mc = pd.DataFrame(data = j_mean_mc.T.copy(), columns = cols)
    df_cost_mean_lin = pd.DataFrame(data = j_mean_lin.T.copy(), columns = cols)
    
    df_cost_mean_all = pd.concat(dict( 
                                  lin = df_cost_mean_lin,
                                  gut = df_cost_mean_gut,
                                  mc = df_cost_mean_mc
                                  ), axis = 1)
    
    
   
#%% Simultation time plot

df_t_gut = pd.DataFrame(data = time_sim_gut.T, columns = ["Run time [s]"])
df_t_gut["Filter"] = "gut"

df_t_lin = pd.DataFrame(data = time_sim_lin.T, columns = ["Run time [s]"])
df_t_lin["Filter"] = "lin"

df_t_mc = pd.DataFrame(data = time_sim_mc.T, columns = ["Run time [s]"])
df_t_mc["Filter"] = "mc"

df_t = pd.concat([df_t_gut, 
                  df_t_lin, 
                   df_t_mc], ignore_index = True)

del df_t_gut, df_t_lin, df_t_mc

fig_rt, ax_rt = plt.subplots(1, 1, layout = "constrained")
sns.violinplot(x = "Filter", y = "Run time [s]", data = df_t, split = True, ax = ax_rt, alpha = .2, legend = False, inner = "stick")
plt.setp(ax_rt.collections, alpha=.3)


handles_leg, labels_leg = ax_rt.get_legend_handles_labels()
ax_rt.legend(handles_leg[-2:], labels_leg[-2:]) 
ax_rt.set_yscale('log')


#%% Save variables
dir_project = pathlib.Path(__file__).parent.parent 
dir_data = os.path.join(dir_project, "data_gasreactor")
if not os.path.exists(dir_data):
    os.mkdir(dir_data)

if overwrite_results:
    df_t.to_csv(os.path.join(dir_data, "sim_time.csv"))
    if "df_cost_rmse_all" in locals(): #check if variable exists
        df_cost_rmse_all.to_csv(os.path.join(dir_data, "df_cost_rmse_all.csv"))
        df_cost_mean_all.to_csv(os.path.join(dir_data, "df_cost_mean_all.csv"))

#%% rmse_mean in table
rmse_mean_table = df_cost_rmse_all.mean().unstack(level = 1)
rmse_std_table = df_cost_rmse_all.std().unstack(level = 1)
print(f"\nrmse_mean_table*100:\n{rmse_mean_table*100}")
print(f"\nrmse_mean/rmse_mean_gut*100:\n{(rmse_mean_table/rmse_mean_table.loc['gut',:])*100}")

print(f"\nrmse_std_table*100:\n{rmse_std_table*100}")



