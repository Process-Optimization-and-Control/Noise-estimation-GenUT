# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:34:15 2023

@author: halvorak
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 07:34:18 2023

@author: halvorak
"""

import numpy as np
import scipy.linalg
import scipy.stats
import sklearn.datasets
import pandas as pd
import pathlib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time

from state_estimator import sigma_points_classes as spc
from state_estimator import unscented_transform as UT

np.set_printoptions(linewidth=np.nan)


dir_project = pathlib.Path(__file__).parent.parent 
dir_data = os.path.join(dir_project, "data_toy_example")

load_old_sim = True #uses data in dir_data
overwrite_results = False #overwrites data in dir_data

N_sim = int(100)
N_mc = int(1e7) #number of MC samples to run

error_func = lambda P_diff: scipy.linalg.norm(P_diff, "fro") #P_diff = Py_ut - Py_mc
df_error = pd.DataFrame(data = np.zeros((N_sim, 2)), columns = ["Pya", "Py*"])


if load_old_sim:
    df_error = pd.read_pickle(os.path.join(dir_data, "df_error.pkl"))
else:
    for ni in range(N_sim):
        ts = time.time()
        #%% Define distributions
        x1 = np.random.uniform(low = 1., high = 2., size = 2)
        P1 = sklearn.datasets.make_spd_matrix(2)
        
        x2 = np.random.uniform(low = 1., high = 2., size = 1)
        P2 = sklearn.datasets.make_spd_matrix(1)
        
        xa = np.hstack((x1,x2))
        dim_xa = xa.shape[0]
        dim_x1 = x1.shape[0]
        dim_x2 = x2.shape[0]
        Pa = scipy.linalg.block_diag(P1, P2)
        
        #%% Define nonlinear function
        func = lambda x: x**2 + x*x[-1]
        func2 = lambda u, v: func(np.hstack((u, v)))
        
        #%% Define sigma-points
        sqrt_method = lambda P: scipy.linalg.cholesky(P, lower = True)
        P_sqrt = sqrt_method(Pa)
        assert np.allclose(P_sqrt @ P_sqrt.T, Pa)
        points_implemented = ["genut", "julier"]
        point_fn = points_implemented[0]
        
        #augmented system
        if point_fn == "julier":
            points = spc.JulierSigmaPoints(dim_xa, sqrt_method = sqrt_method, kappa = 3 - dim_xa)
        elif point_fn == "genut": 
            points = spc.GenUTSigmaPoints(dim_xa, sqrt_method = sqrt_method)
        else:
            raise KeyError
        sigmas, Wm, Wc, P_sqrt = points.compute_sigma_points(xa, Pa)
        ym, Py, Aa = UT.unscented_transform_w_function_eval_wslr(sigmas, Wm, Wc, func)
        
        #v
        if point_fn == "julier":
            points_x2 = spc.JulierSigmaPoints(dim_x2, sqrt_method = sqrt_method, kappa = 3 - dim_x2)
        elif point_fn =="genut":
            points_x2 = spc.GenUTSigmaPoints(dim_x2, sqrt_method = sqrt_method)
        else:
            raise KeyError
        sigmas_x2, Wm_x2, Wc_x2, P_sqrt = points_x2.compute_sigma_points(x2, P2)
        y0 = func2(x1, x2)
        v_func = lambda z: func2(x1, z) - y0
        vm, Pv, Ax2 = UT.unscented_transform_w_function_eval_wslr(sigmas_x2, Wm_x2, Wc_x2, v_func)
        
        #x_nom
        if point_fn =="julier":
            points_x1 = spc.JulierSigmaPoints(dim_x1, sqrt_method = sqrt_method, kappa = 3 - dim_x1)
        elif point_fn =="genut":
            points_x1 = spc.GenUTSigmaPoints(dim_x1, sqrt_method = sqrt_method)
        sigmas_x1, Wm_x1, Wc_x1, P_sqrt = points_x1.compute_sigma_points(x1, P1)
        ynom_func = lambda q: func2(q, x2)
        y_nom, Py_nom, Ax1 = UT.unscented_transform_w_function_eval_wslr(sigmas_x1, Wm_x1, Wc_x1, ynom_func)
        
        #results of reformulated system
        ym2 = y_nom + vm
        Py2 = Py_nom + Pv
        
        #check mean prediction are equal
        assert np.allclose(ym, ym2), "Mean predictions are not equal"
        
        #check covariance estimates are SPD (this will throw an error if they are not SPD)
        scipy.linalg.cholesky(Py, lower = True)
        scipy.linalg.cholesky(Py2, lower = True)
        #%%Monte Carlo result
        x_samples = np.random.multivariate_normal(xa, Pa, size = N_mc)
        y_samples = np.array(list(map(func, x_samples)))
        ym_mc = np.mean(y_samples, axis = 0)
        Py_mc = np.cov(y_samples, rowvar = False)
        
        #%% Evaluate error
        df_error.loc[ni, "Pya"] = error_func(Py - Py_mc)
        df_error.loc[ni, "Py*"] = error_func(Py2 - Py_mc)
        
        #print progress
        if (ni%1 == 0): #print every Xth iteration                                                               
            # print(f"Iter {ni}/{N_sim} done.")
            print(f"Iter {ni}/{N_sim} done. t_iter = {time.time()-ts: .2f} s")
    
    #%%Save results (if selected)
    if overwrite_results:
        df_error.to_pickle(os.path.join(dir_data, "df_error.pkl"))

#%% Plot histogram
fig_hist, ax_hist = plt.subplots(1,1,layout = "constrained")
sns.histplot(df_error, ax = ax_hist)
ax_hist.set_xlabel(r"RMSE [-]")

#%% Compute p-values and statistics
t_stat, p_val = scipy.stats.ttest_ind(df_error["Py*"], df_error["Pya"], alternative = "less")
print(f"df_error.mean():\n{df_error.mean()}\n\ndf_error.std():\n{df_error.std()}")
print(f"\n\n{p_val=}")



