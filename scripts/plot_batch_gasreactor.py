# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:55:28 2023

@author: halvorak
"""


import matplotlib.pyplot as plt
import matplotlib
import pathlib
import os
import pandas as pd
import numpy as np
import scipy.stats

font = {'size': 14}
matplotlib.rc('font', **font)

#%% Directories
dir_project = pathlib.Path(__file__).parent.parent 
dir_data = os.path.join(dir_project, "data_gasreactor") #the case that was most recently simulated
dir_data = os.path.join(dir_project, "data_gasreactor_article_values") #data in the article

#example of reading back to a pandas file
df_t = pd.read_csv(os.path.join(dir_data, "sim_time.csv"))
df_rmse = pd.read_csv(os.path.join(dir_data, "df_cost_rmse_all.csv"), header = [0,1], index_col = 0)
df_cost_mean = pd.read_csv(os.path.join(dir_data, "df_cost_mean_all.csv"), header = [0,1], index_col = 0)

#%%Statistics in a matrix
rmse_mean = df_rmse.mean().unstack(level=1)
rmse_std = df_rmse.std().unstack(level=1)
cost_mean_mean = df_cost_mean.mean().unstack(level=1)
cost_mean_std = df_cost_mean.std().unstack(level=1)

rmse_mean_rel = rmse_mean/rmse_mean.loc["lin",:]-1

case_res = os.path.normpath(dir_data).split(os.path.sep)[-1]

pd.options.display.float_format = "{:,.4f}".format
print(f"Case: {case_res}")
print("RMSE_mean*100:")
print(rmse_mean*100)
print(f"\nRMSE_mean/RMSE_mean_gut*100:\n{rmse_mean/rmse_mean.loc['gut',:]*100}")
print("\nRMSE_std*100:")
print(rmse_std*100)
print("\n")

print("Simulation time [s] in mean (std_dev)")
print(f'Gut: {df_t[df_t["Filter"] == "gut"]["Run time [s]"].mean() :.2f} ({df_t[df_t["Filter"] == "gut"]["Run time [s]"].std() :.2f})\n'
       f'Lin: {df_t[df_t["Filter"] == "lin"]["Run time [s]"].mean() :.2f} ({df_t[df_t["Filter"] == "lin"]["Run time [s]"].std() :.2f})\n',
       f'MC: {df_t[df_t["Filter"] == "mc"]["Run time [s]"].mean() :.2f} ({df_t[df_t["Filter"] == "mc"]["Run time [s]"].std() :.2f})\n'
      )



#%% p-test on RMSE

#p-test on the RMSE values for the different variables. Based on 100 "observations"

var = ["A", "B", "C"]
fig_hist, ax_hist = plt.subplots(3,1)
hist_kwargs = {"bins": 30}
for v, ax_h in zip(var, ax_hist):
    # Perform t-test
    t_stat, p_val = scipy.stats.ttest_ind(df_rmse["gut", v], df_rmse["lin", v], alternative = "less")
    t_stat2, p_val2 = scipy.stats.ttest_ind(df_rmse["gut", v], df_rmse["mc", v], alternative = "less")
    
    print(f"var: {v}")
    # print(f'T-statistic: {t_stat}')
    print(f'P-value GenUT=Lin: {p_val :.3f}')
    print(f'P-value GenUT=MC: {p_val2 :.3f}')

    ax_h.hist(df_rmse["gut", v], label = "GenUT", **hist_kwargs)
    ax_h.hist(df_rmse["lin", v], label = "Lin", **hist_kwargs)
ax_hist[0].legend()
    
