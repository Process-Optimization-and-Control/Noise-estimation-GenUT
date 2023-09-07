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

font = {'size': 14}
matplotlib.rc('font', **font)

#%% Directories
dir_project = pathlib.Path(__file__).parent.parent 
# dir_data = os.path.join(dir_project, "data_gasreactor") #the case that was most recently simulated
dir_data = os.path.join(dir_project, "data_gasreactor_fixed_parametric_mismatch") #case 1 in the paper
# dir_data = os.path.join(dir_project, "data_gasreactor_sample_par_every_dt") #case 2 in the paper

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
print("\nRMSE_std*100:")
print(rmse_std*100)
print("\n")

print("Simulation time [s] in mean (std_dev)")
print(f'Gut: {df_t[df_t["Filter"] == "gut"]["Run time [s]"].mean() :.2f} ({df_t[df_t["Filter"] == "gut"]["Run time [s]"].std() :.2f})\n'
       f'Lin: {df_t[df_t["Filter"] == "lin"]["Run time [s]"].mean() :.2f} ({df_t[df_t["Filter"] == "lin"]["Run time [s]"].std() :.2f})\n',
       f'MC: {df_t[df_t["Filter"] == "mc"]["Run time [s]"].mean() :.2f} ({df_t[df_t["Filter"] == "mc"]["Run time [s]"].std() :.2f})\n'
      )

#%% Plot
filters = ["lin", "mc"]

rmse_mean_rel = (rmse_mean/rmse_mean.loc["gut", :]-1)*100
ax_bar = rmse_mean_rel.loc[filters,:].plot.bar(subplots = True)
x_lim = ax_bar[0].get_xlim()
for axi in ax_bar:
    axi.plot(x_lim, [0,0], 'k')
    axi.set_xlim(x_lim)
fig_bar = axi.get_figure()
fig_bar.suptitle(r"$(J_{mean}/J_{mean,GenUT}-1)*100$")

