# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:55:28 2023

@author: halvorak
"""


import numpy as np

# import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import os
# import matplotlib.patches as plt_patches
# import matplotlib.transforms as plt_transforms
import pandas as pd
import seaborn as sns

import utils_batch_gasreactor as utils_gr
font = {'size': 14}
matplotlib.rc('font', **font)
# cmap = "tab10"
# plt.set_cmap(cmap)

#%% Directories
dir_project = pathlib.Path(__file__).parent.parent 
# dir_data = os.path.join(dir_project, "data_gasreactor")
# dir_data = os.path.join(dir_project, "data_gasreactor_sample_par_every_dt")
dir_data = os.path.join(dir_project, "data_gasreactor_sample_par_every_dt")

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
print("Simulation time [s] in mean (std_dev)")
print(f'Gut: {df_t[df_t["Filter"] == "gut"]["Run time [s]"].mean()} ({df_t[df_t["Filter"] == "gut"]["Run time [s]"].std()})\n'
       f'Lin: {df_t[df_t["Filter"] == "lin"]["Run time [s]"].mean()} ({df_t[df_t["Filter"] == "lin"]["Run time [s]"].std()})\n',
       f'MC: {df_t[df_t["Filter"] == "mc"]["Run time [s]"].mean()} ({df_t[df_t["Filter"] == "mc"]["Run time [s]"].std()})\n'
      )


df_t[df_t["Filter"] == "gut"]["Run time [s]"].mean()
df_t[df_t["Filter"] == "mc"]["Option"].unique()
df_t["Filter"].unique()
#%% Plot
filters = ["gut", "lin", "mc"]

rmse_mean.loc[filters, :].plot.bar(subplots = True)
# df_rmse.mean().unstack(level=1).plot.bar(x=["gut", "lin", "mc"],subplots = True)



