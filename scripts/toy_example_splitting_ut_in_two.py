# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 07:34:18 2023

@author: halvorak
"""

import numpy as np
import scipy.linalg

from state_estimator import sigma_points_classes as spc
from state_estimator import unscented_transform as UT

np.set_printoptions(linewidth=np.nan)
#%% test UT large dimension

def get_corr_std_dev(P):
    std_dev = np.sqrt(np.diag(P))
    std_dev_inv = np.diag([1/si for si in std_dev])
    corr = std_dev_inv @ P @ std_dev_inv
    return std_dev, corr

x1 = np.array([1., 2.])
P1 = np.array([[4., 0],
                [0, 4.]])

x2 = np.array([4.])
P2 = np.array([[4.]])

xa = np.hstack((x1,x2))
dim_xa = xa.shape[0]
dim_x1 = x1.shape[0]
dim_x2 = x2.shape[0]
Pa = scipy.linalg.block_diag(P1, P2)

# def sqrt_method(P):
#     D, U = np.linalg.eig(P)
#     return U @ np.diag(np.sqrt(D))
sqrt_method = lambda P: scipy.linalg.cholesky(P, lower = True)
P_sqrt = sqrt_method(Pa)
assert np.allclose(P_sqrt @ P_sqrt.T, Pa)

func = lambda x: x**2 + x*x[-1]
func2 = lambda u, v: func(np.hstack((u, v)))

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

Aa2 = np.hstack((Ax1, Ax2))


Aa_lin_f = lambda x: np.array([[2*x[0]+x[2], 0, x[0]],
                             [0, 2*x[1]+x[2], x[1]],
                             [0, 0, 4*x[2]]])
Aa_lin = Aa_lin_f(xa)
Py_lin = Aa_lin @ Pa @ Aa_lin.T
Py_sl_a = Aa @ Pa @ Aa.T
Py_sl_a2 = Aa2 @ Pa @ Aa2.T

ym_lin = func(xa)

#Monte Carlo result
N_mc = int(1e7)
x_samples = np.random.multivariate_normal(xa, Pa, size = N_mc)
y_samples = np.array(list(map(func, x_samples)))
ym_mc = np.mean(y_samples, axis = 0)
Py_mc = np.cov(y_samples, rowvar = False)

#print results
ym2 = y_nom + vm
Py2 = Py_nom + Pv

Pe_a = Py - Py_sl_a #regression error covariance
Pe_a2 = Py2 - Py_sl_a2

error_mean = ym - ym2
error_cov = Py - Py2
error_mc2aug_cov = Py_mc - Py
error_mc2prop_cov = Py_mc - Py2
error_lin_grad_cov = Py_mc - Py_lin

norm_error_cov = np.linalg.norm(error_cov)
norm_error_mc2aug_cov = np.linalg.norm(error_mc2aug_cov)
norm_error_mc2prop_cov = np.linalg.norm(error_mc2prop_cov)
norm_error_lin_grad_cov = np.linalg.norm(error_lin_grad_cov)
error_mc2aug_var = np.diag(error_mc2aug_cov)
error_mc2prop_var = np.diag(error_mc2prop_cov)
print(f"{point_fn=}\n",
      f"Mean error: {error_mean}\n",
      f"{norm_error_mc2aug_cov=}\n",
      f"{norm_error_mc2prop_cov=}\n",
      f"{norm_error_lin_grad_cov=}\n",
      f"{error_mc2aug_var=}\n",
      f"{error_mc2prop_var=}"
      )

print(f"{Py_mc=}\n\n{Py=}\n\n{Py2=}\n\n{Py_lin=}")

sig_y, corr_y = get_corr_std_dev(Py)
sig_y2, corr_y2 = get_corr_std_dev(Py2)
sig_y_mc, corr_y_mc = get_corr_std_dev(Py_mc)

print(f"{sig_y_mc=}\n{sig_y=}\n{sig_y2=}")
print(f"{corr_y_mc=}\n{corr_y=}\n{corr_y2=}")
