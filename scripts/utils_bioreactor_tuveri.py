# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:47:07 2021

@author: halvorak
"""
import numpy as np
import scipy.stats
import casadi as cd
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import casadi as cd
import seaborn as sns
import copy
import sklearn.preprocessing
#Self-written modules
import sigma_points_classes as spc
# from myFilter import sigma_points as spc
import unscented_transformation as ut

# from time_out_manager import time_limit

font = {'size': 14}

matplotlib.rc('font', **font)


def ode_model_plant():
    
    #Make parameters
    mu_max = cd.MX.sym("mu_max", 1) # [1/h]
    K_S = cd.MX.sym("K_S", 1) # [g/L]
    Y_XS = cd.MX.sym("Y_XS", 1) # [g/g]
    Y_XCO2 = cd.MX.sym("Y_XCO2", 1) # [g/g]
    k_d = cd.MX.sym("k_d", 1) # [1/h]
    
    #"unscale" the parameters
    scaler_biopar = 1e3
    V0 = 4 # [L]
    mu_max_unsc = mu_max/scaler_biopar
    K_S_unsc = K_S/scaler_biopar
    Y_XS_unsc = Y_XS/scaler_biopar
    Y_XCO2_unsc = Y_XCO2/scaler_biopar
    k_d_unsc = k_d/scaler_biopar
    
    #"secondary" parameters
    S_in = cd.MX.sym("S_in", 1) # [g/L]? Concentration in the substrate
    q_air = cd.MX.sym("S_in", 1) # [g/L]? air flow
    
    #Integration parameters
    t_span = cd.MX.sym("t_span", 1)
    
    #States
    V = cd.MX.sym("V", 1)
    X = cd.MX.sym("X", 1)
    S = cd.MX.sym("S", 1)
    CO2 = cd.MX.sym("CO2", 1)
    
    #Inputs
    F_in = cd.MX.sym("F_in", 1)
    
    #ode system
    x_dot_0 = F_in #dV/dt
    x_dot_1 = -(F_in/V)*X + mu_max_unsc * (S/(K_S_unsc + S))*X - k_d_unsc*X #dXd/dt
    x_dot_2 = (F_in/V)*(S_in - S) - mu_max_unsc * (S/(K_S_unsc + S))*(X/Y_XS_unsc) #dS/dt
    x_dot_3 =  1/(V0 - V) * (mu_max_unsc * (S/(K_S_unsc + S))*(X/Y_XCO2_unsc)*V
                             - q_air*CO2
                             + F_in*CO2) #dCO2/dt #my own correction term, based on mass balance. Same as in the original article, see KRÄMER, D. & KING, R. 2019. A hybrid approach for bioprocess state estimation using NIR spectroscopy and a sigma-point Kalman filter. Journal of process control, 82, 91-104.

    
    #Concatenate equation, states, inputs and parameters
    diff_eq = cd.vertcat(x_dot_0, x_dot_1, x_dot_2, x_dot_3)
    x_var = cd.vertcat(V, X, S, CO2)
    u_var = cd.vertcat(F_in)
    p_var = cd.vertcat(mu_max,
                       K_S, 
                       Y_XS, Y_XCO2, k_d, S_in, q_air)
    p_aug_var = cd.vertcat(u_var, p_var, t_span)
    #Form ode dict and integrator
    ode = {"x": x_var,
           "p": p_aug_var,
           "ode": diff_eq*t_span}
    # opts = {"t0": t0, "tf": tf}
    opts = {#options for solver and casadi
            # "cvode": {#solver specific options
            #            "max_num_steps": 10}
            "max_num_steps": 200}
    # opts = { #options for solver and casadi
    #         # "print_time":0,#suppress print output from Casadi
    #         # "abstol": 1e-9, #Default 1e-2 according to https://www.neuron.yale.edu/neuron/static/new_doc/simctrl/cvode.html#CVode.atol
    #         # "reltol": 1e-3, #Default 1e-2 according to https://www.neuron.yale.edu/neuron/static/new_doc/simctrl/cvode.html#CVode.atol
    #         }
    # opts = { #options for solver and casadi
    #         # "print_time":0,#suppress print output from Casadi
    #         "cvodes": { #solver specific options
    #                    "atol": 1e-5, #Default 1e-2 according to https://www.neuron.yale.edu/neuron/static/new_doc/simctrl/cvode.html#CVode.atol
    #                           }
    #                 }
    # opts = { #options for solver and casadi
    #                 "print_time":0,#suppress print output from Casadi
    #                 "ipopt": { #solver specific options
    #                     # "mehrotra_algorithm": "yes",#performs better for convex QP according to https://projects.coin-or.org/CoinBinary/export/837/CoinAll/trunk/Installer/files/doc/Short%20tutorial%20Ipopt.pdf. On my test problem, it runs slower though
    #                     "hessian_constant": "yes", # Better for QP since IPOPT only gets the Hessian once (solves it as QP). https://coin-or.github.io/Ipopt/OPTIONS.html, search for QP
    #                     "print_level": 0 #suppresses print from IPOPT.
    #                           }
    #                 }
    # F = cd.integrator("F", "cvodes", ode, opts)
    # F = cd.integrator("F", "rk", ode)
    F = cd.integrator("F", "idas", ode)
    
    #Additional functions to keep similar output as Jose
    # res = F(x0=x_var, p = p_aug_var)
    # S_x = cd.Function('S',[x_var],[cd.jacobian(res["xf"],x_var)])
    # S_p = cd.Function('S',[p_aug_var],[cd.jacobian(res["xf"],x_var)])
    # print(S([ 1.5,  1.2, 20. ,  0. ]))
    # print(type(x_var))
    # S_xx = cd.gradient(F, x_var)
    # S_zz = cd.gradient(F, p_var)
    # jac_p = cd.Function("jac_p", [ode["x"], ode["p"]], cd.jacobian(ode["ode"], ode["p"]))
    jac_p_func = cd.jacobian(ode["ode"], ode["p"])
    jac_p = cd.Function("jac_p", [ode["x"], ode["p"]], [jac_p_func])
    # S_xx = None
    S_zz = None
    S_xz = None
    S_xp = None
    S_zp = None
    z_var = None
    alg = None
    obj_fun = None
    return F,jac_p,S_zz,S_xz,S_xp,S_zp,x_var,z_var,u_var,p_var,diff_eq,alg,obj_fun

def integrate_ode(F, x0, t_span, uk, par_fx):
    x0 = cd.vertcat(x0)
    pk = list(uk)
    pk.extend(list(par_fx.values()))
    pk.append(t_span[1] - t_span[0])
    # print(pk)
    Fend = F(x0 = x0, p = cd.vcat(pk))
    xf = Fend["xf"]
    xf_np = np.array(xf).flatten()
    return xf_np

def integrate_ode_parametric_uncertainty(F, x0, t_span, uk, par_fx, dim_par):
    
    xf_ode = integrate_ode(F, x0[dim_par:], t_span, uk, par_fx)
    xf = np.hstack((x0[:dim_par], xf_ode))
    return xf


def get_measurement_matrix(dim_x, par):
    radius = par["radius_tank"] #[m]
    A_tank_btm = area_circle(radius)
    rho = par["rho"]
    g = par["g"]
    H = np.zeros((3, dim_x))
    H[0,0] = 1
    H[1,1] = 1
    H[-1,-1] = 1
    # H[3,5] = 1
    # H[4,6] = ((rho*g/A_tank_btm)*
    #              1/1e3*#1000 m3/L
    #              1e3/1e5 #Pa to mbar
    #              )
    return H

def hx(x, par):#, v):
    #change this later? For volume, we assume dP measurement. From the venturi flowmeter for the subsea pump, we have an accuracy of 0,065%*span_DP*multiplier_accuracy/sigma_level =0,065/100*320mbar*0,5/2 = 0,052 mbar. 
    H = get_measurement_matrix(x.shape[0], par)
    y = np.dot(H, x)# + v
    # y[-1] = dp_measurement(y[-1], par)
    return y

def area_circle(r):
    return np.pi*np.square(r)
def height_cylinder(r, V=1e-3):
    return V/area_circle(r)
def dp_liquid_column(h, rho = 1e3):
    return rho*9.81*h


def dp_measurement(V, par):
    """
    Returns dP measurement in a tank filled with liquid

    Parameters
    ----------
    V : TYPE float
        DESCRIPTION. Volume in the tank [L]
    par : TYPE dict
        DESCRIPTION. Contains keys "radius" [m] and "rho" [kg/m3]

    Returns
    -------
    dp : TYPE float
        DESCRIPTION. dP in tank due to liquid column [mbar]

    """
    radius = par["radius_tank"] #[m]
    liquid_height = height_cylinder(radius, 
                                    V = V*1e-3 #[m3]
                                    )
    dp = dp_liquid_column(liquid_height, rho = par["rho"])/1e5*1e3 #[mbar]
    return dp
# # r = np.linspace(1,5)/1e2 #[m]
# r = 3/1e2 #[m]
# a = area_circle(r)

# # dp = dp_liquid_column(h)/1e5*1e3 #[mbar]
# p = dict(radius_tank = r, rho = 1000)
# V=np.linspace(.3,1)
# # V=np.zeros(50)
# V=np.ones(50)
# dp_meas = dp_measurement(V, p) + np.random.normal(scale = .052, size = V.shape[0])
# plt.scatter(V, dp_meas)
# h = height_cylinder(r, V=V*1e-3)
# plt.plot(r,dp)    

def sigma_repeatability():
    v_rep = np.sqrt(np.array([1e-2, 1., 1e-3]))
    return v_rep

def sigma_measurement(x, par):
    
    # y = x + alpha_y*x*sigma, noise is increasing with x. For NH3, alpha=6%
    # H = get_measurement_matrix(x.shape[0], par)
    sigma = sigma_repeatability()
    # sigma_det  = np.multiply(np.dot(H, x), par["alpha_y"])
    # sigma_det = np.multiply(hx(x), par["alpha_y"])
    # sigma = sigma + sigma_det
    return sigma

def vx(x, par):
    # cov = np.diag(np.square(sigma_measurement(x, par)))
    cov = np.diag(np.square(sigma_repeatability())) #simplified unitl things are worrking
    return scipy.stats.multivariate_normal(cov = cov).rvs()



def fx_for_UT_gen_Q(sigmas, list_of_keys, F, x, t_span, uk, par_fx):
    
    for i in range(len(list_of_keys)):
        key = list_of_keys[i]
        if not key in par_fx:
            raise KeyError("This key should be in par")
        par_fx[key] = sigmas[i]
    x_propagated = integrate_ode(F, x, t_span, uk, par_fx)
    return x_propagated

def fx_for_UT_gen_wbar_Q(sigmas, list_of_keys, t_span, x, par, w):
    for i in range(len(list_of_keys)):
        key = list_of_keys[i]
        if not key in par:
            raise KeyError("This key should be in par")
        par[key] = sigmas[i]
    yi = fx_ukf_ode(ode_model_plant,
                    t_span,
                    x,
                    args_ode = (w, par)
                    )
    # print(yi)
    return yi

def hx_for_UT_gen_R(sigmas, list_of_keys, x, par):
    
    for i in range(len(list_of_keys)):
        key = list_of_keys[i]
        if not key in par:
            raise KeyError("This key should be in par")
        par[key] = sigmas[i]
    yi = hx(x, par)
    # yi = np.array([hx(x, par)])
    # print(yi)
    return yi
        
    

def fx_ukf_ode(ode_model, t_span, x0, args_ode = None, args_solver = {}):
    res = scipy.integrate.solve_ivp(ode_model,
                                    t_span,
                                    x0,
                                    args = args_ode,
                                    **args_solver)
    x_all = res.y
    x_final = x_all[:, -1]
    return x_final

def get_x0_literature():
    # x0 = np.array([1.5, # [], V
    #                1.2, # [], X 
    #                20, # [], S 
    #                .0 # [], CO2 
    #                ])
    x0 = np.array([1.5, # [], V
                   4, # [], X 
                   20, # [], S 
                   .015 # [], CO2 
                   ])
    return x0

def get_x0_kf_literature(sigma_multiplier = 0.):
    x0 = get_x0_literature()
    P0 = get_P0_literature()
    x0_kf = x0 + np.sqrt(np.diag(P0))*sigma_multiplier #same as x0 by default 
    return x0_kf

def get_x0_kf_random(eps = 1e-4):
    """
    Selects starting point for UKF, x0_ukf, from a random draw from a multivariate normal distribution with mean x0_true and covariance P0 subject to all values in x0_ukf > eps

    Returns
    -------
    x0_kf : TYPE np.array((dim_x,))
        DESCRIPTION. Starting point for the UKF.

    """
    x0 = get_x0_literature()
    P0 = get_P0_literature()
    all_positive = False
    while not all_positive:
        x0_kf = np.random.multivariate_normal(x0, P0)
        all_positive = all(x0_kf >= 1e-4)
    return x0_kf

def get_P0_literature():
    #Values from Tuveri
    # P0 = np.diag([2.09e-8, # [], V
    #                1.1e-5, # [], X 
    #                1.09e-4, # [], S 
    #                2.17e-5 # [], CO2 
    #                ])
    #My own values
    P0 = np.diag(np.square([.1, # [L], V ==> [L**2] when squared
                           1.5, # [g/L], X ==> etc
                           1., # [g/L], S 
                           .005 # [%?], CO2 
                           ]))
    # P0 = np.diag(np.square([.05, # [L], V ==> [L**2] when squared. This works
    #                        .5, # [g/L], X ==> etc
    #                        1., # [g/L], S 
    #                        .1 # [%?], CO2 
    #                        ]))
    return P0



def get_u0_literature():
    # u0 = np.array([1e-8 # [], F_in
    #                ])
    u0 = np.array([0 # [], F_in
                   ])
    return u0


def nearestPD(A):
    """Find the nearest positive-definite matrix to input --taken from https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194--
    

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def nearest_well_conditioned_corr_mat(A, condition_number_accept = 1e-4):
    """Find the nearest well conditioned correlation matrix from a covariance matrix. Very much based on the "nearestPD" function, just with a different check.
    

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2
    corr = correlation_from_covariance(A3)
    if is_well_conditioned(corr, condition_number_accept = condition_number_accept):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not is_well_conditioned(corr, condition_number_accept = condition_number_accept):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
        corr = correlation_from_covariance(A3)

    return A3

def is_well_conditioned(A, condition_number_accept = 1e-4):
    kappa = np.linalg.cond(A)
    if kappa >= condition_number_accept:
        return True
    else:
        return False

def correlation_from_covariance(cov):
    """
    Calculate correlation matrix from a covariance matrix

    Parameters
    ----------
    cov : TYPE np.array((dim_p, dim_p))
        DESCRIPTION. Covariance matrix

    Returns
    -------
    corr : TYPE np.array((dim_p, dim_p))
        DESCRIPTION. Correlation matrix

    """
    sigmas = np.sqrt(np.diag(cov))
    dim_p = sigmas.shape[0]
    
    #Create sigma_mat = [[s1, s1 ,.., s1],
    # [s2, s2,...,s2],
    # [s_p, s_p,..,s_p]]
    sigma_mat = np.tile(sigmas.reshape(-1,1), dim_p)
    sigma_cross_mat = np.multiply(sigma_mat, sigma_mat.T)
    # print(f"sigmas: {sigmas}\n",
    #       f"sigma_mat: {sigma_mat}\n",
    #       f"sigma_cross_mat: {sigma_cross_mat}")
    corr = np.divide(cov, sigma_cross_mat) #element wise division
    return corr, sigmas

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        # _ = np.linalg.cholesky(B) #original test
        #This test is taken from scipy.stats.multivariate_normal (or actually from the _PSD class there)
        _ = scipy.stats._multivariate._PSD(B, allow_singular=False)
        
        return True
    except (np.linalg.LinAlgError, ValueError):
        return False

def get_literature_values(N_samples = int(5e3), plot_par = False, plt_kwargs = {}):
    plt_kwargs["corner"] = True #always true
    #Nominal parameter values
    par_mean_fx_multivar = {
        "mu_max": .19445, # [1/h]
        "K_S": .007, # [g/L]
        "Y_XS": .42042, # [g/g]
        "Y_XCO2": .54308, # [g/g]
        "k_d": .006 # [1/h]
        }
    # names_par = [r"$\mu_{max}$", r"$K_{S}$", r"$Y_{XS}$", r"$Y_{XCO_2}$", r"$k_{d}$"]
    names_par = [r"$\mu_{max}$ [1/h]", r"$K_{S}$ [g/L]", r"$Y_{XS}$ [g/g]", r"$Y_{XCO_2}$ [g/g]", r"$k_{d}$ [1/h]"]
    
    # par_cov_fx_multivar = 1e-10*np.array( #Andrea's values
    #     [[.1054,    -.1100,     -.0846,     .0570,  -.0898],
    #       [0,        .01537,     .0508,      -.070,  .0489],
    #       [0,        0,          .1282,      -.0267, .0813],
    #       [0,        0,          0,          .0491, -.0657],
    #       [0,        0,          0,          0,      .2020]
    #        ])
    # par_cov_fx_multivar = 1e-5*np.array( #works well
    #     [[.1054,    -.1100,     -.0846,     .0570,  -.0898],
    #       [0,        .01537,     .0508,      -.070,  .0489],
    #       [0,        0,          .1282,      -.0267, .0813],
    #       [0,        0,          0,          .0491, -.0657],
    #       [0,        0,          0,          0,      .2020]
    #       ])
    # par_cov_fx_multivar = 1e-3*np.array( #works well 190122
    #     [[.1054,    -.1100,     -.0846,     .0570,  -.0898],
    #       [0,        .01537,     .0508,      -.070,  .0489],
    #       [0,        0,          .1282,      -.0267, .0813],
    #       [0,        0,          0,          .0491, -.0657],
    #       [0,        0,          0,          0,      .2020]
    #       ])
    # # par_cov_fx_multivar = 1e-5*np.array( #correlations destroyed
    # #     [[1.054,    -.1100,     -.846,     .570,  -.0898],
    # #       [0,        .01537,     .508,      -.70,  .0489],
    # #       [0,        0,          12.82,      -20.67, 8.13],
    # #       [0,        0,          0,          4.91, -.657],
    # #       [0,        0,          0,          0,      .2020]
    # #       ])
    # # par_cov_fx_multivar[1, :] = par_cov_fx_multivar[1, :]*(3.38e-6/par_cov_fx_multivar[1, 1]) #K_S in MU phase in Tuveri's paper
    # # par_cov_fx_multivar[2, :] = par_cov_fx_multivar[2, :]*(4.91e-3 / par_cov_fx_multivar[2, 2]) #Y_XCO2 in MU phase in Tuveri's paper
    # # par_cov_fx_multivar[2, 2] = 4.91e-3 #Y_XCO2 in MU phase in Tuveri's paper
    
    # sigma_par_fx = np.sqrt(np.diag(par_cov_fx_multivar))
    # rel_sigma = sigma_par_fx/np.array(list(par_mean_fx_multivar.values()))
    # # print(f"rel_sigma_par: {rel_sigma*100} [%]")
    
    # par_cov_fx_multivar = par_cov_fx_multivar.T + par_cov_fx_multivar
    # np.fill_diagonal(par_cov_fx_multivar, np.diag(par_cov_fx_multivar)/2)
    
    
    # if not isPD(par_cov_fx_multivar):
    #     # print("par_cov_fx !>=0. Using nearestPD(par_cov_fx)")
    #     par_cov_fx_multivar = nearestPD(par_cov_fx_multivar) #returns the postive definite matrix closest to the input matrix, as measured by the Frobenius norm
    
    # cov_diag = np.diag(par_cov_fx_multivar)
    # # corr_before = correlation_from_covariance(par_cov_fx_multivar)
    # par_cov_fx_multivar = par_cov_fx_multivar + np.diag(cov_diag)*.05 #add 5% on the diagonal, should make it more robust ("more" positive definite)
    
    # # print(par_cov_fx_multivar)
    par_cov_fx_multivar = np.array([[ 1.25776679e-04, -8.41424781e-05, -8.48131358e-05, 6.68376093e-05, -8.72403997e-05],
     [-8.41424781e-05,  6.49344102e-05,  5.04169428e-05, -5.23194085e-05, 5.35002281e-05],
     [-8.48131358e-05,  5.04169428e-05,  1.34613316e-04, -2.68457358e-05, 8.12620817e-05],
     [ 6.68376093e-05, -5.23194085e-05, -2.68457358e-05,  5.86179931e-05, -6.39498228e-05],
     [-8.72403997e-05,  5.35002281e-05,  8.12620817e-05, -6.39498228e-05, 2.12578139e-04]])
    # corr_after, std_dev = correlation_from_covariance(par_cov_fx_multivar)
    # # print(f"corre_before: {corr_before}\n\n",
    # #       f"corr_after: {corr_after}\n\n",
    # #       f"corr-b-corr_a: {np.linalg.norm(corr_before-corr_after)}")
    # print(f"corr_after: {corr_after}")
    # print(f"std_dev: {std_dev}")
    # print(f"mean: {list(par_mean_fx_multivar.values())}")
    
    # corr_fx = correlation_from_covariance(par_cov_fx_multivar)
    # if not is_well_conditioned(corr_fx):
    #     # print("par_cov_fx !>=0. Using nearestPD(par_cov_fx)")
    #     par_cov_fx_multivar2 = nearest_well_conditioned_corr_mat(par_cov_fx_multivar) #returns the postive definite matrix closest to the input matrix, as measured by the Frobenius norm
    #     print(f"norm:cov2-cov1: {np.linalg.norm(par_cov_fx_multivar2-par_cov_fx_multivar)}")
    #     corr_fx = correlation_from_covariance(par_cov_fx_multivar2)
    
    # print(f"cov_cond: {np.linalg.cond(par_cov_fx_multivar)}\n",
    #       f"corr_cond: {np.linalg.cond(corr_fx)}")
    
    # if not isPD(par_cov_fx_multivar2):
    #     # print("par_cov_fx !>=0. Using nearestPD(par_cov_fx)")
    #     par_cov_fx_multivar3 = nearestPD(par_cov_fx_multivar2) #returns the postive definite matrix closest to the input matrix, as measured by the Frobenius norm
    #     print(f"norm cov3-cov1: {np.linalg.norm(par_cov_fx_multivar3-par_cov_fx_multivar)}")
    #     corr_fx = correlation_from_covariance(par_cov_fx_multivar3)
    # print(f"cov_cond: {np.linalg.cond(par_cov_fx_multivar)}\n",
    #       f"corr_cond: {np.linalg.cond(corr_fx)}")
    
    
    par_mean_fx_univar = {
        "S_in": 100., # [g/L]
        "q_air": 2.*60 #[NL/min] => [NL/h]
        }
    # names_par.extend([r"$S_{in}$", r"$q_{air}$"])
    names_par.extend([r"$S_{in}$ [g/L]", r"$q_{air}$ [NL/h]"])
    par_sigma_fx_univar = {
        "S_in": 5, # [g/L]
        # "q_air": 2*.05*60 #[NL/min] => [NL/h]
        "q_air": 1.5*.05*60 #[NL/min] => [NL/h]
        }
    
    par_dist_spec_fx_univar = {} #specify distributions for the parameters here
    for key, var_coef in par_sigma_fx_univar.items():
        par_dist_spec_fx_univar[key] = {"dist": "gamma",
                                  "kwargs": {} #if e.g. student-t, we need more arguments
                                  }
    
    par_mean_hx_univar = dict(radius_tank = 3/100, #[m]
                        rho = 1000, #[kg/m3], density of liquid in reactor
                        g = 9.81 #[m/s2]
                        )
    par_sigma_hx_univar = dict(radius_tank = .5/1000 #[m] (mm std dev)
                        )
    par_dist_spec_hx_univar = {"radius_tank": {"dist": "gamma",
                                        "kwargs": {}}
                        }
    
    par_dist_fx = {}
    par_scaling_fx = {"scaler_biopar": 1e3 #scaler for bioparameters to get better condition number
                      }
    par_det_fx = {}
    par_dist_hx = {}
    par_det_hx = {}
    
    #All parameters which are not in par_dist_spec_fx are deterministic. Add them to dict
    for par_name, val in par_mean_fx_univar.items():
        if not par_name in par_dist_spec_fx_univar:
            par_det_fx[par_name] = val
    for par_name, val in par_mean_hx_univar.items():
        if not par_name in par_dist_spec_hx_univar:
            par_det_hx[par_name] = val
    
    #Set distributions from scipy.stats for fx
    for par_name, dist_spec in par_dist_spec_fx_univar.items():
        if dist_spec["dist"] == "norm":
            par_dist_fx[par_name] = scipy.stats.norm(loc = par_mean_fx_univar[par_name],
                                                  scale = par_sigma_fx_univar[par_name])
        elif dist_spec["dist"] == "gamma":
            alpha, loc, beta = get_param_gamma_dist(par_mean_fx_univar[par_name],
                                                    par_sigma_fx_univar[par_name], 
                                                    num_std = 2)
            par_dist_fx[par_name] = scipy.stats.gamma(alpha, loc = loc, scale = 1/beta)
            # print(f"par_name: {par_name}\n",
            #       f"hyperpar: {alpha}, {loc}, {beta}")
        
        elif dist_spec["dist"] == "student_t":
            dof = dist_spec["kwargs"]["dof"]
            par_dist_fx[par_name] = scipy.stats.t(dof, 
                                                loc = par_mean_fx_univar[par_name],
                                                scale = par_sigma_fx_univar[par_name])
            # data = scipy.stats.norm(loc = par_mean_fx[par_name], 
            #                         scale = par_sigma_fx[par_name])
            # t_fitted = scipy.stats.t.fit(data)
            # par_dist[par_name] = scipy.stats.t(dof, 
            #                                     loc = par_mean_fx[par_name])
        else:
            raise KeyError("Have not implemented this distribution")
        
    #Set distributions from scipy.stats for hx
    for par_name, dist_spec in par_dist_spec_hx_univar.items():
        if dist_spec["dist"] == "norm":
            par_dist_hx[par_name] = scipy.stats.norm(loc = par_mean_hx_univar[par_name],
                                                  scale = par_sigma_hx_univar[par_name])
        elif dist_spec["dist"] == "gamma":
            alpha, loc, beta = get_param_gamma_dist(par_mean_hx_univar[par_name],
                                                    par_sigma_hx_univar[par_name], 
                                                    num_std = 2)
            par_dist_hx[par_name] = scipy.stats.gamma(alpha, loc = loc, scale = 1/beta)
        
        elif dist_spec["dist"] == "student_t":
            dof = dist_spec["kwargs"]["dof"]
            par_dist_hx[par_name] = scipy.stats.t(dof, 
                                                loc = par_mean_hx_univar[par_name],
                                                scale = par_sigma_hx_univar[par_name])
            # data = scipy.stats.norm(loc = par_mean_hx[par_name], 
            #                         scale = par_sigma_hx[par_name])
            # t_fitted = scipy.stats.t.fit(data)
            # par_dist[par_name] = scipy.stats.t(dof, 
            #                                     loc = par_mean_hx[par_name])
        else:
            raise KeyError("Have not implemented this distribution")
        
    #Draw samples from these distributions
    dist_multivar = scipy.stats.multivariate_normal(mean = np.array(list(par_mean_fx_multivar.values()))*par_scaling_fx["scaler_biopar"],
                                                        cov = par_cov_fx_multivar*par_scaling_fx["scaler_biopar"]**2)
    par_samples_fx = [dist_multivar.rvs(size = int(3*N_samples)).T]
    par_names_fx = list(par_mean_fx_multivar.keys())
    mode_fx = {}
    for key, dist in par_dist_fx.items(): #samples from univariate distribution
        par_samples_fx.append(dist.rvs(size = int(3*N_samples)).reshape(1,-1))
        par_names_fx.append(key)
        # mode_fx = 
    par_samples_hx = []
    par_names_hx = []
    for key, dist in par_dist_hx.items(): #samples from univariate distribution
        par_samples_hx.append(dist.rvs(size = int(3*N_samples)).reshape(1,-1))
        par_names_hx.append(key)
    
    #concatenate the samples
    par_samples_fx = np.vstack(par_samples_fx)
    par_samples_hx = np.vstack(par_samples_hx)
    
    #remove values below zero, and get correct dimensionality
    epsilon_sample = 1e-7
    idx_keep = (par_samples_fx > epsilon_sample).all(axis=0)
    par_samples_fx = par_samples_fx[:, idx_keep] #keep obly positive parameters
    par_samples_fx = par_samples_fx[:, :N_samples] #get correct dimension
    # par_samples_hx = par_samples_hx[:, (par_samples_hx < 0).any(axis=0)] #same for hx
    par_samples_hx = par_samples_hx[:, :N_samples] 
    #Do plotting if required
    plt_output = []
    if plot_par:
        # par_samples_fx_unscaled = par_samples_fx.T/par_scaling_fx["scaler_biopar"]
        # print(par_samples_fx_unscaled.shape)
        # print(names_par)
        df_fx = pd.DataFrame(data = par_samples_fx.T/par_scaling_fx["scaler_biopar"], columns = names_par)
        df_hx = pd.DataFrame(data = par_samples_hx.T, columns = par_names_hx)
        font = {'size': 16}
        matplotlib.rc('font', **font)
        sns_grid_fx = sns.pairplot(df_fx, **plt_kwargs)
        plt.tight_layout()
        sns_grid_hx = sns.pairplot(df_hx, **plt_kwargs)
        plt_output = [sns_grid_fx, sns_grid_hx]
    
    #Kalman filter values in the description
    Q = np.diag(np.ones(4)*1e-2)
    #Measurement noise
    R = np.diag(np.square(sigma_repeatability())) #assume this initially
    return par_samples_fx, par_samples_hx, par_names_fx, par_names_hx, par_det_fx, par_det_hx, Q, R, plt_output, par_dist_fx, par_scaling_fx


def compute_performance_index_valappil(x_kf, x_ol, x_true, cost_func = "RMSE"):
    if cost_func == "RMSE":
        J = np.linalg.norm(x_kf - x_true, axis = 1, ord = 2)
    elif cost_func == "valappil": #valappil's cost index
        J = np.divide(np.linalg.norm(x_kf - x_true, axis = 1, ord = 2),
                      np.linalg.norm(x_ol - x_true, axis = 1, ord = 2))
    else:
        raise ValueError("cost function is wrongly specified. Must be RMSE or valappil.")
    return J

def truth_within_estimate(x_true, x_kf, P_diag, sigma_multiplier = 1):
    """
    Checks if the true value is within the confidence band for the estimator

    Parameters
    ----------
    x_true : TYPE np.array((dim_x, dim_t))
        DESCRIPTION. True value
    x_kf : TYPE np.array((dim_x, dim_t))
        DESCRIPTION. Estimated value
    P_diag : TYPE np.array((dim_x, dim_t))
        DESCRIPTION. Diagonal elements of covariance matrix over time
    sigma_multiplier : TYPE, optional int
        DESCRIPTION. The default is 1. x_kf - sigma_multiplier*std <= x_true <= x_kf + sigma_multiplier*std

    Returns
    -------
    None.

    """
    std = np.sqrt(P_diag)
    below = (x_kf - sigma_multiplier*std <= x_true)
    above = (x_true <= x_kf + sigma_multiplier*std)
    within_band = below*above
    # within_band = ((x_kf - sigma_multiplier*std <= x_true) and 
    #                (x_true <= x_kf + sigma_multiplier*std))
    return within_band

def get_param_ukf_case1(plot_dist = True):
    
    par_dist_fx = {}
    par_det_fx = {}
    par_dist_hx = {}
    par_det_hx = {}
    
    par_mean_fx, par_sigma_fx, par_dist_spec_fx, par_mean_hx, par_sigma_hx, par_dist_spec_hx, Q, R = get_literature_values()
    
    #All parameters which are not in par_dist_spec_fx are deterministic. Add them to dict
    for par_name, val in par_mean_fx.items():
        if not par_name in par_dist_spec_fx:
            par_det_fx[par_name] = val
    for par_name, val in par_mean_hx.items():
        if not par_name in par_dist_spec_hx:
            par_det_hx[par_name] = val
    
    #Set distributions from scipy.stats for fx
    for par_name, dist_spec in par_dist_spec_fx.items():
        if dist_spec["dist"] == "norm":
            par_dist_fx[par_name] = scipy.stats.norm(loc = par_mean_fx[par_name],
                                                  scale = par_sigma_fx[par_name])
        elif dist_spec["dist"] == "gamma":
            alpha, loc, beta = get_param_gamma_dist(par_mean_fx[par_name],
                                                    par_sigma_fx[par_name], 
                                                    num_std = 2)
            par_dist_fx[par_name] = scipy.stats.gamma(alpha, loc = loc, scale = 1/beta)
        
        elif dist_spec["dist"] == "student_t":
            dof = dist_spec["kwargs"]["dof"]
            par_dist_fx[par_name] = scipy.stats.t(dof, 
                                                loc = par_mean_fx[par_name],
                                                scale = par_sigma_fx[par_name])
            # data = scipy.stats.norm(loc = par_mean_fx[par_name], 
            #                         scale = par_sigma_fx[par_name])
            # t_fitted = scipy.stats.t.fit(data)
            # par_dist[par_name] = scipy.stats.t(dof, 
            #                                     loc = par_mean_fx[par_name])
        else:
            raise KeyError("Have not implemented this distribution")
        
    #Set distributions from scipy.stats for hx
    for par_name, dist_spec in par_dist_spec_hx.items():
        if dist_spec["dist"] == "norm":
            par_dist_hx[par_name] = scipy.stats.norm(loc = par_mean_hx[par_name],
                                                  scale = par_sigma_hx[par_name])
        elif dist_spec["dist"] == "gamma":
            alpha, loc, beta = get_param_gamma_dist(par_mean_hx[par_name],
                                                    par_sigma_hx[par_name], 
                                                    num_std = 2)
            par_dist_hx[par_name] = scipy.stats.gamma(alpha, loc = loc, scale = 1/beta)
        
        elif dist_spec["dist"] == "student_t":
            dof = dist_spec["kwargs"]["dof"]
            par_dist_hx[par_name] = scipy.stats.t(dof, 
                                                loc = par_mean_hx[par_name],
                                                scale = par_sigma_hx[par_name])
            # data = scipy.stats.norm(loc = par_mean_hx[par_name], 
            #                         scale = par_sigma_hx[par_name])
            # t_fitted = scipy.stats.t.fit(data)
            # par_dist[par_name] = scipy.stats.t(dof, 
            #                                     loc = par_mean_hx[par_name])
        else:
            raise KeyError("Have not implemented this distribution")
        
    
    if plot_dist: #only fx dists so far
        dim_dist = len(par_dist_fx)
        nrows = int(np.ceil(np.sqrt(dim_dist)))
        fig, ax = plt.subplots(nrows, nrows)
        
    
    # if plot_dist:
    #     # par_dist_fx.pop("k")
    #     dim_dist = len(par_dist_fx)
    #     fig, ax = plt.subplots(dim_dist, 1)
        if dim_dist == 1:
            ax = [ax]
        i = 0
        row = 0
        col = 0
        for key, dist in par_dist_fx.items():
            #compute the mode numerically, as dist.mode() is not existing
            mode = scipy.optimize.minimize(lambda x: -dist.pdf(x),
                                            dist.mean(),
                                            tol = 1e-10)
            mode = mode.x
            # print(key, mode, dist.mean())
            x = np.linspace(dist.ppf(1e-3), dist.ppf(.999), 100)
            
            ax[row][col].plot(x, dist.pdf(x), label = "pdf")
            ax[row][col].set_xlabel(key)
            ax[row][col].set_ylabel("pdf")
            ylim = ax[row][col].get_ylim()
            
            # ax[row][col].plot([par_mean[key], par_mean[key]], [ylim[0], ylim[1]], 
            #         label = "nom")
            # ax[row][col].plot([dist.mean(), dist.mean()], [ylim[0], ylim[1]], 
            #         linestyle = "dashed", label = "Mean")
            # ax[row][col].plot([dist.mean()*(1-std_dev_prct), dist.mean()*(1-std_dev_prct)], [ylim[0], ylim[1]], 
            #         label = "Mean-std_lit")
            # ax[row][col].plot([dist.mean() - dist.std(), dist.mean() - dist.std()], [ylim[0], ylim[1]], 
            #         linestyle = "dashed", label = "Mean-std_gamma")
            # ax[row][col].plot([mode, mode], [ylim[0], ylim[1]], label = "Mode")
            
            
            ax[row][col].scatter(par_mean_fx[key], dist.pdf(par_mean_fx[key]), label = r"$\mu = \mu_{lit} = \theta_{UKF}$")
            ax[row][col].scatter([par_mean_fx[key] - dist.std(), par_mean_fx[key] + dist.std()], 
                              [dist.pdf(par_mean_fx[key] - dist.std()), dist.pdf(par_mean_fx[key] + dist.std())], label = r"$\mu \pm \sigma = \mu_{lit} \pm \sigma_{lit}$")
            ax[row][col].scatter(mode, dist.pdf(mode), label = r"$\theta_{true}$")
            ndist = 2
            ax[row][col].scatter(dist.mean() - ndist*dist.std(), 
                          dist.pdf(dist.mean() - ndist*dist.std()), 
                          label = r"$\mu_{lit} - 2\sigma_{lit}$")
            
            # ax[row][col].plot([dist.mode(), dist.mode()], [ylim[0], ylim[1]], 
            #         linestyle = "dashed", label = "Mean")
            ax[row][col].set_ylim(ylim)
            # ax[row][col].legend()
            i += 1
            col += 1
            if col >= nrows:
                col = 0
                row += 1
            
        #plot hx
        dim_dist_h = len(par_dist_hx)
        fig_h, ax_h = plt.subplots(dim_dist_h, 1)
        if dim_dist_h == 1:
            ax_h = [ax_h]
        i = 0
        for key, dist in par_dist_hx.items():
            #compute the mode numerically, as dist.mode() is not existing
            mode = scipy.optimize.minimize(lambda x: -dist.pdf(x),
                                            dist.mean(),
                                            tol = 1e-10)
            mode = mode.x
            # print(key, mode, dist.mean())
            x = np.linspace(dist.ppf(1e-3), dist.ppf(.999), 100)
            ax_h[i].plot(x, dist.pdf(x), label = "pdf")
            ax_h[i].set_xlabel(key)
            ax_h[i].set_ylabel("pdf")
            ylim = ax_h[i].get_ylim()
            
            # ax_h[i].plot([par_mean[key], par_mean[key]], [ylim[0], ylim[1]], 
            #         label = "nom")
            # ax_h[i].plot([dist.mean(), dist.mean()], [ylim[0], ylim[1]], 
            #         linestyle = "dashed", label = "Mean")
            # ax_h[i].plot([dist.mean()*(1-std_dev_prct), dist.mean()*(1-std_dev_prct)], [ylim[0], ylim[1]], 
            #         label = "Mean-std_lit")
            # ax_h[i].plot([dist.mean() - dist.std(), dist.mean() - dist.std()], [ylim[0], ylim[1]], 
            #         linestyle = "dashed", label = "Mean-std_gamma")
            # ax_h[i].plot([mode, mode], [ylim[0], ylim[1]], label = "Mode")
            
            
            ax_h[i].scatter(par_mean_hx[key], dist.pdf(par_mean_hx[key]), label = r"$\mu = \mu_{lit} = \theta_{UKF}$")
            ax_h[i].scatter([par_mean_hx[key] - dist.std(), par_mean_hx[key] + dist.std()], 
                              [dist.pdf(par_mean_hx[key] - dist.std()), dist.pdf(par_mean_hx[key] + dist.std())], label = r"$\mu \pm \sigma = \mu_{lit} \pm \sigma_{lit}$")
            ax_h[i].scatter(mode, dist.pdf(mode), label = r"$\theta_{true}$")
            ndist = 2
            ax_h[i].scatter(dist.mean() - ndist*dist.std(), 
                          dist.pdf(dist.mean() - ndist*dist.std()), 
                          label = r"$\mu_{lit} - 2\sigma_{lit}$")
            
            # ax_h[i].plot([dist.mode(), dist.mode()], [ylim[0], ylim[1]], 
            #         linestyle = "dashed", label = "Mean")
            ax_h[i].set_ylim(ylim)
            ax_h[i].legend()
            i += 1
    else:
        fig, ax, fig_h, ax_h = None, None, None, None
    return par_dist_fx, par_det_fx, par_dist_hx, par_det_hx, [fig, fig_h], [ax, ax_h]
    

def get_param_gamma_dist(mean, std_dev, num_std = 3):
    
    ##For making (mean_gamma = mean) AND (var_gamma = std_dev**2)
    loc = mean - std_dev*num_std
    alpha = num_std**2
    beta = num_std/std_dev
    
    #For putting (mode= mean-std_dev) AND 
    # loc = mean - std_dev*num_std
    # beta = 1/std_dev
    # # alpha = num_std#*std_dev**2
    # alpha = beta**2*std_dev**2#*std_dev**2
    return alpha, loc, beta

def get_sigmapoints_and_weights(par_in, samples = False):
    """
    Returns sigma points and weights for the distributions in the container par_dist.

    Parameters
    ----------
    par_in : TYPE list, dict, a container which is iterable by for loop. len(par_in) = n
        DESCRIPTION. each element contains a scipy.dist
    samples : TYPE optional, default is False. Boolean
        DESCRIPTION. If False, then par_in is a container (dict, list) which contains scipy.stats.dist instances. If True, par_in containts samples (typically posterior samples from Bayesian parameter estimation)

    Returns
    -------
    sigmas : TYPE np.array((n, (2n+1)))
        DESCRIPTION. (2n+1) sigma points
    w : TYPE np.array(2n+1,)
        DESCRIPTION. Weight for every sigma point

    """
    
    n = len(par_in) #dimension of parameters
    
    # Compute the required statistics
    if not samples:
        mean = np.array([dist.mean() for k, dist in par_in.items()])
        cov = np.diag([dist.var() for k, dist in par_in.items()])
        cm3 = np.array([scipy.stats.moment(dist.rvs(size = int(1e6)), moment = 3) 
                        for k, dist in par_in.items()]) #3rd central moment
        cm4 = np.array([scipy.stats.moment(dist.rvs(size = int(1e6)), moment = 4) 
                        for k, dist in par_in.items()]) #4th central moment
    else: #samples == True
        mean = np.mean(par_in, axis = 1)
        cov = np.cov(par_in)
        cm3 = scipy.stats.moment(par_in, moment = 3, axis = 1)
        cm4 = scipy.stats.moment(par_in, moment = 4, axis = 1)
    
    # print(f"mean: {mean}\n",
    #       f"var: {var}\n",
    #       f"cm3: {cm3}\n",
    #       f"cm4: {cm4}")
    
    #Generate sigma points
    sigma_points = spc.GenUTSigmaPoints(n) #initialize the class
    s, w = sigma_points.compute_scaling_and_weights(cov,  #generate scaling and weights
                                                    cm3, 
                                                    cm4)
    try:
        sigmas, P_sqrt = sigma_points.compute_sigma_points(mean, #sigma points and P_sqrt
                                                            cov, 
                                                            s)
    except:
        sigmas, w, _, P_sqrt = sigma_points.compute_sigma_points(mean, cov, S = cm3, K = cm4, s1 = None, sqrt_method = None)
    return sigmas, w
def get_sigmapoints_and_weights_julier(par_in, samples = False, kappa = 0.):
    """
    Returns sigma points and weights for the distributions in the container par_in. Sigma point and weights are made on the assumption that these distributions are symmetrical around the mean (but mean and covariance are calculated based on the distributions in the container par_in)

    Parameters
    ----------
    par_in : TYPE list, dict, a container which is iterable by for loop. len(par_in) = n
        DESCRIPTION. each element contains a scipy.dist
    samples : TYPE optional, default is False. Boolean
        DESCRIPTION. If False, then par_in is a container (dict, list) which contains scipy.stats.dist instances. If True, par_in containts samples (typically posterior samples from Bayesian parameter estimation)

    Returns
    -------
    sigmas : TYPE np.array((n, (2n+1)))
        DESCRIPTION. (2n+1) sigma points
    w : TYPE np.array(2n+1,)
        DESCRIPTION. Weight for every sigma point

    """
    
    n = len(par_in) #dimension of parameters
    
    # Compute the required statistics
    if not samples:
        mean = np.array([dist.mean() for k, dist in par_in.items()])
        cov = np.diag([dist.var() for k, dist in par_in.items()])
    else: #samples == True
        mean = np.mean(par_in, axis = 1)
        cov = np.cov(par_in)
    
    #Generate sigma points
    sigma_points = spc.JulierSigmaPoints(n, kappa = kappa) #initialize the class
    sigmas, P_sqrt = sigma_points.compute_sigma_points(mean, 
                                               cov)
    w = sigma_points.compute_weights()
    return sigmas, w

def get_lhs_points(dist_dict, N_lhs_dist = 10, plot_mc_samples = False, labels = None):
    """
    

    Parameters
    ----------
    dist_dict : TYPE dict
        DESCRIPTION. {"a": scipy.stats.gamma(alpha, beta),
                      "b": scipy.stats.lognormal(mu, cov)}
    N_lhs_dist : TYPE, optional iny
        DESCRIPTION. The default is 10. number of LHS samples
    plot_mc_samples : TYPE, optional bool
        DESCRIPTION. The default is False. Plot or not
    labels : TYPE, optional list
        DESCRIPTION. The default is None.

    Returns
    -------
    x_lhs : TYPE
        DESCRIPTION.
    sample : TYPE
        DESCRIPTION.
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    """
    dim_dist = len(dist_dict)
    sampler = scipy.stats.qmc.LatinHypercube(d = dim_dist)
    # mean_lhs = np.zeros(N_lhs_dist))
    # var_lhs = np.zeros(len(N_lhs_dist))
    # for k in range(len(N_lhs)):
    sample = sampler.random(N_lhs_dist) #samples are done in the CDF (cumulative distribution function)
        
    #convert to values in the state domain
    x_lhs = {}
    x_rand = {}
    i = 0
    for key, dist in dist_dict.items():
        x_lhs[key] = dist.ppf(sample[:, i])#.reshape(-1,1)
        x_rand[key] = dist.rvs(size = N_lhs_dist)
        i += 1
    if not plot_mc_samples:
        fig, ax = (None, None)
    else: #do plot
        if labels is None:
            labels = ["" for k in range(dim_dist)]
        
        #A plot of how the points distribute in the pdf of the dist
        fig_pdf, ax_pdf = plt.subplots(dim_dist, 1)
        if dim_dist == 1:
            ax_pdf = [ax_pdf]
        # labels = [r"$\rho$ $[kg/m^3]$", "d [m]"]
        line_color = []
        i = 0
        for key, dist in dist_dict.items(): #plot the pdf of each distribution
            x_dummy = np.linspace(dist.ppf(.001), dist.ppf(.999), 100)    
            l = ax_pdf[i].plot(x_dummy, dist.pdf(x_dummy), label = "pdf")
            ax_pdf[i].scatter(x_lhs[key], np.zeros(N_lhs_dist), 
                          label = "LHS",
                          marker = "x")
            # ax_pdf[i].scatter(x_rand[key], np.zeros(N_lhs_dist), label = "rvs")
            ax_pdf[i].set_xlabel(labels[i])
            ax_pdf[i].set_ylabel("pdf")
            i += 1
            
        #Also a plot of the CDF + LHS
        fig_cdf, ax_cdf = plt.subplots(dim_dist, 1)
        if dim_dist == 1:
            ax_cdf = [ax_cdf]
        grid_cdf = np.linspace(0,.999,num = N_lhs_dist+1)
        i = 0
        for key, dist in dist_dict.items(): #plot the pdf of each distribution
            x_dummy = np.linspace(dist.ppf(.001), dist.ppf(.999), 100)
            x_cdf = dist.ppf(grid_cdf)
            l = ax_cdf[i].plot(x_dummy, dist.cdf(x_dummy), label = "CDF")
            scat_lhs = ax_cdf[i].scatter(x_lhs[key], sample[:, i], marker= "x", label = "LHS sample")
            # ax_cdf[i].scatter(x_rand[key], dist.cdf(x_rand[key]), label = "rvs")
            xlim = ax_cdf[i].get_xlim()
            ylim = ax_cdf[i].get_ylim()
            
            for j in range(len(grid_cdf)):
                grid_x = np.array([xlim[0], x_cdf[j], x_cdf[j]])
                grid_y = np.array([grid_cdf[j], grid_cdf[j], ylim[0]])
                line_lhs_grid, = ax_cdf[i].plot(grid_x, grid_y, linestyle = "dashed", color = 'r')#, label = "LHS grid")
            line_lhs_grid.set_label("LHS grid")
            ax_cdf[i].set_ylabel("CDF")
            ax_cdf[i].set_xlabel(labels[i])
            i += 1
        
    
        fig = [fig_pdf, fig_cdf]
        ax = [ax_pdf, ax_cdf]
        for axi in ax:
            try:
                axi[0].legend()
            except:
                axi.legend()
    return x_lhs, sample, fig, ax
def get_mc_points(par_samples, N_mc_dist = 1000, plot_mc_samples = False, labels = None):
    """
    

    Parameters
    ----------
    par_samples : TYPE np.array((dim_par, dim_par_samples))
        DESCRIPTION. dim_par_samples of samples from the posterior parameter distribution (the number of parameters are dim_par)
    N_mc_dist : TYPE, optional int
        DESCRIPTION. The default is 1000. number of MC samples to draw. N_mc_dist <= dim_par_samples
    plot_mc_samples : TYPE, optional bool
        DESCRIPTION. The default is False. Plot or not
    labels : TYPE, optional list
        DESCRIPTION. The default is None.

    Returns
    -------
    x_lhs : TYPE
        DESCRIPTION.
    sample : TYPE
        DESCRIPTION.
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    """
    (dim_par, dim_par_samples) = par_samples.shape
    if N_mc_dist > dim_par_samples:
        raise ValueError("N_mc_dist must be smaller than the posterior sample size")
    #Draw N_mc_dist samples without replacement from par_samples
    idx_mc = np.random.choice(dim_par_samples, #high value for randint
                                           size = N_mc_dist, 
                                           replace = False #draw without replacement
                                           )
    x_mc = par_samples[:, idx_mc].copy()
    # i = 0
    # for key, dist in dist_dict.items():
    #     x_mc[key] = dist.rvs(size = N_mc_dist)
    #     # x_rand[key] = dist.rvs(size = N_mc_dist)
    #     i += 1
    if not plot_mc_samples:
        sns_grid = None
    else: #do plot
        if labels is None:
            labels = ["" for k in range(dim_par)]
        kwargs_par_hist = {"corner": True}
        df_par_post = pd.DataFrame(data = par_samples.T, columns = list(labels))
        df_par_post["dist"] = "Full posterior"
        
        #A separate df for MC samples. If it is the same, then it may look like MC samples are not a subset of the full posterior distribution
        df_par_mc = pd.DataFrame(data = x_mc.T, columns = list(labels))
        df_par_mc["dist"] = "MC"
        
        df_par_all = pd.concat([df_par_post, df_par_mc], ignore_index = True)
        # print(df_par_post)
        # df_par_post["dist"].iloc[idx_mc] = "MC"
        # print(df_par_post)
        # sns_grid = sns.pairplot(df_par_post, **kwargs_par_hist)
        sns_grid = sns.pairplot(df_par_all, hue = "dist", **kwargs_par_hist)
        # sns_grid = sns.pairplot(df_par_post, hue = "dist", **kwargs_par_hist)
        sns_grid.fig.suptitle(r"MC samples used of $\theta_{fx}$")
        # #A plot of how the points distribute in the pdf of the dist
        # fig_pdf, ax_pdf = plt.subplots(dim_dist, 1)
        # if dim_dist == 1:
        #     ax_pdf = [ax_pdf]
        # # labels = [r"$\rho$ $[kg/m^3]$", "d [m]"]
        # line_color = []
        # i = 0
        # for key, dist in dist_dict.items(): #plot the pdf of each distribution
        #     x_dummy = np.linspace(dist.ppf(.001), dist.ppf(.999), 100)    
        #     l = ax_pdf[i].plot(x_dummy, dist.pdf(x_dummy), label = "pdf")
        #     ax_pdf[i].scatter(x_mc[key], np.zeros(N_mc_dist), 
        #                   label = "MC",
        #                   marker = "x")
        #     # ax_pdf[i].scatter(x_rand[key], np.zeros(N_lhs_dist), label = "rvs")
        #     ax_pdf[i].set_xlabel(labels[i])
        #     ax_pdf[i].set_ylabel("pdf")
        #     i += 1
            
    return x_mc, sns_grid

# def get_wmean_Q_from_lhs(par_lhs, x0, t_span, w_plant, par_nom):
#     x_nom = fx_ukf_ode(ode_model_plant, t_span, x0, args_ode = (w_plant, par_nom))
#     N_mc = list(par_lhs.values())[0].shape[0] #the number of MC samples (or LHS)
#     dim_x = x0.shape[0]
#     x_stoch = np.zeros((dim_x, N_mc))
#     par_i = {}
#     for i in range(N_mc):
#         for key, val in par_lhs.items():
#             par_i[key] = val[i]
#         x_stoch[:, i] = fx_ukf_ode(ode_model_plant, t_span, x0, args_ode = (w_plant, par_i))
    
#     Q_lhs = np.cov(x_stoch)
#     w_mean = np.mean(x_stoch - x_nom, axis = 1)
#     return w_mean, Q_lhs

def get_w_realizations_from_mc(par_mc, F, x, t_span, u, par_nom):
    x_nom = integrate_ode(F, x, t_span, u, par_nom)
    (dim_par, N_mc) = par_mc.shape #the number of MC samples (or LHS)
    dim_x = x.shape[0]
    x_stoch = np.zeros((dim_x, N_mc))
    par_i = par_nom.copy() #copy.deepcopy() not required here, since par_nom is not a nested dict (all vals contains a number only)
    for i in range(N_mc): #iterate through all the MC samples
        j = 0
        for key in par_i.keys(): #change dictionary values to this MC sample
            par_i[key] = par_mc[j, i]
            j += 1
        x_stoch[:, i] = integrate_ode(F, x, t_span, u, par_i) #compute x_stoch with this parameter sample
    w_stoch = x_stoch - x_nom.reshape(-1,1)
    return w_stoch

def get_wmean_Q_from_mc(par_mc, F, x, t_span, u, par_nom):
    
    w_stoch = get_w_realizations_from_mc(par_mc, F, x, t_span, u, par_nom) #get all realizations of w = f(par_sample) - f(par_nom)
    w_mean = np.mean(w_stoch, axis = 1)
    Q = np.cov(w_stoch)
    return w_mean, Q

def get_wmode_Q_from_mc(par_mc, F, x, t_span, u, par_nom, kwargs_solver = {}, plot_density = False):
    
    # dim_x = x.shape[0]
    # w_mode = np.zeros(dim_x)
    w_stoch = get_w_realizations_from_mc(par_mc, F, x, t_span, u, par_nom)
    w_mean = np.mean(w_stoch, axis = 1)
    # print(f"w_mean: {w_mean.shape}")
    
    #Find the mode by minimizing the pdf. An approximate pdf is made by kernel density estimation (kde). As we're optimizing/minimizing later, we scale the variables first
    w_stoch2 = w_stoch[1:, :] #drop V as first state
    w_mean2 = w_mean[1:] #dropV
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    w_scaled = scaler.fit_transform(w_stoch2.T)
    w_scaled = w_scaled.T
    
    kernel = scipy.stats.gaussian_kde(w_scaled) #create kde based on scaled values of w
    
    w_mean_scaled = scaler.transform(w_mean2.reshape(1, -1)).flatten() #initial guess for the mode
    
    #solve by scipy
    min_func = lambda x: -kernel.logpdf(x)
    res = scipy.optimize.minimize(min_func,
                                  w_mean_scaled,
                                  **kwargs_solver
                                  )
    mode_w_scaled = res.x
    solver_success = res.success
    
    #solve by casadi. Slower, probably since I am constructing the problem every time. Casadi and scipy give same solution, so I just stick with scipy now.
    # opts_solver = {"ipopt": {"acceptable_tol": 1e-8}}
    # res, solver_cd = get_w_mode_casadi(w_scaled, kernel.inv_cov, w_mean_scaled, opts_solver = kwargs_solver)
    # mode_w_scaled = np.array(res["x"]).flatten()
    # solver_success = solver_cd.stats()["success"]
    
    # print(solver_cd.stats()["success"])
    # mode_pdf = min_func(mode_w_scaled)
    
    mode_w = scaler.inverse_transform(mode_w_scaled.reshape(1, -1)).flatten()
    mode_w = np.insert(mode_w, 0, w_mean[0]) #insert mean value for w_V in the zero indez in mode_w
    # print(f"w_mean: {w_mean}\n",
    #       f"w_mode: {mode_w}\n",
    #       f"min_func(mean): {min_func(w_mean_scaled)}\n",
    #       f"min_func(mode): {min_func(mode_w_scaled)}")
    Q = np.cov(w_stoch)
    
    if plot_density:
        w_stoch_plt = w_stoch[1:, :]
        density = kernel(w_scaled)
        print(w_stoch_plt.shape)
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        x, y, z = w_stoch_plt
        scatter_plot = ax.scatter(x, y, z, c=density, label = r"$w_i$ samples")
        ax.scatter(mode_w[1], mode_w[2], mode_w[3], c = 'r', label = "Mode")
        
        ax.set_xlabel(r"$w_X [g/L?]$")
        ax.set_ylabel(r"$w_S [g/L?]$")
        ax.set_zlabel(r"$w_CO_2 [%]$")
        ax.set_box_aspect([np.ptp(i) for i in w_stoch_plt])  # equal aspect ratio

        cbar = fig.colorbar(scatter_plot, ax=ax)
        cbar.set_label(r"$KDE(w) \approx pdf(w)$")
        ax.legend()
    else:
        fig, ax = [None, None]
        
    return mode_w, Q, solver_success, fig, ax
def get_wmode_statistics(par_mc, F, x, t_span, u, par_nom, solver_nlp, hess_inv, plot_density = False):
    
    # dim_x = x.shape[0]
    # w_mode = np.zeros(dim_x)
    w_stoch = get_w_realizations_from_mc(par_mc, F, x, t_span, u, par_nom)
    w_mean = np.mean(w_stoch, axis = 1)
    # print(f"w_mean: {w_mean.shape}")
    
    #Find the mode by minimizing the pdf. An approximate pdf is made by kernel density estimation (kde). As we're optimizing/minimizing later, we scale the variables first
    w_stoch2 = w_stoch[1:, :] #drop V as first state
    w_mean2 = w_mean[1:] #dropV
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    w_scaled = scaler.fit_transform(w_stoch2.T)
    w_scaled = w_scaled.T
    
    kernel = scipy.stats.gaussian_kde(w_scaled) #create kde based on scaled values of w
    
    w_mean_scaled = scaler.transform(w_mean2.reshape(1, -1)).flatten() #initial guess for the mode
    
    inv_cov_kernel = kernel.inv_cov
    
    mode_w_scaled, Q_m, solver_success = get_w_mode_Q_cd(solver_nlp, hess_inv, w_scaled, inv_cov_kernel)
    
    # print(solver_success)
    mode_w = scaler.inverse_transform(mode_w_scaled.reshape(1, -1)).flatten()
    mode_w = np.insert(mode_w, 0, w_mean[0]) #insert mean value for w_V in the zero indez in mode_w
    # print(f"w_mean: {w_mean}\n",
    #       f"w_mode: {mode_w}\n",
    #       f"min_func(mean): {min_func(w_mean_scaled)}\n",
    #       f"min_func(mode): {min_func(mode_w_scaled)}")
    
    #need to rescale Q_m. Factorise as corr = std_dev @ corr @ std_dev and rescale std_dev
    std_dev_scaled = np.sqrt(np.diag(Q_m))
    Dinv = np.diag(1 / std_dev_scaled)
    corr = Dinv @ Q_m @ Dinv
    std_dev = np.diag(scaler.inverse_transform(std_dev_scaled.reshape(1, -1)).flatten())
    Q_calc = std_dev @ corr @ std_dev
    Q = np.cov(w_stoch) #in this way, we get at least some sense of w_V when we feed more sugar
    
    # print(f"Q_calc: {Q_calc}")
    # print(f"Q_sample: {Q}")
    Q[1:, 1:] = Q_calc #insert the calculated covariance matrix
    if plot_density:
        w_stoch_plt = w_stoch[1:, :]
        density = kernel(w_scaled)
        print(w_stoch_plt.shape)
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        x, y, z = w_stoch_plt
        scatter_plot = ax.scatter(x, y, z, c=density, label = r"$w_i$ samples")
        ax.scatter(mode_w[1], mode_w[2], mode_w[3], c = 'r', label = "Mode")
        
        ax.set_xlabel(r"$w_X [g/L?]$")
        ax.set_ylabel(r"$w_S [g/L?]$")
        ax.set_zlabel(r"$w_CO_2 [%]$")
        ax.set_box_aspect([np.ptp(i) for i in w_stoch_plt])  # equal aspect ratio

        cbar = fig.colorbar(scatter_plot, ax=ax)
        cbar.set_label(r"$KDE(w) \approx pdf(w)$")
        ax.legend()
    else:
        fig, ax = [None, None]
        
    return mode_w, Q, solver_success, fig, ax

def get_w_mode_hessian_inv_functions(dim_x, dim_N, opts_solver = {"print_time": 0, "ipopt": {"acceptable_tol": 1e-8, "print_level": 0}}):
    """
    Defines optimization problem (casadi) to minimize -log pdf of kernel density estimate of w with multivariate normal kernels.

    Parameters
    ----------
    dim_x : TYPE int
        DESCRIPTION. Dimension of solution
    dim_N : TYPE int
        DESCRIPTION. Number of points used when creating the KDE
    opts_solver : TYPE, optional dict
        DESCRIPTION. The default is {"ipopt": {"acceptable_tol": 1e-8}}.

    Returns
    -------
    S : TYPE cd.Function(x0=..., p=...)
        DESCRIPTION. The NLP to be minimized
    hf_inv : TYPE cd.Function(w_scaled, inv_cov_kernel, x)
        DESCRIPTION. Inverse of the Hessian

    """
    # opts = { #options for solver and casadi
    #                 "print_time":0,#suppress print output from Casadi
    #                 "ipopt": { #solver specific options
    #                     # "mehrotra_algorithm": "yes",#performs better for convex QP according to https://projects.coin-or.org/CoinBinary/export/837/CoinAll/trunk/Installer/files/doc/Short%20tutorial%20Ipopt.pdf. On my test problem, it runs slower though
    #                     "hessian_constant": "yes", # Better for QP since IPOPT only gets the Hessian once (solves it as QP). https://coin-or.github.io/Ipopt/OPTIONS.html, search for QP
    #                     "print_level": 0 #suppresses print from IPOPT.
    #                           }
    x = cd.MX.sym("x", dim_x, 1)
    w_scaled = cd.MX.sym("w_scaled", dim_x, dim_N)
    inv_cov_kernel = cd.MX.sym("inv_cov_kernel", dim_x, dim_x)
    
    #create function which should be minimized (min_x -logpdf_kernel(x))
    f = [-.5*(x-w_scaled[:, i]).T @ inv_cov_kernel @ (x-w_scaled[:, i]) 
         for i in range(dim_N)]
    f = cd.vertcat(*f)
    f = -cd.log(np.ones((1, dim_N)) @ cd.exp(f))
    
    #concatenate input parameters and create nlp
    p = cd.horzcat(w_scaled, inv_cov_kernel)
    nlp = {"x": x,
           "p": p,
           "f": f}
    S = cd.nlpsol("S", "ipopt", nlp, opts_solver) #create solver instance
    
    #calculate inverse of the Hessian
    [hf, gf] = cd.hessian(f, x)
    hf_inv = cd.Function("hf", [w_scaled, inv_cov_kernel, x], [cd.inv(hf)])
    
    return S, hf_inv

def get_w_mode_Q_cd(S, hf_inv, w_scaled, inv_cov_kernel):
    
    (dim_x, dim_p) = w_scaled.shape
    
    p_in = np.concatenate((w_scaled, inv_cov_kernel), axis = 1)
    res = S(x0 = w_scaled.mean(axis=1), p = p_in)
    solver_stats = S.stats() #can check if solver has converged or not
    # print(solver_stats)
    w_mode = res["x"]
    w_cov = np.array(hf_inv(w_scaled, inv_cov_kernel, w_mode))
    return np.array(w_mode).flatten(), w_cov, solver_stats["success"]

# w_scaled = np.load("C:/Users/halvorak/Dokumenter/SUBPRO Digital twin/Est-Q-by-GenUT/scripts/w_scaled.npy")
# dim_x, dim_p = w_scaled.shape
# S, hf_inv = get_w_mode_hessian_inv_functions(dim_x, dim_p)
# w_mode, Q_m, solver_stats = get_w_mode_Q_cd(S, hf_inv, w_scaled)


def evaluate_jac_p(jac_p_fun, x, t_span, u, par_nom):
    """
    Calculate df/dp|x, u, par_nom

    Parameters
    ----------
    jac_p_fun : TYPE casadi.Function
        DESCRIPTION. Takes as input [x, p_aug]
    x : TYPE np.array((dim_x,))
        DESCRIPTION. Values of x. dim_x must correspond to casadi variable x in ode_model_plant
    t_span : TYPE tuple
        DESCRIPTION. Integration time. dt=t_span[1]-t_span[0]
    u : TYPE np.array
        DESCRIPTION. Input
    par_nom : TYPE dict
        DESCRIPTION. Nominal parameter values. p_aug = [u, par_nom.values(), dt]

    Returns
    -------
    TYPE np.array((dim_x, dim_u + dim_par + 1))
        DESCRIPTION.

    """
    par_aug = np.hstack((u, np.array(list(par_nom.values())), t_span[1] - t_span[0]))
    jac_p_args = [x, par_aug]
    jac_p_aug_val = jac_p_fun(*jac_p_args) #cd.DM type. Shape: ((dim_f=dim_x, dim_p_aug))
    jac_p_aug_val = np.array(jac_p_aug_val) #cast to numpy
    
    #Extract the correct jacobian. Have df/dp_aug, want only df_dp
    dim_u = u.shape[0]
    dim_x = x.shape[0]
    dim_par = len(par_nom)
    jac_p_val = jac_p_aug_val[:, dim_u:-1]
    dim_par = jac_p_val.shape[1]
    if not (dim_par == jac_p_val.shape[1]) and (jac_p_val.shape[0] == dim_x):
        raise ValueError(f"Dimension mismatch. Par: {jac_p_val.shape[1]} and {dim_par}. States: {jac_p_val.shape[0]} and {dim_x}")
        
    return jac_p_val


def get_Q_from_linearization(jac_p_fun, x, t_span, u, par_nom, par_cov):
    jac_p = evaluate_jac_p(jac_p_fun, x, t_span, u, par_nom)
    Q = jac_p @ par_cov @ jac_p.T
    return Q

def evaluate_jac_p_num(F, x, t_span, u, par_nom, h = 1e-8):
    dim_x = x.shape[0]
    dim_par = len(par_nom)
    jac_par = np.zeros((dim_x, dim_par))
    x_nom = integrate_ode(F, x, t_span, u, par_nom)

    for i in range(dim_par):
        par_name = list(par_nom.keys())[i]
        par_i = par_nom.copy() #This is a shallow copy. Ok here since we don't have nested dictionary for this specific case
        par_i[par_name] = par_i[par_name] + h
        xi = integrate_ode(F, x, t_span, u, par_i)
        jac_par[:, i] = (xi - x_nom)/h
    return jac_par
    
def get_Q_from_numerical_linearization(F, x, t_span, u, par_nom, par_cov, h = 1e-8):
    jac_p = evaluate_jac_p_num(F, x, t_span, u, par_nom, h = h)
    Q = jac_p @ par_cov @ jac_p.T
    return Q
# def get_vmean_R_from_lhs(par_lhs, x, dim_y):
#     N_mc = list(par_lhs.values())[0].shape[0] #the number of MC samples (or LHS)
#     y = np.zeros((dim_y, N_mc))
#     par_i = {}
#     for i in range(N_mc):
#         for key, val in par_lhs.items():
#             par_i[key] = val[i]
#         y[:, i] = hx(x, par_i)
#     v_lhs = np.mean(y, axis = 1)
#     R_lhs = np.cov(y)
#     return v_lhs, R_lhs

def get_v_realizations_from_mc(par_mc, x, dim_y, par_nom):
    y_nom = hx(x, par_nom)
    N_mc = list(par_mc.values())[0].shape[0] #the number of MC samples (or LHS)
    y_stoch = np.zeros((dim_y, N_mc))
    par_i = {}
    for i in range(N_mc):
        for key, val in par_mc.items():
            par_i[key] = val[i]
        y_stoch[:, i] = hx(x, par_i)
    v_stoch = y_stoch - y_nom.reshape(-1,1)
    return v_stoch

def get_vmean_R_from_mc(par_mc, x, dim_y, par_nom):
    
    v_stoch = get_v_realizations_from_mc(par_mc, x, dim_y, par_nom)
    v_mean = np.mean(v_stoch, axis = 1)
    R = np.cov(v_stoch)
    return v_mean, R

def get_vmode_R_from_mc(par_mc, x, dim_y, par_nom, nbins = 20):
    
    v_stoch = get_v_realizations_from_mc(par_mc, x, dim_y, par_nom)
    hist, bin_edges = np.histogram(v_stoch, bins = nbins)
    idx = np.argmax(hist) #returns the index where there are most samples in the bin
    mode_limits = np.array([bin_edges[idx], bin_edges[idx+1]]) #the mode is somewhere within these limits
    v_mode = np.mean(mode_limits)
    R = np.cov(v_stoch)
    return v_mode, R


def solve_constrained_sigmapoint_updatefunction_qp(Dk, Rk, Pk_prior, yk, sigma_prop, sigma_lb, x0 = None):
    #See Kolås's article from 2009
    if x0 is None:
        x0 = sigma_prop.copy()
        
    Pk_inv = np.linalg.inv(Pk_prior)
    Rk_inv = np.linalg.inv(Rk)
    
    dim_x = sigma_prop.shape[0]
    
    #Create symbolic variable for the sigma points we want to find
    x = cd.MX.sym("x", dim_x)
    
    J = x.T @cd.DM(Dk.T @ Rk_inv @ Dk + Pk_inv) @ x - 2*cd.DM(
        yk.T @ Rk_inv @ Dk + sigma_prop.T @ Pk_inv) @ x
    
    #Create NLP problem for Casadi/ipopt
    nlp = {"x": x, "f": J}
    solver = cd.nlpsol('solver', 'ipopt', nlp)
    res = solver(x0 = x0,
                 lbx = sigma_lb)
    x_sol = np.array(res["x"]).flatten()
    return x_sol, res

def select_points_from_mean_multivar_mahalanobis(vals, std_low, std_high):
    # From https://stackoverflow.com/questions/44998025/selecting-points-in-dataset-that-belong-to-a-multivariate-gaussian-distribution
    
    # Compute covariance matrix and its inverse
    cov = np.cov(vals.T)
    cov_inverse = np.linalg.inv(cov)
    
    # Mean center the values
    mean = np.mean(vals, axis=0)
    centered_vals = vals - mean
    
    # Compute Mahalanobis distance
    dist = np.sqrt(np.sum(centered_vals * cov_inverse.dot(centered_vals.T).T, axis=1))
    # Find points that are "far away" from the mean
    idx_low = std_low <= dist
    idx_high =  dist <= std_high
    idx = (dist >= std_low)*(dist <= std_high)
    idx = np.where(idx)[0]
    return vals[idx[0], :]

def solve_schur_complement_normal_cholesky():
    n_x = cd.SX.sym("n_x", 1)
    n_p = cd.SX.sym("n_p", 1)
    
    # schur_complement_based = (1/3)*n_x**3 + (n_x**2)*(1+n_p) + 2*n_x*n_p**2
    schur_complement_based = cd.log(4*n_x*n_p**2
                              + 2*n_p*n_x**2
                              +n_x**2
                                +(1/3)*n_x**3
                              )
    normal_cholesky = cd.log((1/3)*(
        n_x
                                    + n_p
                                    )**3)
    res = schur_complement_based - normal_cholesky
    
    res = cd.Function("res", 
                      [n_p, n_x], 
                      [cd.vertcat(res)] #equations/residuals
                      )
    
    #Form ode dict and integrator
    # opts = {"abstol": 1e-14,
    #         "linear_solver": "csparse"
    #         }
    opts = {}
    F = cd.rootfinder("F", "newton", res, opts)
    return F
def solve_opt_tradeoff(F, x0, par):
    x0 = cd.vertcat(x0)
    xk = F(x0, cd.vcat(par))
    # xf = Fend["xf"]
    # xf_np = np.array(xf).flatten()
    # return xf_np
    return np.array(xk).flatten()

# get_literature_values(plot_par=True)
# H = np.eye(2)
# Hcd = cd.DM(H)

# g = np.zeros((2, 1))
# g[0,0,] = 1
# gcd = cd.DM(g)
# x = cd.MX.sym("x", 2, 1)

# nlp = {"x": x, "f": x.T@Hcd@x + gcd.T@x}

# S = cd.nlpsol('S', 'ipopt', nlp)
# r = S(x0=[2.5,3.0],
#       # lbx=.5*cd.DM(np.ones((2,1))))
#        lbx=cd.DM(np.array([[.5], [.3]]))
#     # lbx = .5
#     )
# print(S)

# x_np = np.array(r["x"]).flatten()