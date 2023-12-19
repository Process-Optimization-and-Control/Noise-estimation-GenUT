# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:47:07 2021

@author: halvorak
"""
import numpy as np
import scipy.stats
import casadi as cd
from state_estimator import sigma_points_classes as spc


def ode_model_plant(dt, power_par = 2.):
    #Make parameters
    k1 = cd.MX.sym("k1", 1) # [-]
    k2 = cd.MX.sym("k2", 1) # [-]
    k3 = cd.MX.sym("k3", 1) # [-]
    k4 = cd.MX.sym("k4", 1) # [-]
    
    #States
    cA = cd.MX.sym("cA", 1)
    cB = cd.MX.sym("cB", 1)
    cC = cd.MX.sym("cC", 1)
    
    # parameters exponentiated to the right power
    # power_par = 4.
    k1p = k1**power_par
    k2p = k2**power_par
    k3p = k3**power_par
    k4p = k4**power_par
    
    #ode system
    cA_dot = -k1p*cA + k2p*cB*cC
    cB_dot = k1p*cA - k2p*cB*cC - 2*k3p*cB**2 + 2*k4p*cC
    cC_dot = k1p*cA - k2p*cB*cC + k3p*cB**2 - k4p*cC
    # #ode system
    # cA_dot = -k1*cA + k2*cB*cC
    # cB_dot = k1*cA - k2*cB*cC - 2*k3*cB**2 + 2*k4*cC
    # cC_dot = k1*cA - k2*cB*cC + k3*cB**2 - k4*cC
    
    
    #Concatenate equation, states, inputs and parameters
    diff_eq = cd.vertcat(cA_dot, cB_dot, cC_dot)
    x_var = cd.vertcat(cA, cB, cC)
    p_var = cd.vertcat(k1, k2, k3, k4)
    
    #Form ode dict and integrator
    ode = {"x": x_var,
           "p": p_var,
           "ode": diff_eq*dt}
    
    opts = {#options for solver and casadi
            # "cvode": {#solver specific options
            #            "max_num_steps": 10}
            "max_num_steps": 200}
    opts = {}
    F = cd.integrator("F", "cvodes", ode, opts)
    # F = cd.integrator("F", "rk", ode)
    # F = cd.integrator("F", "idas", ode)
    
    #Jacobian for the linearized tuning approach
    jac_p_func = cd.jacobian(ode["ode"], ode["p"])
    jac_p = cd.Function("jac_p", [ode["x"], ode["p"]], [jac_p_func])

    return F,jac_p,x_var,p_var,diff_eq

def integrate_ode(F, x0, par_fx):
    x0 = cd.vertcat(x0)
    pk = list(par_fx.values())
    # print(pk)
    Fend = F(x0 = x0, p = cd.vertcat(pk))
    xf = Fend["xf"]
    xf_np = np.array(xf).flatten()
    return xf_np


def hx(x):
    """
    Measurement model

    Parameters
    ----------
    x : TYPE np.array(dim_x,)
        DESCRIPTION. Current state value

    Returns
    -------
    y : TYPE np.array(dim_y,)
        DESCRIPTION. Measurement (without measurement noise)

    """
    RT = 32.84
    y = np.array([RT, RT, RT]) @ x
    return np.atleast_1d(y)



def get_literature_values_points_dist(dt, N_samples = int(1e4), power_par = 2.):
    """
    Initial values, parameters etc. Made here for making main script cleaner.

    Returns
    -------
    x0 : TYPE np.array(dim_x,)
        DESCRIPTION. Starting point for UKF (mean value of initial guess)
    P0 : TYPE np.array((dim_x, dim_x))
        DESCRIPTION. Initial covariance matrix. Gives uncertainty of starting point. The starting point for the true system is drawn as x_true ~ N(x0,P0)
    par_mean_fx : TYPE dict
        DESCRIPTION. Parameters for the process model.
    par_cov_fx : TYPE dixt
        DESCRIPTION. Parameters covariance
    Q : TYPE np.array((dim_w, dim_w))
        DESCRIPTION. Process noise covariance matrix
    R : TYPE np.array((dim_v, dim_v))
        DESCRIPTION. Measurement noise covariance matrix

    """
    
    #Nominal parameter values 
    par_mean_fx = dict(k1 = .5, k2 = .05, k3 = .2, k4 = .01)
    par_cov_fx = np.array(
        [[3.7e-6, 9.5e-6, -5.83e-6, 2.36e-8],
         [9.5e-6, 3.37e-4, -2.55e-4, -2.68e-6],
         [-5.83e-6, -2.55e-4, 1.97e-4, 2.31e-6],
         [2.36e-8, -2.68e-6, 2.31e-6, 4.79e-8]
         ])
    assert (par_cov_fx == par_cov_fx.T).all(), f"par_cov_fx is not symmetric, {(par_cov_fx == par_cov_fx.T)}"
    dim_par = par_cov_fx.shape[0]
    
    std_dev_par = np.sqrt(np.diag(par_cov_fx))
    std_dev_inv = np.diag([1/si for si in std_dev_par])
    corr_par = std_dev_inv @ par_cov_fx @ std_dev_inv
    
    #rescale the standard devaiation, so that we don't sample negative values for k2
    # std_dev_par[0] *=200
    std_dev_par[1] *= 2.5
    std_dev_par[-1] *= 1.2
    # print(f"std_dev: {std_dev_par}")
    # print(f"corr_par: {corr_par}")
    # print(f"mean_par: {par_mean_fx}")
    std_dev_par = np.diag(std_dev_par)
    # corr_par = np.eye(len(par_mean_fx))
    par_cov_fx = std_dev_par @ corr_par @ std_dev_par
    
    dist_multivar = scipy.stats.multivariate_normal(list(par_mean_fx.values()), par_cov_fx)
    
    #Sample values from dist_multivar and accept the samples if all values are above the constraint
    N_samples = int(1e4)
    par_samples = get_points(dist_multivar, None, N = N_samples, constraint = 1e-8)
    
    # par_samples = np.sqrt(par_samples)
    par_samples = np.power(par_samples, 1/power_par)
    # par_samples = np.power(par_samples, .25)
    
    #evaluate mean, cov, cm3, cm4
    par_mean_fx_val = np.mean(par_samples, axis = 0)
    par_keys = par_mean_fx.keys()
    del par_mean_fx, par_cov_fx, std_dev_par,corr_par,std_dev_inv,dist_multivar
    par_mean_fx = {key: val for key, val in zip(par_keys, par_mean_fx_val)}
    par_cov_fx = np.cov(par_samples, rowvar = False)
    cm3_par = scipy.stats.moment(par_samples, moment = 3, axis = 0)
    cm4_par = scipy.stats.moment(par_samples, moment = 4, axis = 0)
    
    if False:
        #check that Pearsons inequality is fulfilled (equation 35 in the GenUT paper)
        P_sqrt = scipy.linalg.cholesky(par_cov_fx, lower = True)
        P_sqrt_pow3_inv = scipy.linalg.inv(np.power(P_sqrt, 3))
        
        
        S_std = P_sqrt_pow3_inv @ cm3_par
        #check that inequality constraint for K is fulfilled (ensures u>0)
        K_lb = np.power(P_sqrt, 4) @ np.square(S_std)
        if not (cm4_par > K_lb).all():
            print(f"I utils. Har cm4_par_old = {cm4_par} og K_lb = {K_lb}")
            cm4_par = np.array([Ki if Ki > Ki_lb else Ki_lb + 1e-15 for Ki, Ki_lb in zip(cm4_par, K_lb)])
        assert (cm4_par > K_lb).all(), f"Pearsons inequality not fulfilled. K should be larger than K_lb. Have K = {cm4_par} and K_lb = {K_lb}"
    
    # cm3_par = np.zeros(dim_par)
    # cm4_par = spc.GenUTSigmaPoints.compute_cm4_isserlis_for_multivariate_normal(par_cov_fx)
    
    # dist_univar = {}
                                                    
    
    #Initial state and uncertainty
    x0 = np.array([.5, .05, .0])
    std_dev0 = np.diag([1e-1, 1e-1, 1e-3])
    corr_0 = np.eye(x0.shape[0])
    P0 = std_dev0 @ corr_0 @ std_dev0
    P0 = .5*(P0 + P0.T)
    
    #Process and measurement noise
    Q = np.diag(np.square([1e-3, 1e-3, 1e-3]))
    R = np.diag([.25**2])
    
    return x0, P0, par_mean_fx, par_cov_fx, cm3_par, cm4_par, par_samples, Q, R

def get_points(dist_multivar, dist_univar, N = int(1e3), constraint = None):
    assert isinstance(N, int), f"N must be an integer, it is now {type(N)}"
    
    if constraint is None:
        points_multivar = dist_multivar.rvs(size = N)
    else:
        assert isinstance(constraint, (float, int)), f"constraint must be a float or int if it is specified, it is {type(constraint)}"
        points_multivar = get_positive_point(dist_multivar, eps = constraint, N = N)
    if dist_univar: #check if it is empty dict
        raise ValueError("Implement sampling with univar as well")
    return points_multivar

def evaluate_jac_p(jac_p_fun, x, par_nom):
    """
    Calculate df/dp|x, u, par_nom

    Parameters
    ----------
    jac_p_fun : TYPE casadi.Function
        DESCRIPTION. Takes as input [x, p_aug]
    x : TYPE np.array((dim_x,))
        DESCRIPTION. Values of x. dim_x must correspond to casadi variable x in ode_model_plant
    par_nom : TYPE dict
        DESCRIPTION. Nominal parameter values. p_aug = [u, par_nom.values(), dt]

    Returns
    -------
    TYPE np.array((dim_x, dim_u + dim_par + 1))
        DESCRIPTION.

    """
    # par_aug = np.hstack((u, np.array(list(par_nom.values())), t_span[1] - t_span[0]))
    jac_p_args = [x, np.array(list(par_nom.values()))]
    jac_p_aug_val = jac_p_fun(*jac_p_args) #cd.DM type. Shape: ((dim_f=dim_x, dim_p_aug))
    jac_p_val = np.array(jac_p_aug_val) #cast to numpy
    # jac_p_aug_val = np.array(jac_p_aug_val) #cast to numpy
    
    #Extract the correct jacobian. Have df/dp_aug, want only df_dp
    dim_x = x.shape[0]
    dim_par = len(par_nom)
    if not (dim_par == jac_p_val.shape[1]) and (jac_p_val.shape[0] == dim_x):
        raise ValueError(f"Dimension mismatch. Par: {jac_p_val.shape[1]} and {dim_par}. States: {jac_p_val.shape[0]} and {dim_x}")
        
    return jac_p_val


def get_Q_from_linearization(jac_p_fun, x, par_nom, par_cov):
    jac_p = evaluate_jac_p(jac_p_fun, x, par_nom)
    Q = jac_p @ par_cov @ jac_p.T
    return Q
    
def evaluate_jac_p_num(F, x, par_nom, h = 1e-6):
    dim_x = x.shape[0]
    dim_par = len(par_nom)
    jac_par = np.zeros((dim_x, dim_par))
    x_nom = integrate_ode(F, x, par_nom)

    for i in range(dim_par):
        par_name = list(par_nom.keys())[i]
        par_i = par_nom.copy() #This is a shallow copy. Ok here since we don't have nested dictionary for this specific case
        par_i[par_name] = par_i[par_name] + h
        xi = integrate_ode(F, x,par_i)
        jac_par[:, i] = (xi - x_nom)/h
    return jac_par
    
def get_Q_from_numerical_linearization(F, x, par_nom, par_cov, h = 1e-8):
    jac_p = evaluate_jac_p_num(F, x, par_nom, h = h)
    Q = jac_p @ par_cov @ jac_p.T
    return Q

def fx_for_UT_gen_Q(sigmas, list_of_keys, F, x, par_fx):
    
    for i in range(len(list_of_keys)):
        key = list_of_keys[i]
        if not key in par_fx:
            raise KeyError("This key should be in par")
        par_fx[key] = sigmas[i]
    x_propagated = integrate_ode(F, x, par_fx)
    return x_propagated

def get_w_realizations_from_mc(par_mc, F, x, par_nom):
    (dim_par, N_mc) = par_mc.shape #the number of MC samples (or LHS)
    #check that the input is correct
    assert N_mc > dim_par, f"Probably the matrix par_mc should be transposed. Have now that dim_par = {dim_par} and N_mc = {N_mc}. Most likely N_mc should be significantly larger than dim_par"
    
    dim_x = x.shape[0]
    x_stoch = np.zeros((dim_x, N_mc))
    par_i = par_nom.copy() #copy.deepcopy() not required here, since par_nom is not a nested dict (all vals contains a number only)
    
    x_nom = integrate_ode(F, x, par_nom)
    for i in range(N_mc): #iterate through all the MC samples
        j = 0
        for key in par_i.keys(): #change dictionary values to this MC sample
            par_i[key] = par_mc[j, i]
            j += 1
        x_stoch[:, i] = integrate_ode(F, x, par_i) #compute x_stoch with this parameter sample
    w_stoch = x_stoch - x_nom.reshape(-1,1)
    return w_stoch

def get_wmean_Q_from_mc(par_mc, F, x, par_nom):
    
    w_stoch = get_w_realizations_from_mc(par_mc, F, x, par_nom) #get all realizations of w = f(par_sample) - f(par_nom)
    w_mean = np.mean(w_stoch, axis = 1)
    Q = np.cov(w_stoch)
    return w_mean, Q

def compute_performance_index_valappil(x_kf, x_ol, x_true, cost_func = "RMSE"):
    if cost_func == "RMSE":
        # J = np.linalg.norm(x_kf - x_true, axis = 1, ord = 2)
        J = np.sqrt(np.square(x_kf - x_true).mean(axis = 1))
    elif cost_func == "valappil": #valappil's cost index
        J = np.divide(np.linalg.norm(x_kf - x_true, axis = 1, ord = 2),
                      np.linalg.norm(x_ol - x_true, axis = 1, ord = 2))
    else:
        raise ValueError("cost function is wrongly specified. Must be RMSE or valappil.")
    return J


def get_positive_point(dist, eps = 1e-4, N = 1):
    """
    Sample a point from the distribution dist, and requires all points to be positive

    Returns
    -------
    point : TYPE np.array((dim_x,))
        DESCRIPTION. Points with all positive values

    """
    dim_x = dist.rvs(size = 1).shape[0]
    points = np.zeros((N, dim_x))
    
    sampled_points = dist.rvs(size = 10*N)
    k = 0
    for i in range(sampled_points.shape[0]):
        if (sampled_points[i,:] > eps).all(): #accept the point
            points[k, :] = sampled_points[i,:]
            k += 1
            if k >= N:
                break
    assert k >= N, "Did not find enough points above the constraint"
    assert (points > eps).all(), "Not all points are above the constraint"
    
    if N == 1:
        points = points.flatten()
    # all_positive = False
    # i = 0
    # while not all_positive:
    #     point = dist.rvs(size = N)
    #     all_positive = (point >= eps).all()
    #     i += 1
    #     if i > 100:
    #         raise ValueError(f"Have tried {i} random draws, and not all points were above {eps}. Distribution is wrong or {eps} is too high.")
    return points
    

def get_corr_std_dev(P):
    std_dev = np.sqrt(np.diag(P))
    std_dev_inv = np.diag([1/si for si in std_dev])
    corr = std_dev_inv @ P @ std_dev_inv
    return std_dev, corr
    