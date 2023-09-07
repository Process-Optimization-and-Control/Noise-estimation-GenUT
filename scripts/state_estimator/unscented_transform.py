import numpy as np
import scipy.linalg

"""
NB: Documentation of the classes/functions may be outdated
"""

def unscented_transformation_gut(sigmas, wm, wc, symmetrization = True):
    """
    Calculates mean and covariance of sigma points by the unscented transform.

    Parameters
    ----------
    sigmas : TYPE np.ndarray(n, dim_sigma)
        DESCRIPTION. Array of sigma points. Each column contains a sigma point
    wm : TYPE np.array(dim_sigma,)
        DESCRIPTION. Weights for the mean calculation of each sigma point.
    wc : TYPE np.array(dim_sigma,)
        DESCRIPTION. Weights for the covariance calculation of each sigma point.
    symmetrization : TYPE bool, optional
        DESCRIPTION. Default is true. Symmetrize covariance/correlation matrix afterwards with Py = .5*(Py+Py.T)
    

    Returns
    -------
    mean : TYPE np.array(dim_y,)
        DESCRIPTION. Mean value of Y=f(X), X is a random variable (RV) 
    Py : TYPE np.array(dim_y,dim_y)
        DESCRIPTION. Covariance matrix, cov(Y) where Y=f(X)

    """
    try:
        (n, dim_sigma) = sigmas.shape
    except ValueError: #sigmas is 1D
        sigmas = np.atleast_2d(sigmas)
        (n, dim_sigma) = sigmas.shape 
        assert dim_sigma == wm.shape[0], "Dimensions are wrong"
    
    mean = sigmas @ wm
    
    #will only work with residuals between sigmas and mean, so "recalculate" sigmas
    sigmas = sigmas - mean.reshape(-1, 1)
    
    # Py = sum([wc_i*(np.outer(sig_i, sig_i)) for wc_i, sig_i in zip(wc, sigmas.T)])
    Py = (wc*sigmas) @ sigmas.T

    if symmetrization:
        Py = .5*(Py + Py.T)
    
    return mean, Py


def unscented_transform_w_function_eval(sigmas, wm, wc, func, first_yi = None):
    dim_x, dim_sigma = sigmas.shape
    if first_yi is None: #the first (or zeroth) sigma-point has not been calculated outside this function. compute it here
        first_yi = func(sigmas[:, 0])
    
    dim_y = first_yi.shape[0]
    sig_y = np.zeros((dim_y, dim_sigma))
    sig_y[:, 0] = first_yi.copy()
    for i in range(1, dim_sigma):
        sig_y[:, i] = func(sigmas[:, i]) 
    mean_y, P_y = unscented_transformation_gut(sig_y, wm, wc, symmetrization = True)
    return mean_y, P_y

def unscented_transform_w_function_eval_wslr(sigmas, wm, wc, func, first_yi = None):
    dim_x, dim_sigma = sigmas.shape
    if first_yi is None: #the first (or zeroth) sigma-point has not been calculated outside this function. compute it here
        first_yi = func(sigmas[:, 0])
    
    dim_y = first_yi.shape[0]
    sig_y = np.zeros((dim_y, dim_sigma))
    sig_y[:, 0] = first_yi.copy()
    for i in range(1, dim_sigma):
        sig_y[:, i] = func(sigmas[:, i])
        
    mean_y = sig_y @ wm
    
    #normalized sigmas
    sig_yn = sig_y - mean_y.reshape(-1, 1)
    sig_yn_wc = wc*sig_yn
    Py = sig_yn_wc @ sig_yn.T
    Py = .5*(Py + Py.T) #symmetrize
    
    #cross co-variance
    sig_xn = sigmas - sigmas[:,0].reshape(-1,1)
    sig_xn_wc = wc*sig_xn
    Pxy = sig_xn_wc @ sig_yn.T
    
    #covariance of Px
    Px = sig_xn_wc @ sig_xn.T
    Px = .5*(Px + Px.T) #symmetrize
    assert (dim_x, dim_x) == Px.shape, f"Px dimension wrong, dims are {Px.shape}"
    assert (dim_x, dim_y) == Pxy.shape, f"Pxy dimension wrong, dims are {Pxy.shape}"
    
    #Linear regression parameter
    # A = scipy.linalg.solve(Px, Pxy.T)
    A = Pxy.T @ np.linalg.inv(Px)
    
    return mean_y, Py, A
