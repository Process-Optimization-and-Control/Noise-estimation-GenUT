# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:17:56 2022

@author: halvorak

NB: Documentation of the classes/functions may be outdated
"""

from . import unscented_transform

import numpy as np
import scipy.linalg

class UKFBase():
    r"""
    Base class for UKF implementations


    Parameters
    ----------

    dim_w : int
        Process noise dimension.


    dim_v : int
        Measurement noise dimension


    hx : function(x,**hx_args)
        Measurement function. Converts state vector x into a measurement
        vector of shape (dim_y,).

    fx : function(x,**fx_args)
        Propagation of states from current time step to the next.

    points_x : class
        Class which computes the sigma points and weights for a UKF
        algorithm. You can vary the UKF implementation by changing this
        class. 

    msqrt : callable(ndarray), default=scipy.linalg.sqrtm
        Defines how we compute the square root of a matrix, which has
        no unique answer. Uses the same square-root as points_x. Alternatives are principal matrix square-root and Cholesky decomposition. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix. Daid et al recommends principal matrix square root, others (Julier, Grewal) recommends Cholesky.




    Attributes
    ----------

    R : numpy.array(dim_y, dim_y)
        measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        process noise matrix

    K : numpy.array
        Kalman gain

    y_res : numpy.array
        innovation residual


    """

    def __init__(self, fx, hx, points_x, Q, R, 
                 w_mean = None, v_mean = None, name=None, check_negative_sigmas = False):
        """
        Create a Kalman filter. IMPORTANT: Additive white noise is assumed!

        """
        self.check_negative_sigmas = check_negative_sigmas
        
        #dimensions
        dim_w = Q.shape[0]
        dim_v = R.shape[0]
        Q = np.atleast_2d(Q)
        R = np.atleast_2d(R)
        
        # check inputs
        assert ((dim_w, dim_w) == Q.shape)
        assert ((dim_v, dim_v) == R.shape)
        assert (Q == Q.T).all() #symmtrical
        assert (R == R.T).all() #symmtrical
        
        if w_mean is None:
            w_mean = np.zeros((dim_w,))
        
        if v_mean is None:
            v_mean = np.zeros((dim_v,))

        self._dim_w = dim_w
        self._dim_v = dim_v
        self.w_mean = w_mean
        self.Q = Q
        self.v_mean = v_mean
        self.R = R
        
        #save functions etc
        self.points_fn_x = points_x
        self._num_sigmas_x = points_x.num_sigma_points()
        self.hx = hx
        self.fx = fx
        self.msqrt = points_x.sqrt #use the same square-root function as the sigma-points
        self._name = name  # object name, handy when printing from within class

    def compute_transformed_sigmas(self, sigmas_in, func, **func_args):
        """
        Send sigma points through a nonlinear function. Call general distribution z, dimension of this variable is dim_z

        Parameters
        ----------
        sigmas_in : TYPE np.array((dim_z, dim_sigmas))
            DESCRIPTION. Sigma points to be propagated
        func : TYPE function(np.array(dim_z,), **func_args). F(dim_z)=>dim_q, q output dimension
            DESCRIPTION. function the sigma points are propagated through
        **func_args : TYPE list, optional
            DESCRIPTION. Additional inputs to func

        Returns
        -------
        sigmas_out : TYPE np.array((dim_q, dim_sigmas))
            DESCRIPTION. Propagated sigma points

        """
        sigmas_out = map(func, sigmas_in.T)
        sigmas_out = np.array(list(sigmas_out)).T
        
        if self.check_negative_sigmas:
            if ((sigmas_in < 0).any() or (sigmas_out < 0).any()):
                raise ValueError("Negative sigma-points detected")
        return sigmas_out
    def compute_transformed_sigmas2(self, sigmas_in, func, func_args = [], func_kwargs = {}):
        """
        Send sigma points through a nonlinear function. Call general distribution z, dimension of this variable is dim_z

        Parameters
        ----------
        sigmas_in : TYPE np.array((dim_z, dim_sigmas))
            DESCRIPTION. Sigma points to be propagated
        func : TYPE function(np.array(dim_z,), **func_args). F(dim_z)=>dim_q, q output dimension
            DESCRIPTION. function the sigma points are propagated through
        **func_args : TYPE list, optional
            DESCRIPTION. Additional inputs to func

        Returns
        -------
        sigmas_out : TYPE np.array((dim_q, dim_sigmas))
            DESCRIPTION. Propagated sigma points

        """
        # print(len(func_args))
        # sigmas_out = np.vstack([func(si, *func_args) for si in sigmas_in.T]).T
        f = lambda x: func(x, *func_args, **func_kwargs)
        sigmas_out = map(f, sigmas_in.T)
        sigmas_out = np.array(list(sigmas_out)).T
        if self.check_negative_sigmas:
            if ((sigmas_in < 0).any() or (sigmas_out < 0).any()):
                raise ValueError("Negative sigma-points detected")
        return sigmas_out

    def cross_covariance(self, sigmas_x, sigmas_y, W_c):
        """
        Cross-covariance between two probability distribution x,y which are already centered around their mean values x_mean, y_mean

        Parameters
        ----------
        sigmas_x : TYPE np.array((dim_x, dim_sigmas))
            DESCRIPTION. Sigma-points created from the x-distribution, centered around x_mean
        sigmas_y : TYPE np.array((dim_y, dim_sigmas))
            DESCRIPTION. Sigma-points created from the y-distribution, centered around y_mean
        W_c : TYPE np.array(dim_sigmas,)
            DESCRIPTION. Weights to compute the covariance

        Returns
        -------
        P_xy : TYPE np.array((dim_x, dim_y))
            DESCRIPTION. Cross-covariance between x and y

        """
        try:
            (dim_x, dim_sigmas_x) = sigmas_x.shape
        except ValueError:  # sigmas_x is 1D
            sigmas_x = np.atleast_2d(sigmas_x)
            (dim_x, dim_sigmas_x) = sigmas_x.shape
            assert dim_sigmas_x == W_c.shape[0], "Dimensions are wrong"
        try:
            (dim_y, dim_sigmas_y) = sigmas_y.shape
        except ValueError:  # sigmas_y is 1D
            sigmas_y = np.atleast_2d(sigmas_y)
            (dim_y, dim_sigmas_y) = sigmas_y.shape
            assert dim_sigmas_y == dim_sigmas_x, "Dimensions are wrong"
        
        #NB: could/should be changed to matrix product
        #Calculate cross-covariance -
        P_xy = sum([Wc_i*np.outer(sig_x,sig_y) for Wc_i, sig_x, sig_y in zip(W_c, sigmas_x.T, sigmas_y.T)])
        assert (dim_x, dim_y) == P_xy.shape
        return P_xy
    
    def correlation_from_covariance(self, cov, sigmas = None):
        """
        Calculate correlation matrix from a covariance matrix

        Parameters
        ----------
        cov : TYPE np.array((dim_p, dim_p))
            DESCRIPTION. Covariance matrix
        sigmas : TYPE Optional, defualt is None
            DESCRIPTION. Standard deviation. If None is supplied, it calculates the exact standard deviation. If it is supplied, it must be a np.array((dim_p,))

        Returns
        -------
        corr : TYPE np.array((dim_p, dim_p))
            DESCRIPTION. Correlation matrix

        """
        if sigmas is None: #calculate exact standard deviation matrix
            var = np.diag(cov)
            if (var <= 0).any():
                print(f"Negative variance, {var}")
                print(f"Negative variance, changing this now. Var = {var}")    
                var.setflags(write = True)
                var[var < 0] = 1e-10
                
            sigmas = np.sqrt(var)
        assert sigmas.ndim == 1
        dim_p = sigmas.shape[0]
        
        
        #Create sigma_cross_mat = [[s1s1, s1s2 ,.., s1sp],
        # [s2s1, s2s2,...,s2sp],
        # [sps1, sps2,..,spsp]]
        sigma_cross_mat = np.outer(sigmas, sigmas)
        corr = np.divide(cov, sigma_cross_mat) #element wise division
        return corr, sigmas
    
    def correlation_from_cross_covariance(self, Pxy, sig_x, sig_y):
        #Create sigma_mat = [[sx1sy1,.., sx1syy],
        # [sx2sy1,...,sx2syy],
        # [sxxsy1,..,sxxsyy]]
        dim_x = sig_x.shape[0]
        dim_y = sig_y.shape[0]
        assert (dim_x, dim_y) == Pxy.shape
        
        sigma_cross_mat = np.outer(sig_x, sig_y)
        assert sigma_cross_mat.shape == (dim_x, dim_y) 
        
        cross_corr = np.divide(Pxy, sigma_cross_mat) #element wise division
        return cross_corr
    
  

class UKF_additive_noise(UKFBase):
    
    def __init__(self, x0, P0, fx, hx, points_x, Q, R, 
                 w_mean = None, v_mean = None, name=None, check_negative_sigmas = False):
        """
        Create a Kalman filter. IMPORTANT: Additive white noise is assumed!

        """
        super().__init__(fx, hx, points_x, Q, R, 
                     w_mean = w_mean, v_mean = v_mean, name = name, check_negative_sigmas = check_negative_sigmas)
        
        dim_x = x0.shape[0]
        assert x0.ndim == 1, f"x0 should be 1d array, it is {x0.ndim}"
        assert P0.ndim == 2, f"P0 should be 2d array, it is {P0.ndim}"
        assert (dim_x, dim_x) == P0.shape #check input
        assert (P0 == P0.T).all() #symmtrical
        
        #set init values
        self.x_post = x0
        self.P_post = P0
        self.x_prior = self.x_post.copy()
        self.P_prior = self.P_post.copy()
        self._dim_x = dim_x

        #as the noise is additive, dim_x = dim_w, dim_y = dim_v
        assert self._dim_x == self._dim_w
        self._dim_y = self._dim_v
        
        # create sigma-points
        self.sigmas_raw_fx = np.zeros((self._dim_x, self._num_sigmas_x))
        
        # sigma-points propagated through fx to form prior distribution
        self.sigmas_prop = np.zeros((self._dim_x, self._num_sigmas_x))
        
        # sigma-points based on prior distribution
        self.sigmas_raw_hx = np.zeros((self._dim_x, self._num_sigmas_x))
        
        # sigma-points propagated through measurement equation. Form posterior distribution
        self.sigmas_meas = np.zeros((self._dim_y, self._num_sigmas_x))

        self.y_res = np.zeros((self._dim_y, 1))           # residual
        self.y = np.array([[None]*self._dim_y]).T  # measurement
    

    def predict(self, UT=None, kwargs_sigma_points={}, fx=None, w_mean = None, Q = None, **fx_args):
        r"""
        Performs the predict step of the UKF. On return, self.x_prior and
        self.P_prior contain the predicted state (x) and covariance (P). '

        Important: this MUST be called before update() is called for the first
        time.
        
        
        Solves the equation
        wk = fx(x, p) - fx(x_post, E[p])
        fx(x,p) = fx(x_post, E[p]) + wk

        Parameters
        ----------

        fx : callable f(x, **fx_args), optional
            State transition function. If not provided, the default
            function passed in during construction will be used.

        UT : function(sigmas, Wm, Wc, kwargs_sigma_points), optional
            Optional function to compute the unscented transform for the sigma
            points passed. If the points are GenUT, you can pass 3rd and 4th moment through kwargs_sigma_points (see description of sigma points class for details)

    

        **fx_args : keyword arguments
            optional keyword arguments to be passed into f(x).
        """

        if fx is None:
            fx = self.fx
        
        if w_mean is None:
            w_mean = self.w_mean
        
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self._dim_x) * Q

        if UT is None:
            UT = unscented_transform.unscented_transformation_gut

        # calculate sigma points for given mean and covariance for the states
        self.sigmas_raw_fx, self.Wm_x, self.Wc_x, P_sqrt = self.points_fn_x.compute_sigma_points(
            self.x_post, self.P_post, **kwargs_sigma_points)

        # propagate all the sigma points through fx
        self.sigmas_prop = self.compute_transformed_sigmas(
            self.sigmas_raw_fx, fx, **fx_args)

        # pass the propagated sigmas of the states (not the augmented states) through the unscented transform to compute prior
        self.x_prior, self.P_prior = UT(
            self.sigmas_prop, self.Wm_x, self.Wc_x)
        
        #add process noise
        self.x_prior += w_mean
        self.P_prior += Q

    def update(self, y, R=None, v_mean = None, UT=None, hx=None, kwargs_sigma_points={}, **hx_args):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        y : numpy.array of shape (dim_y)
            measurement vector

        R : numpy.array((dim_y, dim_y)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.
        v_mean : numpy.array((dim_y,)), optional
            Mean of measurement noise. If provided, it is added to self.y_pred

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. 

        hx : callable h(x, **hx_args), optional
            Measurement function. If not provided, the default
            function passed in during construction will be used.

        **hx_args : keyword argument
            arguments to be passed into h(x) after x -> h(x, **hx_args)
        """

        if y is None:
            self.y = np.array([[None]*self._dim_y]).T
            self.x_post = self.x_prior.copy()
            self.P_post = self.P_prior.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform.unscented_transformation_gut

        if v_mean is None:
            v_mean = self.v_mean
            
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self._dim_y) * R

        # recreate sigma points
        (self.sigmas_raw_hx,
         self.Wm, self.Wc,
         P_sqrt) = self.points_fn_x.compute_sigma_points(self.x_prior,
                                                       self.P_prior,
                                                       **kwargs_sigma_points
                                                       )

        # send sigma points through measurement equation
        self.sigmas_meas = self.compute_transformed_sigmas(
            self.sigmas_raw_hx, hx, **hx_args)

        # compute mean and covariance of the predicted measurement
        y_pred, Py_pred = UT(self.sigmas_meas, self.Wm, self.Wc)
        
        # add measurement noise
        y_pred += v_mean
        Py_pred += R 
        self.y_pred = y_pred
        self.Py_pred = Py_pred.copy()

        # Innovation term of the UKF
        self.y_res = y - y_pred
        
        #Kalman gain. Start with cross_covariance
        Pxy = self.cross_covariance(self.sigmas_raw_hx - self.x_prior.reshape(-1,1),
                                    self.sigmas_meas - y_pred.reshape(-1,1), self.Wc)
        self.Pxy = Pxy

        # Kalman gain
        # solve K@Py_pred = P_xy <=> PY_pred.T @ K.T = P_xy.T
        self.K = scipy.linalg.solve(Py_pred.T, Pxy.T, assume_a = "pos").T
        # self.K = np.linalg.solve(Py_pred.T, Pxy.T).T
        # self.K = np.linalg.lstsq(Py_pred.T, Pxy.T)[0].T #also an option
        assert self.K.shape == (self._dim_x, self._dim_y)

        # calculate posterior
        self.x_post = self.x_prior + self.K @ self.y_res
        self.P_post = self.P_prior - self.K @ Py_pred @ self.K.T
