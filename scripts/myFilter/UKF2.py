# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:17:56 2022

@author: halvorak
"""

from . import unscented_transform

from copy import deepcopy
# from math import log, exp, sqrt
import sys
import numpy as np
# from numpy import eye, zeros, dot, isscalar, outer
# from scipy.linalg import cholesky
import scipy.linalg 

class UnscentedKalmanFilter(object):
    r"""
    Implements the UKF


    Parameters
    ----------

    dim_x : int
        Number of state variables for the filter.


    dim_y : int
        Number of of measurements


    hx : function(x,**hx_args)
        Measurement function. Converts state vector x into a measurement
        vector of shape (dim_y,).

    fx : function(x,**fx_args)
        Propagation of states from current time step to the next.

    points : class
        Class which computes the sigma points and weights for a UKF
        algorithm. You can vary the UKF implementation by changing this
        class. 
        
    sqrt_fn : callable(ndarray), default=None (implies scipy.linalg.sqrtm)
        Defines how we compute the square root of a matrix, which has
        no unique answer. Principal matrix square root is the default choice. Typically the alternative is Cholesky decomposition. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix. Daid et al recommends principal matrix square root


    

    Attributes
    ----------

    x : numpy.array(dim_x)
        state estimate vector

    P : numpy.array(dim_x, dim_x)
        covariance estimate matrix

    x_prior : numpy.array(dim_x)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.

    x_post : numpy.array(dim_x)
        Posterior (updated) state estimate. Read Only.

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.

    y : ndarray
        Last measurement used in update(). Read only.

    R : numpy.array(dim_y, dim_y)
        measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        process noise matrix

    K : numpy.array
        Kalman gain

    y_res : numpy.array
        innovation residual


    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead:

        .. code-block:: Python

            kf.inv = np.linalg.pinv



    References
    ----------

    .. [1] Julier, Simon J. "The scaled unscented transformation,"
        American Control Converence, 2002, pp 4555-4559, vol 6.

        Online copy:
        https://www.cs.unc.edu/~welch/kalman/media/pdf/ACC02-IEEE1357.PDF

    .. [2] E. A. Wan and R. Van der Merwe, “The unscented Kalman filter for
        nonlinear estimation,” in Proc. Symp. Adaptive Syst. Signal
        Process., Commun. Contr., Lake Louise, AB, Canada, Oct. 2000.

        Online Copy:
        https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf

    .. [3] S. Julier, J. Uhlmann, and H. Durrant-Whyte. "A new method for
           the nonlinear transformation of means and covariances in filters
           and estimators," IEEE Transactions on Automatic Control, 45(3),
           pp. 477-482 (March 2000).

    .. [4] E. A. Wan and R. Van der Merwe, “The Unscented Kalman filter for
           Nonlinear Estimation,” in Proc. Symp. Adaptive Syst. Signal
           Process., Commun. Contr., Lake Louise, AB, Canada, Oct. 2000.

           https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf

    .. [5] Wan, Merle "The Unscented Kalman Filter," chapter in *Kalman
           Filtering and Neural Networks*, John Wiley & Sons, Inc., 2001.

    .. [6] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
           Inference in Dynamic State-Space Models" (Doctoral dissertation)
    """

    def __init__(self, dim_x, dim_y, hx, fx, points,
                 sqrt_fn=None, name = None):
        """
        Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        """

        #pylint: disable=too-many-arguments

        self.x_prior = np.zeros((dim_x, 1))
        self.P_prior = np.eye(dim_x)
        self.x_post = np.copy(self.x_prior)
        self.P_post = np.copy(self.P_prior)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_y)
        self._dim_x = dim_x
        self._dim_y = dim_y
        self.points_fn = points
        self._num_sigmas = points.num_sigma_points()
        self.hx = hx
        self.fx = fx
        self._name = name #object name, handy when printing from within class

        if sqrt_fn is None:
            self.msqrt = scipy.linalg.sqrtm
        else:
            self.msqrt = sqrt_fn

        # weights for the means and covariances.
        # self.Wm, self.Wc = points.Wm, points.Wc
        # try:
        #     self.Wm, self.Wc = points.Wm, points.Wc
        # except AttributeError:
        #     print(f"At {type(self)} __init__, the points.Wm and/or points.Wc did not exist. Points_fn are {type(points)}. Assuming this is a property of the specific sigma points, and continuing the script.")


        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update

        self.sigmas_raw_fx = np.zeros((self._dim_x, self._num_sigmas))
        self.sigmas_raw_hx = np.zeros((self._dim_x, self._num_sigmas))
        self.sigmas_prop = np.zeros((self._dim_x, self._num_sigmas))
        self.sigmas_meas = np.zeros((self._dim_y, self._num_sigmas))

        self.K = np.zeros((dim_x, dim_y))    # Kalman gain
        self.y_res = np.zeros((dim_y, 1))           # residual
        self.y = np.array([[None]*dim_y]).T  # measurement
        # self.S = np.zeros((dim_z, dim_y))    # system uncertainty
        # self.SI = np.zeros((dim_z, dim_y))   # inverse system uncertainty

        self.inv = np.linalg.inv


    def predict(self, UT=None, Q = None, kwargs_sigma_points = {}, fx=None, **fx_args):
        r"""
        Performs the predict step of the UKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P). '

        Important: this MUST be called before update() is called for the first
        time.

        Parameters
        ----------

        fx : callable f(x, dt, **fx_args), optional
            State transition function. If not provided, the default
            function passed in during construction will be used.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        **fx_args : keyword arguments
            optional keyword arguments to be passed into f(x).
        """
        
        if fx is None:
            fx = self.fx
        
        if Q is None:
            Q = self.Q

        if UT is None:
            UT = unscented_transform.unscented_transformation_gut
        
        
        # calculate sigma points for given mean and covariance
        self.sigmas_raw_fx, self.Wm, self.Wc, P_sqrt = self.points_fn.compute_sigma_points(self.x_post, self.P_post, **kwargs_sigma_points)
        
        # print(f"sigmas_raw_fx: {self.sigmas_raw_fx.shape}")
        
        # self.sigmas_prop = self.compute_process_sigmas(self.sigmas_raw_fx, fx = fx, **fx_args)
        self.sigmas_prop = self.compute_transformed_sigmas(self.sigmas_raw_fx, fx, **fx_args)
        
        #and pass sigmas through the unscented transform to compute prior
        self.x_prior, self.P_prior = UT(self.sigmas_prop, self.Wm, self.Wc)
        self.P_prior += Q #add process noise

    # def compute_process_sigmas(self, sigmas_raw_fx, fx=None, **fx_args):
    #     """
    #     computes the values of sigmas_f. Normally a user would not call
    #     this, but it is useful if you need to call update more than once
    #     between calls to predict (to update for multiple simultaneous
    #     measurements), so the sigmas correctly reflect the updated state
    #     x, P.
    #     """

    #     if fx is None:
    #         fx = self.fx
        
    #     sigmas_prop = map(fx, sigmas_raw_fx.T)
    #     sigmas_prop = np.array(list(sigmas_prop)).T
    #     return sigmas_prop
    
    def compute_transformed_sigmas(self, sigmas_in, func, **func_args):
        sigmas_out = map(func, sigmas_in.T)
        sigmas_out = np.array(list(sigmas_out)).T
        return sigmas_out
        
        # for i, s in enumerate(sigmas):
        #     if ((any(s<=1e-8)) and (self._name != "qf")):
        #     # if any(s<=1e-8):
        #         # print(f"{self._name} has low value of sigma point: s = {s}")
        #         raise myExceptions.NegativeSigmaPoint(f"Negative sigma point detected in {self._name}, s = {s}")
                
        #     self.sigmas_f[i] = fx(s, dt, **fx_args)
            
    def update(self, y, R=None, UT=None, hx=None, kwargs_sigma_points = {}, **hx_args):
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

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self._dim_y) * R
            
        #recreate sigma points
        self.sigmas_raw_hx, self.Wm, self.Wc, P_sqrt = self.points_fn.compute_sigma_points(self.x_prior, self.P_prior, **kwargs_sigma_points)
        # self.sigmas_f = self.points_fn.sigma_points(self.x, self.P)

        # # pass prior sigmas through h(x) to get measurement sigmas
        # # the shape of sigmas_h will vary if the shape of z varies, so
        # # recreate each time
        # sigmas_h = []
        # for s in self.sigmas_f:
        #     sigmas_h.append(hx(s, **hx_args))

        # self.sigmas_h = np.atleast_2d(sigmas_h)
        
        #send sigma points through measurement equation
        self.sigmas_meas = self.compute_transformed_sigmas(
            self.sigmas_raw_hx, hx, **hx_args)
        
        #compute mean and covariance of the predicted measurement
        y_pred, Py_pred = UT(self.sigmas_meas, self.Wm, self.Wc)
        Py_pred += R #add measurement noise
        # Py_pred_inv = self.inv(Py_pred)
        
        Pxy = self.cross_covariance(self.x_prior, y_pred, self.sigmas_raw_hx, self.sigmas_meas, self.Wc)
        
        #Kalman gain
        # self.K = Pxy @ Py_pred_inv #should be better way - avoid inverting Py OR use the Cholesky factorization (if self.sqrt_fn is Cholesky) to calculate Py_pred_inv
        
        #solve K@Py_pred = P_xy <=> PY_pred.T @ K.T = P_xy.T
        # self.K = np.linalg.lstsq(Py_pred.T, Pxy.T)[0].T #also an option
        self.K = np.linalg.solve(Py_pred.T, Pxy.T).T
        assert self.K.shape == (self._dim_x, self._dim_y)
        
        self.y_res = y - y_pred #innovation term
        
        #calculate posterior
        self.x_post = self.x_prior + self.K @ self.y_res
        self.P_post = self.P_prior - self.K @ Py_pred @ self.K.T
        

        # # mean and covariance of prediction passed through unscented transform
        # # print(f"sigmas_h: {self.sigmas_h.shape}\n",
        # #       f"Wm: {self.Wm.shape}")
        # zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)
        # self.SI = self.inv(self.S)

        # # compute cross variance of the state and the measurements
        # Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)


        # self.K = dot(Pxz, self.SI)        # Kalman gain
        # self.y = self.residual_z(z, zp)   # residual

        # # update Gaussian state estimate (x, P)
        # self.x = self.state_add(self.x, dot(self.K, self.y))
        # self.P = self.P - dot(self.K, dot(self.S, self.K.T))

        # # save measurement and posterior state
        # self.z = deepcopy(z)
        # self.x_post = self.x.copy()
        # self.P_post = self.P.copy()

        # # set to None to force recompute
        # self._log_likelihood = None
        # self._likelihood = None
        # self._mahalanobis = None
        
    def cross_covariance(self, x_mean, y_mean, sigmas_x, sigmas_y, W_c):
        """
        Cross-covariance between two probability distribution x,y

        Parameters
        ----------
        x_mean : TYPE np.array(dim_x,)
            DESCRIPTION. Mean of the distribution x
        y_mean : TYPE np.array(dim_y,)
            DESCRIPTION. Mean of the distribution y
        sigmas_x : TYPE np.array((dim_x, dim_sigmas))
            DESCRIPTION. Sigma-points created from the x-distribution
        sigmas_y : TYPE np.array((dim_y, dim_sigmas))
            DESCRIPTION. Sigma-points created from the y-distribution
        W_c : TYPE np.array(dim_sigmas,)
            DESCRIPTION. Weights to compute the covariance

        Returns
        -------
        P_xy : TYPE np.array((dim_x, dim_y))
            DESCRIPTION. Cross-covariance between x and y

        """
        try:
            (dim_x, dim_sigmas_x) = sigmas_x.shape
        except ValueError: #sigmas_x is 1D
            sigmas_x = np.atleast_2d(sigmas_x)
            (dim_x, dim_sigmas_x) = sigmas_x.shape 
            assert dim_sigmas_x == W_c.shape[0], "Dimensions are wrong"
        try:
            (dim_y, dim_sigmas_y) = sigmas_y.shape
        except ValueError: #sigmas_y is 1D
            sigmas_y = np.atleast_2d(sigmas_y)
            (dim_y, dim_sigmas_y) = sigmas_y.shape 
            assert dim_sigmas_y == dim_sigmas_x, "Dimensions are wrong"
        # dim_x, dim_sigmas_x = sigmas_x.shape
        # dim_y, dim_sigmas_y = sigmas_y.shape
        # assert dim_sigmas_x == dim_sigmas_y, f"dim_sigmas_x != dim_sigmas_y: {dim_sigmas_x} != {dim_sigmas_y}"
        
        P_xy = np.zeros((dim_x, dim_y))
        for i in range(dim_sigmas_x):
            P_xy += W_c[i]*((sigmas_x[:, i] - x_mean).reshape(-1,1)
                           @ (sigmas_y[:, i] - y_mean).reshape(-1,1).T)
        return P_xy

    # def cross_variance(self, x, z, sigmas_f, sigmas_h):
    #     """
    #     Compute cross variance of the state `x` and measurement `z`.
    #     """

    #     Pxz = zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
    #     N = sigmas_f.shape[0]
    #     for i in range(N):
    #         dx = self.residual_x(sigmas_f[i], x)
    #         dz = self.residual_z(sigmas_h[i], z)
    #         Pxz += self.Wc[i] * outer(dx, dz)
    #     return Pxz

