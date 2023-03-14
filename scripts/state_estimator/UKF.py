# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:17:56 2022

@author: halvorak
"""

from . import unscented_transform

# from copy import deepcopy
import numpy as np
import scipy.linalg
# import casadi as cd
# from numba import jit
# from numba import jitclass          # import the decorator
# from numba import int32, float32    # import the types

# spec = [
#     ('value', int32),               # a simple scalar field
#     ('array', float32[:]),          # an array field
# ]

# @jitclass(spec)
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
                 w_mean = None, v_mean = None, name=None, check_negative_sigmas = True):
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
                 w_mean = None, v_mean = None, name=None):
        """
        Create a Kalman filter. IMPORTANT: Additive white noise is assumed!

        """
        super().__init__(fx, hx, points_x, Q, R, 
                     w_mean = None, v_mean = None, name=None)
        
        dim_x = x0.shape[0]
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

      


class NUKF_additive_noise_corr_lim(UKFBase):
    """
    NUKF, using the Normalized Unscented Transformation (NUT), with possibilities to have limits on the correlation. IMPORTANT: Additive white noise is assumed!
    """
    def __init__(self, x0, P0, fx, hx, points_x, Q, R, 
                 w_mean = None, v_mean = None, 
                 corr_post_lim = np.inf, corr_prior_lim = np.inf, 
                 corr_y_lim = np.inf, corr_xy_lim = np.inf, UT = None, name=None):
        
        super().__init__(fx, hx, points_x, Q, R, 
                     w_mean = w_mean, v_mean = v_mean, name = name)
        
        dim_x = x0.shape[0]
        assert (dim_x, dim_x) == P0.shape #check input
        assert (P0 == P0.T).all() #symmtrical
        
        corr0, std_dev0 = self.correlation_from_covariance(P0)
        assert (corr0 == corr0.T).all() #check function self.correlation_from_covariance(P0) works as intended
        
        #set __init__ values
        self.x_post = x0
        self.std_dev_post = np.diag(std_dev0) #(dim_x, dim_x)
        self.corr_post = corr0
        self.x_prior = self.x_post.copy()
        self.std_dev_prior = self.std_dev_post.copy()
        self.corr_prior = self.corr_post.copy()
        self._dim_x = dim_x
        self.P_dummy = np.nan*np.zeros((dim_x, dim_x)) #dummy matrix, sent to functions which calculate P_sqrt
        
        #Need initial standard deviation of y. Set it equal to the noise values
        self.std_dev_y = np.diag(np.sqrt(np.diag(R))) #elementwise standard deviation of the diagonals (R may have values on the off-diagonals)

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
    
        if UT is None:
            self.UT = unscented_transform.normalized_unscented_transformation_additive_noise
        else:
            self.UT = UT
    
        #whether or not we should check limits for correlatio
        if corr_post_lim is np.inf:
            self.check_corr_post_lim = False
        else:
            self.check_corr_post_lim = True
        
        if corr_prior_lim is np.inf:
            self.check_corr_prior_lim = False
        else:
            self.check_corr_prior_lim = True
       
        if corr_y_lim is np.inf:
            self.check_corr_y_lim = False
        else:
            self.check_corr_y_lim = True
       
        if corr_xy_lim is np.inf:
            self.check_corr_xy_lim = False
        else:
            self.check_corr_xy_lim = True
        
        #set limits for correlation
        if np.isscalar(corr_post_lim):
            assert ~np.isnan(corr_post_lim), "np.nan invalid input"
            if corr_post_lim < 0:
                corr_post_lim = -corr_post_lim
            corr_post_lim = np.array([-corr_post_lim, corr_post_lim])
        
        if np.isscalar(corr_prior_lim):
            assert ~np.isnan(corr_prior_lim), "np.nan invalid input"
            if corr_prior_lim < 0:
                corr_prior_lim = -corr_prior_lim
            corr_prior_lim = np.array([-corr_prior_lim, corr_prior_lim])
        
        if np.isscalar(corr_y_lim):
            assert ~np.isnan(corr_y_lim), "np.nan invalid input"
            if corr_y_lim < 0:
                corr_y_lim = -corr_y_lim
            corr_y_lim = np.array([-corr_y_lim, corr_y_lim])
        
        
        if np.isscalar(corr_xy_lim):
            assert ~np.isnan(corr_xy_lim), "np.nan invalid input"
            if corr_xy_lim < 0:
                corr_xy_lim = -corr_xy_lim
            corr_xy_lim = np.array([-corr_xy_lim, corr_xy_lim])
        
        self.corr_post_lim = corr_post_lim
        self.corr_prior_lim = corr_prior_lim
        self.corr_y_lim = corr_y_lim
        self.corr_xy_lim = corr_xy_lim
        
    
    def predict(self, UT=None, kwargs_sigma_points={}, fx=None, w_mean = None, Q = None, fx_args = [], fx_kwargs = {}, kwargs_UT = {}):
        if fx is None:
            fx = self.fx
        
        if w_mean is None:
            w_mean = self.w_mean
        
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self._dim_w) * Q
        self.Q = Q
        if UT is None:
            UT = self.UT
        
        #calculate the square-root of the covariance matrix by using standard deviations and correlation matrix
        corr_sqrt = self.msqrt(self.corr_post)
        P_sqrt = self.std_dev_post @ corr_sqrt
        
        # calculate sigma points for given mean and covariance for the states
        (self.sigmas_raw_fx, self.Wm_x, 
         self.Wc_x, P_sqrt) = self.points_fn_x.compute_sigma_points(self.x_post, 
                                                                    self.P_dummy, 
                                                                    P_sqrt = P_sqrt,
                                                                    **kwargs_sigma_points)

        # propagate all the sigma points through fx
        # self.sigmas_prop = self.compute_transformed_sigmas2(
        #     self.sigmas_raw_fx, fx, func_args = fx_args, func_kwargs = fx_kwargs)
        
        #propagate here to easier do troubleshooting
        for i in range(self.sigmas_raw_fx.shape[1]):
            try:
                self.sigmas_prop[:, i] = fx(self.sigmas_raw_fx[:, i], *fx_args, **fx_kwargs)
            except RuntimeError as err:
                print(f"Error in predict at sigma point {i}. Values of inputs are {self.sigmas_raw_fx[:, i]}\n\n")
                raise err
        
        self.x_prior, self.corr_prior, self.std_dev_prior = UT(self.sigmas_prop, self.Wm_x, self.Wc_x, Q, **kwargs_UT)
        
        self.x_prior += w_mean #add mean of the noise. 
        
        if self.check_corr_prior_lim:
            self.corr_prior = self.corr_limit(self.corr_prior,
                                              self.corr_prior_lim)
        
        
        return None
    def nlp_update(self, y, nlp_update, pk_nlp_func, UT = None, R = None, v_mean = None, lbx = [], ubx = [], lbg = [], ubg = [], kwargs_sigma_points={}, calc_y_pred = True, hx_args = [], kwargs_UT = {}):
        
        #pk_nlp_func(R_inv_array, sig_f, P_prior_inv_array). Takes the indicated arguments and returns a cd.vertcat() with all arguments required for the nlp_update
        
        
        if y is None:
            self.y = np.array([[None]*self._dim_y]).T
            self.x_post = self.x_prior.copy()
            self.corr_post = self.corr_prior.copy()
            self.std_dev_post = self.std_dev_prior.copy()
            return

        if UT is None:
            UT = self.UT

        if R is None:
            R = self.R
            R_inv_array = self.R_inv_array
        else:
            R_inv_array = scipy.linalg.inv(R).flatten()
        
        if v_mean is None:
            v_mean = np.zeros(self._dim_y)
        
        #Calculate P_sqrt
        corr_sqrt = self.msqrt(self.corr_prior)
        P_sqrt = self.std_dev_prior @ corr_sqrt

        # recreate sigma points
        (self.sigmas_raw_hx,
         self.Wm_x, self.Wc_x,
         P_sqrt) = self.points_fn_x.compute_sigma_points(self.x_prior, 
                                                         self.P_dummy, 
                                                         P_sqrt = P_sqrt, 
                                                         **kwargs_sigma_points
                                                         )
        #prepare for NLP update
        #invert the prior covariance matrix. NB: can be done faster since we already have computed P_sqrt (which is most probably Cholesky decomposition)
        P_prior = self.std_dev_prior @ self.corr_prior @ self.std_dev_prior
        P_prior_inv = scipy.linalg.inv(P_prior)
        P_prior_inv_array = P_prior_inv.flatten() #input parameter for nlp-function
        
        self.sigmas_post = np.zeros((self._dim_x, self._num_sigmas_x)) #allocate space
        for i in range(self._num_sigmas_x):
            sig_f = self.sigmas_raw_hx[:, i]
            pk = pk_nlp_func(R_inv_array, sig_f, P_prior_inv_array, v_mean)
            # pk = cd.vertcat(y, R_inv_array, sig_f, P_prior_inv_array, omega)
            sol_nlp = nlp_update(x0 = sig_f, p = pk, lbx = lbx, ubx = ubx, lbg = lbg, ubg = ubg)
            self.sol_nlp = sol_nlp
            self.sigmas_post[:, i] = np.array(sol_nlp["x"]).flatten()

        # mean and covariance of prediction passed through unscented transform
        self.x_post, self.corr_post, self.std_dev_post = UT(self.sigmas_post, self.Wm_x, self.Wc_x, noise_mat = None, **kwargs_UT)
        
        #check if we should add constraints to the correlation term
        if self.check_corr_post_lim:
            self.corr_post = self.corr_limit(self.corr_post,
                                              self.corr_post_lim)
        
        #NB: This is not required for the nlp-update scheme. It is simply an option for diagnostics (to see what is the predicted measurements)
        if calc_y_pred:
            # send sigma points through measurement equation
            self.sigmas_meas = self.compute_transformed_sigmas2(self.sigmas_raw_hx, 
                                                               self.hx, func_args = hx_args)

            # pass the propagated sigmas of the states through the unscented transform to compute the predicted measurement, corr_y and std_dev_y
            self.y_pred, self.corr_y, self.std_dev_y = UT(self.sigmas_meas, self.Wm_x, self.Wc_x, R, **kwargs_UT)
        else:    
            #not appliccable for the nlp update
            self.y_pred = np.nan
            self.std_dev_y = np.eye(self._dim_y)*np.nan
            self.corr_y = np.eye(self._dim_y)*np.nan
        
        
        if self.check_corr_post_lim:
            self.corr_post = self.corr_limit(self.corr_post,
                                             self.corr_post_lim)
        
        return None
        
    def update(self, y, R=None, v_mean = None, UT=None, hx=None, kwargs_sigma_points={}, hx_args=[], kwargs_UT = {}):
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
            self.corr_post = self.corr_prior.copy()
            self.std_dev_post = self.std_dev_prior.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = self.UT

        if v_mean is None:
            v_mean = self.v_mean
            
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self._dim_y) * R
        
        #Calculate P_sqrt
        corr_sqrt = self.msqrt(self.corr_prior)
        P_sqrt = self.std_dev_prior @ corr_sqrt

        # recreate sigma points
        (self.sigmas_raw_hx,
         self.Wm_x, self.Wc_x,
         P_sqrt) = self.points_fn_x.compute_sigma_points(self.x_prior, 
                                                         self.P_dummy, 
                                                         P_sqrt = P_sqrt, 
                                                         **kwargs_sigma_points
                                                         )

        # send sigma points through measurement equation
        self.sigmas_meas = self.compute_transformed_sigmas2(self.sigmas_raw_hx, 
                                                           hx, func_args = hx_args)

        # pass the propagated sigmas of the states through the unscented transform to compute the predicted measurement, corr_y and std_dev_y
        self.y_pred, self.corr_y, self.std_dev_y = UT(self.sigmas_meas, self.Wm_x, self.Wc_x, R, **kwargs_UT)
        
        self.y_pred += v_mean #add mean of the noise. 
        
        #check if we should add constraints to the correlation term
        if self.check_corr_y_lim:
            self.corr_y = self.corr_limit(self.corr_y, self.corr_y_lim)
        

        # Innovation term of the UKF
        self.y_res = y - self.y_pred
        self.std_dev_y_inv = np.diag([1/sig_y for sig_y in np.diag(self.std_dev_y)])#inverse of diagonal matrix is inverse of each diagonal element - to be multiplied with innovation term
        
        #Obtain the cross_covariance
        sig_x_norm = np.divide(self.sigmas_raw_hx - self.x_prior.reshape(-1,1),
                               np.diag(self.std_dev_prior).reshape(-1,1))
        sig_y_norm = np.divide(self.sigmas_meas - self.y_pred.reshape(-1,1),
                               np.diag(self.std_dev_y).reshape(-1,1))
        
        self.corr_xy = self.cross_covariance(sig_x_norm, sig_y_norm, self.Wc_x)
        
        #check if we should add constraints to the cross-correlation term
        if self.check_corr_xy_lim:
            self.corr_xy = self.corr_limit(self.corr_xy,
                                              self.corr_xy_lim, cross_corr = True)
        #Kalman gain
        self.K = scipy.linalg.solve(self.corr_y, self.corr_xy.T, assume_a = "pos").T
        assert self.K.shape == (self._dim_x, self._dim_y)

        # calculate posterior
        self.x_post = self.x_prior + self.std_dev_prior @ self.K @ self.std_dev_y_inv @ self.y_res
        
        #obtain posterior correlation and standard deviation
        self.corr_post = self.corr_prior - self.K @ self.corr_xy.T # this is not the true posterior - it is scaled with std_dev_prior
        
        #find the true correlation (values between [-1,1]) and the update factor for the standard deviation
        self.corr_post, std_dev_post_update = self.correlation_from_covariance(self.corr_post)
        
        #Get the posterior standard deviation
        self.std_dev_post = np.diag(std_dev_post_update*np.diag(self.std_dev_prior))
        
        #check if we should add constraints to the correlation term
        if self.check_corr_post_lim:
            self.corr_post = self.corr_limit(self.corr_post,
                                              self.corr_post_lim)
        
    @staticmethod    
    def corr_limit(corr, corr_lim, cross_corr = False):
        #input: correlation matrix. Check wheter elements are above or below limit and if they are, set the correlation to that limit
        
        ##Try to do it fast by using numpy functions
        # idx = np.tril_indices_from(corr, k = -1)
        # idx_lower = (corr[idx] < corr_lim[0])
        # idx_higher = (corr[idx] > corr_lim[1])
        # if idx_lower.any():
        #     idx_l = np.nonzero(idx_lower)[0]
        #     idx_ut = np.triu_indices_from(corr, k = 1)
        #     corr[idx[idx_l]] = corr_lim[0]
                
            
        # if idx_higher.any():
        #     print("high")
        
        #correct way, can be speeded up by using default numpy functions
        dim_x, dim_y = corr.shape
        if not cross_corr:
            #symmetrical matrix - check only lower triangle and change matrix values two places
            assert dim_x == dim_y, "Dimension mismatch for normal correlation matrix"
            for r in range(1,dim_x):
                for c in range(r):
                    if corr[r,c] < corr_lim[0]:
                        corr[r,c] = corr_lim[0]
                        corr[c,r] = corr_lim[0]
                    if corr[r,c] > corr_lim[1]:
                        corr[r,c] = corr_lim[1]
                        corr[c,r] = corr_lim[1]
        else:
            #NOT symmetrical matrix - check whole matrix and change only one limit
            for r in range(dim_x):
                for c in range(dim_y):
                    if corr[r,c] < corr_lim[0]:
                        corr[r,c] = corr_lim[0]
                    if corr[r,c] > corr_lim[1]:
                        corr[r,c] = corr_lim[1]
        # print(corr)
        return corr
        
class UKF_nonlinear_noise(UKFBase):
    
    def __init__(self, x0, P0, fx, hx, points_xw, points_xv, Q, R, dim_y, 
                 w_mean = None, v_mean = None, name=None):
        """
        Create a Kalman filter. Assume Q-distribution is constant. We do NOT use the "smart" way of splitting up the matrix square-root (easier to not implement that)

        """
        super().__init__(fx, hx, points_xw, Q, R, 
                     w_mean = None, v_mean = None, name=None)
        
        #override behaviour for points_fn_x in super().__init__()
        del self.points_fn_x
        del self._num_sigmas_x
        self.points_fn_xw = points_xw
        self._num_sigmas_xw = points_xw.num_sigma_points()
        self.points_fn_xv = points_xv
        self._num_sigmas_xv = points_xv.num_sigma_points()
       
        
        dim_x = x0.shape[0]
        assert (dim_x, dim_x) == P0.shape #check input
        assert (P0 == P0.T).all() #symmtrical
        
        #set init values
        self.x_post = x0
        self.P_post = P0
        self.x_prior = self.x_post.copy()
        self.P_prior = self.P_post.copy()
        self._dim_x = dim_x
        self._dim_y = dim_y  
        self._dim_xw = self._dim_x + self._dim_w #augmented states- process noise
        self._dim_xv = self._dim_x + self._dim_v #augmented states- measurement noise
        
        #check shapes are set correctly
        assert self._dim_xw == points_xw.n
        assert self._dim_xv == points_xv.n
        
        #matrices for the xw-augemented covariance matrix. Use P_post here since that's what we will make the sigma-points based on
        self.P_xwa = scipy.linalg.block_diag(self.P_post, Q) #for obtaining the prior (model prediction step)
        self.P_xva = scipy.linalg.block_diag(self.P_prior, R) #for obtaining the posterior (measurement update step)
        
        # create sigma-points
        self.sigmas_raw_fx = np.zeros((self._dim_xw, self._num_sigmas_xw))
        
        # sigma-points propagated through fx to form prior distribution
        self.sigmas_prop = np.zeros((self._dim_x, self._num_sigmas_xw))
        
        # sigma-points based on prior distribution
        self.sigmas_raw_hx = np.zeros((self._dim_xv, self._num_sigmas_xv))
        
        # sigma-points propagated through measurement equation. Form posterior distribution
        self.sigmas_meas = np.zeros((self._dim_y, self._num_sigmas_xv))

        self.y_res = np.zeros((self._dim_y, 1))           # residual
        self.y = np.array([[None]*self._dim_y]).T  # measurement
    

    def predict(self, UT=None, kwargs_sigma_points={}, fx=None, w_mean = None, Q = None, **fx_args):
        r"""
        Performs the predict step of the UKF. On return, self.x_prior and
        self.P_prior contain the predicted state (x) and covariance (P). '

        

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
        
        if Q is not None:
            # must update the augmented covariance distribution
            self.P_xwa[self._dim_x:, self._dim_x:] = Q

        if UT is None:
            UT = unscented_transform.unscented_transformation_gut
            
        #base everything on the augmented xw-system
        self.xw_mean = np.hstack((self.x_post, w_mean))
        self.P_xwa[:self._dim_x, :self._dim_x] = self.P_post #update augmented covariance

        # calculate sigma points for given mean and covariance for the states
        self.sigmas_raw_fx, self.Wm_x, self.Wc_x, P_sqrt = self.points_fn_xw.compute_sigma_points(
            self.xw_mean, self.P_xwa, **kwargs_sigma_points)

        # propagate all the sigma points through fx
        self.sigmas_prop = self.compute_transformed_sigmas(
            self.sigmas_raw_fx, fx, **fx_args)

        # pass the propagated sigmas of the states (not the augmented states) through the unscented transform to compute prior
        self.x_prior, self.P_prior = UT(
            self.sigmas_prop, self.Wm_x, self.Wc_x)
        

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
            
        if R is not None:
            # must update the augmented covariance distribution
            self.P_xva[self._dim_x:, self._dim_x:] = R
           
        #base everything on the augmented xw-system
        self.xv_mean = np.hstack((self.x_prior, v_mean))
        self.P_xva[:self._dim_x, :self._dim_x] = self.P_prior #update augmented covariance

        # recreate sigma points
        (self.sigmas_raw_hx,
         self.Wm, self.Wc,
         P_sqrt) = self.points_fn_xv.compute_sigma_points(self.xv_mean,
                                                       self.P_xva,
                                                       **kwargs_sigma_points
                                                       )

        # send sigma points through measurement equation
        self.sigmas_meas = self.compute_transformed_sigmas(
            self.sigmas_raw_hx, hx, **hx_args)

        # compute mean and covariance of the predicted measurement
        y_pred, Py_pred = UT(self.sigmas_meas, self.Wm, self.Wc)
        
        self.y_pred = y_pred
        self.Py_pred = Py_pred.copy()

        # Innovation term of the UKF
        self.y_res = y - y_pred
        
        #Kalman gain. Start with cross_covariance - ER HER JEG FÅR PROBLEMER. Får nok P_xy inkluderer v. Må bytte self.x_prior og så slice matrisen slik at jeg får det riktig
        Pxy_a = self.cross_covariance(self.sigmas_raw_hx - self.xv_mean.reshape(-1,1),
                                    self.sigmas_meas - y_pred.reshape(-1,1), self.Wc)
        self.Pxy = Pxy_a[:self._dim_x,:]

        # Kalman gain
        # solve K@Py_pred = P_xy <=> PY_pred.T @ K.T = P_xy.T
        self.K = scipy.linalg.solve(Py_pred.T, self.Pxy.T, assume_a = "pos").T
        # self.K = np.linalg.solve(Py_pred.T, Pxy.T).T
        # self.K = np.linalg.lstsq(Py_pred.T, Pxy.T)[0].T #also an option
        assert self.K.shape == (self._dim_x, self._dim_y)

        # calculate posterior
        self.x_post = self.x_prior + self.K @ self.y_res
        self.P_post = self.P_prior - self.K @ Py_pred @ self.K.T

class NUKF_nonlinear_noise_corr_lim(UKFBase):
    """
    NUKF, using the Normalized Unscented Transformation (NUT), with possibilities to have limits on the correlation. IMPORTANT: Additive white noise is assumed!
    """
    def __init__(self, x0, P0, fx, hx, points_xw, points_xv, Q, R, dim_y, 
                     w_mean = None, v_mean = None, 
                     corr_post_lim = np.inf, corr_prior_lim = np.inf, 
                     corr_y_lim = np.inf, corr_xy_lim = np.inf, name=None):
        super().__init__(fx, hx, points_xw, Q, R, 
                     w_mean = w_mean, v_mean = v_mean, name = name)
        
        dim_x = x0.shape[0]
        assert (dim_x, dim_x) == P0.shape #check input
        assert (P0 == P0.T).all() #symmtrical
        
        corr0, std_dev0 = self.correlation_from_covariance(P0)
        assert (corr0 == corr0.T).all() #check function self.correlation_from_covariance(P0) works as intended
        
        #set __init__ values
        self.x_post = x0
        self.std_dev_post = np.diag(std_dev0) #(dim_x, dim_x)
        self.corr_post = corr0
        self.x_prior = self.x_post.copy()
        self.std_dev_prior = self.std_dev_post.copy()
        self.corr_prior = self.corr_post.copy()
        self._dim_x = dim_x
        self.P_dummy = np.nan*np.zeros((dim_x, dim_x)) #dummy matrix, sent to functions which calculate P_sqrt
        self._dim_y = dim_y
        
        #Need initial standard deviation of y. Set it equal to the noise values
        self.std_dev_y = np.diag(np.sqrt(np.diag(R))) #elementwise standard deviation of the diagonals (R may have values on the off-diagonals)

        
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
    
        
        #whether or not we should check limits for correlatio
        if corr_post_lim is np.inf:
            self.check_corr_post_lim = False
        else:
            self.check_corr_post_lim = True
        
        if corr_prior_lim is np.inf:
            self.check_corr_prior_lim = False
        else:
            self.check_corr_prior_lim = True
       
        if corr_y_lim is np.inf:
            self.check_corr_y_lim = False
        else:
            self.check_corr_y_lim = True
       
        if corr_xy_lim is np.inf:
            self.check_corr_xy_lim = False
        else:
            self.check_corr_xy_lim = True
        
        #set limits for correlation
        if np.isscalar(corr_post_lim):
            assert ~np.isnan(corr_post_lim), "np.nan invalid input"
            if corr_post_lim < 0:
                corr_post_lim = -corr_post_lim
            corr_post_lim = np.array([-corr_post_lim, corr_post_lim])
        
        if np.isscalar(corr_prior_lim):
            assert ~np.isnan(corr_prior_lim), "np.nan invalid input"
            if corr_prior_lim < 0:
                corr_prior_lim = -corr_prior_lim
            corr_prior_lim = np.array([-corr_prior_lim, corr_prior_lim])
        
        if np.isscalar(corr_y_lim):
            assert ~np.isnan(corr_y_lim), "np.nan invalid input"
            if corr_y_lim < 0:
                corr_y_lim = -corr_y_lim
            corr_y_lim = np.array([-corr_y_lim, corr_y_lim])
        
        
        if np.isscalar(corr_xy_lim):
            assert ~np.isnan(corr_xy_lim), "np.nan invalid input"
            if corr_xy_lim < 0:
                corr_xy_lim = -corr_xy_lim
            corr_xy_lim = np.array([-corr_xy_lim, corr_xy_lim])
        
        self.corr_post_lim = corr_post_lim
        self.corr_prior_lim = corr_prior_lim
        self.corr_y_lim = corr_y_lim
        self.corr_xy_lim = corr_xy_lim
        
    
    def predict(self, UT=None, kwargs_sigma_points={}, fx=None, w_mean = None, Q = None, **fx_args):
        if fx is None:
            fx = self.fx
        
        if w_mean is None:
            w_mean = self.w_mean
        
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self._dim_w) * Q
        
        if UT is None:
            UT = unscented_transform.normalized_unscented_transformation_additive_noise
        
        #calculate the square-root of the covariance matrix by using standard deviations and correlation matrix
        corr_sqrt = self.msqrt(self.corr_post)
        P_sqrt = self.std_dev_post @ corr_sqrt
        
        # calculate sigma points for given mean and covariance for the states
        (self.sigmas_raw_fx, self.Wm_x, 
         self.Wc_x, P_sqrt) = self.points_fn_x.compute_sigma_points(self.x_post, 
                                                                    self.P_dummy, 
                                                                    P_sqrt = P_sqrt,
                                                                    **kwargs_sigma_points)

        # propagate all the sigma points through fx
        self.sigmas_prop = self.compute_transformed_sigmas(
            self.sigmas_raw_fx, fx, **fx_args)
        
        self.x_prior, self.corr_prior, self.std_dev_prior = UT(self.sigmas_prop, self.Wm_x, self.Wc_x, Q)
        
        self.x_prior += w_mean #add mean of the noise. 
        
        if self.check_corr_prior_lim:
            self.corr_prior = self.corr_limit(self.corr_prior,
                                              self.corr_prior_lim)
        
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
            self.corr_post = self.corr_prior.copy()
            self.std_dev_post = self.std_dev_prior.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform.normalized_unscented_transformation_additive_noise

        if v_mean is None:
            v_mean = self.v_mean
            
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self._dim_y) * R
        
        #Calculate P_sqrt
        corr_sqrt = self.msqrt(self.corr_prior)
        P_sqrt = self.std_dev_prior @ corr_sqrt

        # recreate sigma points
        (self.sigmas_raw_hx,
         self.Wm_x, self.Wc_x,
         P_sqrt) = self.points_fn_x.compute_sigma_points(self.x_prior, 
                                                         self.P_dummy, 
                                                         P_sqrt = P_sqrt, 
                                                         **kwargs_sigma_points
                                                         )

        # send sigma points through measurement equation
        self.sigmas_meas = self.compute_transformed_sigmas(self.sigmas_raw_hx, 
                                                           hx, **hx_args)

        # pass the propagated sigmas of the states through the unscented transform to compute the predicted measurement, corr_y and std_dev_y
        self.y_pred, self.corr_y, self.std_dev_y = UT(self.sigmas_meas, self.Wm_x, self.Wc_x, R)
        
        self.y_pred += v_mean #add mean of the noise. 
        
        #check if we should add constraints to the correlation term
        if self.check_corr_y_lim:
            self.corr_y = self.corr_limit(self.corr_y, self.corr_y_lim)
        

        # Innovation term of the UKF
        self.y_res = y - self.y_pred
        self.std_dev_y_inv = np.diag([1/sig_y for sig_y in np.diag(self.std_dev_y)])#inverse of diagonal matrix is inverse of each diagonal element - to be multiplied with innovation term
        
        #Obtain the cross_covariance
        sig_x_norm = np.divide(self.sigmas_raw_hx - self.x_prior.reshape(-1,1),
                               np.diag(self.std_dev_prior).reshape(-1,1))
        sig_y_norm = np.divide(self.sigmas_meas - self.y_pred.reshape(-1,1),
                               np.diag(self.std_dev_y).reshape(-1,1))
        self.corr_xy = self.cross_covariance(sig_x_norm, sig_y_norm, self.Wc_x)
        
        #check if we should add constraints to the cross-correlation term
        if self.check_corr_xy_lim:
            self.corr_xy = self.corr_limit(self.corr_xy,
                                              self.corr_xy_lim, cross_corr = True)
        #Kalman gain
        self.K = scipy.linalg.solve(self.corr_y, self.corr_xy.T, assume_a = "pos").T
        assert self.K.shape == (self._dim_x, self._dim_y)

        # calculate posterior
        self.x_post = self.x_prior + self.std_dev_prior @ self.K @ self.std_dev_y_inv @ self.y_res
        
        #obtain posterior correlation and standard deviation
        self.corr_post = self.corr_prior - self.K @ self.corr_xy.T # this is not the true posterior - it is scaled with std_dev_prior
        
        #find the true correlation (values between [-1,1]) and the update factor for the standard deviation
        self.corr_post, std_dev_post_update = self.correlation_from_covariance(self.corr_post)
        
        #Get the prior standard deviation
        self.std_dev_post = np.diag(std_dev_post_update*np.diag(self.std_dev_prior))
        
        #check if we should add constraints to the correlation term
        if self.check_corr_post_lim:
            self.corr_post = self.corr_limit(self.corr_post,
                                              self.corr_post_lim)
        
        
    def corr_limit(self, corr, corr_lim, cross_corr = False):
        #input: correlation matrix. Check wheter elements are above or below limit and if they are, set the correlation to that limit
        
        ##Try to do it fast by using numpy functions
        # idx = np.tril_indices_from(corr, k = -1)
        # idx_lower = (corr[idx] < corr_lim[0])
        # idx_higher = (corr[idx] > corr_lim[1])
        # if idx_lower.any():
        #     idx_l = np.nonzero(idx_lower)[0]
        #     idx_ut = np.triu_indices_from(corr, k = 1)
        #     corr[idx[idx_l]] = corr_lim[0]
                
            
        # if idx_higher.any():
        #     print("high")
        
        #correct way, can be speeded up by using default numpy functions
        dim_x, dim_y = corr.shape
        if not cross_corr:
            #symmetrical matrix - check only lower triangle and change matrix values two places
            assert dim_x == dim_y, "Dimension mismatch for normal correlation matrix"
            for r in range(1,dim_x):
                for c in range(r):
                    if corr[r,c] < corr_lim[0]:
                        corr[r,c] = corr_lim[0]
                        corr[c,r] = corr_lim[0]
                    if corr[r,c] > corr_lim[1]:
                        corr[r,c] = corr_lim[1]
                        corr[c,r] = corr_lim[1]
        else:
            #NOT symmetrical matrix - check whole matrix and change only one limit
            for r in range(dim_x):
                for c in range(dim_y):
                    if corr[r,c] < corr_lim[0]:
                        corr[r,c] = corr_lim[0]
                    if corr[r,c] > corr_lim[1]:
                        corr[r,c] = corr_lim[1]
        # print(corr)
        return corr
        
      
  
