# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:17:56 2022

@author: halvorak
"""

from . import unscented_transform

# from copy import deepcopy
import numpy as np
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

    x_prior : numpy.array(dim_x)
        Prior (predicted) state estimate. 

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. .

    x_post : numpy.array(dim_x)
        Posterior (updated) state estimate. .

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. .

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

    def __init__(self, dim_x, dim_y, hx, fx, points_x, dim_par, sigmas_par, W_par, kappa_Q=None, Q_min=None,
                 sqrt_fn=None, name=None):
        """
        Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        """
        # check inputs
        assert isinstance(dim_x, int)
        assert isinstance(dim_y, int)
        assert isinstance(dim_par, int)
        assert sigmas_par.shape == ((dim_par, dim_par*2+1))
        assert W_par.shape[0] == (dim_par*2+1)

        #pylint: disable=too-many-arguments

        self.x_prior = np.zeros((dim_x, 1))
        self.P_prior = np.eye(dim_x)
        self.x_post = np.copy(self.x_prior)
        self.P_post = np.copy(self.P_prior)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_y)
        self._dim_x = dim_x
        self._dim_y = dim_y
        self.points_fn = points_x
        self._num_sigmas_x = points_x.num_sigma_points()
        self.hx = hx
        self.fx = fx
        self._dim_par = dim_par  # sigma points for the parameters
        self.sigmas_par = sigmas_par  # sigma points for the parameters
        # associated weights for the sigma-points (for the parameters)
        self.W_par = W_par
        self.mean_par = sigmas_par[:, 0]  # 0th sigma-point is the mean
        self._name = name  # object name, handy when printing from within class

        # if sqrt_fn is None:
        #     self.msqrt = scipy.linalg.sqrtm
        # self.msqrt = sqrt_fn

        if kappa_Q is None:
            kappa_Q = 1.
        self.kappa_Q = kappa_Q

        if Q_min is None:
            Q_min = np.eye(self._dim_x)*1e-10
        self.Q_min = Q_min

        # augmented state vector
        self._dim_xa = self._dim_x + self._dim_par
        self._num_sigmas_a = 2*self._dim_xa + 1

        # weights for the augemented set
        self.Wm_a = np.zeros(self._num_sigmas_a)
        self.Wm_a[-2*dim_par:] = W_par[1:]  # W_par[0] is unused
        self.Wc_a = self.Wm_a.copy()

        # create sigma-points
        self.sigmas_raw_fx = np.zeros((self._dim_xa, self._num_sigmas_a))
        """
        Structure of sigmas_raw_fx is (E[x]_m) is E[x]_matrix of appropriate dimensions:
        [[E[x],   sigmas_x1,   sigmas_x2,     E[x]_m,       E[x]_m]
         [E[p],   E[p]_m,      E[p]_m,        sigmas_p1,    sigmas_p2]]
        
        = [[E[x],   sigmas_x,     E[x]_m]
           [E[p],   E[p]_m        sigmas_p]]
        
        It contains 1) the sigma-points for the posterior distribution of x and 2) the sigma-points for the constant parameter distribution
        """
        # create the part for the parameters
        sigmas_par_a = np.tile(
            self.mean_par.reshape(-1, 1), self._num_sigmas_a)
        sigmas_par_a[:, -dim_par*2:] = sigmas_par[:, 1:]

        # insert the sigma-points for the parameters into the augemented sigma-point set. These are constant throughout the simulation
        self.sigmas_raw_fx[self._dim_x:, :] = sigmas_par_a
        # self.sigmas_par_a = sigmas_par_a #not neccessary to save it I guess

        # other saved sigma points
        # propagated through fx and form prior distribution
        self.sigmas_prop = np.zeros((self._dim_x, self._num_sigmas_a))
        self.sigmas_w = np.zeros(
            (self._dim_x, self._num_sigmas_a))  # process noise
        # based on prior distribution
        self.sigmas_raw_hx = np.zeros((self._dim_x, self._num_sigmas_x))
        # propagated through measurement equation. Form posterior distribution
        self.sigmas_meas = np.zeros((self._dim_y, self._num_sigmas_x))

        self.K = np.zeros((dim_x, dim_y))    # Kalman gain
        self.y_res = np.zeros((dim_y, 1))           # residual
        self.y = np.array([[None]*dim_y]).T  # measurement

        self.inv = np.linalg.inv  # not really required

    def predict(self, UT=None, Q_min=None, kappa_Q=None, kwargs_sigma_points={}, fx=None, **fx_args):
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

        kappa_Q : float, optional. If None, self.kappa_Q is used.
            Scaling parameter for the estimated process noise covariance.

        Q_min : np.array((dim_x, dim_x)), optional. If None, self.Q is used.
            Minimum process noise covariance always added to the estimated process noise covariance. It is ADDITIVE to the estimated prior covariance.

        **fx_args : keyword arguments
            optional keyword arguments to be passed into f(x).
        """

        if fx is None:
            fx = self.fx

        if Q_min is None:
            Q_min = self.Q

        if kappa_Q is None:
            kappa_Q = self.kappa_Q

        if UT is None:
            UT = unscented_transform.unscented_transformation_gut

        # calculate sigma points for given mean and covariance for the states
        sigmas_x, self.Wm_x, self.Wc_x, P_sqrt = self.points_fn.compute_sigma_points(
            self.x_post, self.P_post, **kwargs_sigma_points)

        # insert the sigma-points for the states into the augmented sigma-point matrix
        # inserting sigma-points for x
        self.sigmas_raw_fx[:self._dim_x, :self._num_sigmas_x] = sigmas_x
        self.sigmas_raw_fx[:self._dim_x, self._num_sigmas_x:] = np.tile(
            self.x_post.reshape(-1, 1), 2*self._dim_par)  # inserting mean values elsewhere

        # make weights for the augmented sigma-points
        self.Wm_a[1:self._num_sigmas_x] = self.Wm_x[1:]  # insert Wm_x
        self.Wc_a[1:self._num_sigmas_x] = self.Wc_x[1:]
        self.Wm_a[0] = 1 - np.sum(self.Wm_a[1:])  # sum should be 1
        self.Wc_a[0] = 1 - np.sum(self.Wc_a[1:])

        # propagate all the sigma points through fx
        self.sigmas_prop = self.compute_transformed_sigmas(
            self.sigmas_raw_fx, fx, **fx_args)

        # calculate sigmas for the process noise estimation
        self.sigmas_w[:, self._num_sigmas_x:] = self.sigmas_prop[:,
                                                                 self._num_sigmas_x:] - self.sigmas_prop[:, 0].reshape(-1, 1)

        # pass the propagated sigmas of the states (not the augmented states) through the unscented transform to compute prior
        self.x_prior, self.P_prior = UT(
            self.sigmas_prop[:, :self._num_sigmas_x], self.Wm_x, self.Wc_x)

        if True:  # add process noise
            self.w_mean, self.Q_est = UT(self.sigmas_w, self.Wm_a, self.Wc_a)
            self.x_prior += self.w_mean
            self.P_prior += self.kappa_Q*self.Q_est + Q_min  # add process noise

        if True:  # add cross-covariance between the states and estimated process noise

            # P_xa_w = [[P_xw],
            #           [P_pw]]
            # print("sigmas raw:\n",
            #       f"{self.sigmas_raw_fx}\n")
            xa_in, Pa_in = UT(self.sigmas_raw_fx, self.Wm_a, self.Wc_a)
            # print(f"xa_in: {xa_in}\n",
            #       f"Pa_in: {Pa_in}\n",
            #       f"x_post: {self.x_post}\n",
            #       f"P_post: {self.P_post}\n",
            #       f"Pa_in: {Pa_in}\n",
            #       )

            # print("start P_x_w")
            P_xa_w = self.cross_covariance(self.sigmas_raw_fx[:, 0],
                                           self.w_mean,
                                           self.sigmas_raw_fx,
                                           self.sigmas_w,
                                           self.Wc_a)
            self.P_xw = P_xa_w[:self._dim_x, :]
            self.P_xa_w = P_xa_w
            # print("end P_x_w")

            # Calculate Ak=P_xk_xk1.T @ P_post^-1
            self.P_xk_xk1 = self.cross_covariance(self.x_post,
                                                  self.x_prior - self.w_mean,  # check this, if w_mean should be subtracted or not
                                                  sigmas_x,
                                                  self.sigmas_prop[:self._dim_x,
                                                                   :self._num_sigmas_x],
                                                  self.Wc_x)

            # #most readable option, worst numerically
            # P_post_inv = np.linalg.inv(self.P_post)
            # self.Ak = self.P_xk_xk1.T @ P_post_inv

            # best numerical solution
            self.Ak = np.linalg.solve(self.P_post, self.P_xk_xk1).T

            self.P_prior += (self.Ak @ self.P_xw) + (
                self.Ak @ self.P_xw).T

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
        return sigmas_out

    def update(self, y, R=None, UT=None, hx=None, kwargs_sigma_points={}, **hx_args):
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

        # recreate sigma points
        (self.sigmas_raw_hx,
         self.Wm, self.Wc,
         P_sqrt) = self.points_fn.compute_sigma_points(self.x_prior,
                                                       self.P_prior,
                                                       **kwargs_sigma_points
                                                       )

        # send sigma points through measurement equation
        self.sigmas_meas = self.compute_transformed_sigmas(
            self.sigmas_raw_hx, hx, **hx_args)

        # compute mean and covariance of the predicted measurement
        y_pred, Py_pred = UT(self.sigmas_meas, self.Wm, self.Wc)
        Py_pred += R  # add measurement noise

        Pxy = self.cross_covariance(
            self.x_prior, y_pred, self.sigmas_raw_hx, self.sigmas_meas, self.Wc)

        # Kalman gain
        # solve K@Py_pred = P_xy <=> PY_pred.T @ K.T = P_xy.T
        self.K = np.linalg.solve(Py_pred.T, Pxy.T).T
        # self.K = np.linalg.lstsq(Py_pred.T, Pxy.T)[0].T #also an option
        assert self.K.shape == (self._dim_x, self._dim_y)

        # Innovation term of the UKF
        self.y_res = y - y_pred

        # calculate posterior
        self.x_post = self.x_prior + self.K @ self.y_res
        self.P_post = self.P_prior - self.K @ Py_pred @ self.K.T

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
        # dim_x, dim_sigmas_x = sigmas_x.shape
        # dim_y, dim_sigmas_y = sigmas_y.shape
        # assert dim_sigmas_x == dim_sigmas_y, f"dim_sigmas_x != dim_sigmas_y: {dim_sigmas_x} != {dim_sigmas_y}"

        P_xy = np.zeros((dim_x, dim_y))
        # print(f"P_xy: {P_xy}")
        for i in range(dim_sigmas_x):
            P_xy += W_c[i]*((sigmas_x[:, i] - x_mean.flatten()).reshape(-1, 1)
                            @ (sigmas_y[:, i] - y_mean.flatten()).reshape(-1, 1).T)
            # print(P_xy)
            # print(f"i={i}")
        return P_xy


class UnscentedKalmanFilter_EKF_based(UnscentedKalmanFilter):
    
    def __init__(self, dim_x, dim_y, hx, fx, points_x, dim_par, sigmas_par, W_par, kappa_Q=None, Q_min=None,
                 sqrt_fn=None, name=None):
        
        
        super().__init__(dim_x, dim_y, hx, fx, points_x, dim_par, sigmas_par, W_par, kappa_Q=kappa_Q, Q_min=Q_min,
                     sqrt_fn=sqrt_fn, name=name)
        
    def predict(self, UT=None, Q_min=None, kappa_Q=None, kwargs_sigma_points={}, fx=None, **fx_args):
        """ 
        See docstring for parent class
        
        Difference:
        Solves the equation (x_post is the mean of the posterior distribution, x is the full distribution)
        wk = fx(x, p) - fx(x_post, E[p])
        fx(x,p) = fx(x_post, E[p]) + wk
        
        Where fx(x_post, E[p]) is EKF-based prediction step, hence, the name. This means that the cross-covariance between wk and xk is not zero.
        
        """
        
        if fx is None:
            fx = self.fx

        if Q_min is None:
            Q_min = self.Q

        if kappa_Q is None:
            kappa_Q = self.kappa_Q

        if UT is None:
            UT = unscented_transform.unscented_transformation_gut

        # calculate sigma points for given mean and covariance for the states
        sigmas_x, self.Wm_x, self.Wc_x, P_sqrt = self.points_fn.compute_sigma_points(
            self.x_post, self.P_post, **kwargs_sigma_points)

        # insert the sigma-points for the states into the augmented sigma-point matrix
        # inserting sigma-points for x
        self.sigmas_raw_fx[:self._dim_x, 
                           :self._num_sigmas_x] = sigmas_x
        
        self.sigmas_raw_fx[:self._dim_x, 
                           self._num_sigmas_x:] = np.tile(
                               self.x_post.reshape(-1, 1),                                              
                               2*self._dim_par)  # inserting mean values elsewhere

        # make weights for the augmented sigma-points (Wm=Wc for the GenUT)
        #structure is: Wm_a = [Wm_a[0], Wm_x1, Wm_x2, Wm_p1, Wm_p2].
        self.Wm_a[1:self._num_sigmas_x] = self.Wm_x[1:]  # insert Wm_x
        self.Wc_a[1:self._num_sigmas_x] = self.Wc_x[1:]
        self.Wm_a[0] = 1 - np.sum(self.Wm_a[1:])  # sum should be 1
        self.Wc_a[0] = 1 - np.sum(self.Wc_a[1:])

        # propagate all the sigma points through fx
        self.sigmas_prop = self.compute_transformed_sigmas(
            self.sigmas_raw_fx, fx, **fx_args)
        
        #the nominal prediction is f(x_post, E[p])=self.sigmas_prop[:,0]
        self.x_prior = self.sigmas_prop[:, 0].copy()
        
        # pass the propagated sigmas of the states (not the augmented states) through the unscented transform to compute the prior covariance. NB Here we calculate the mean of x_prior internally, but based on the UT!
        _, self.P_prior = UT(
            self.sigmas_prop[:, :self._num_sigmas_x], self.Wm_x, self.Wc_x)

        # calculate sigmas for the process noise estimation. Subtract f(x_post, E[p])=sigmas_w[0] from all the sigma points ==> only sigmas_w[:,0]=0.
        self.sigmas_w[:, 1:] = (self.sigmas_prop[:, 1:] - 
                                self.sigmas_prop[:, 0].reshape(-1, 1)) # subtract sigmas_w[0]
        
        
        if True:  # add process noise
            self.w_mean, self.Q_est = UT(self.sigmas_w, self.Wm_a, self.Wc_a)
            self.x_prior += self.w_mean
            self.P_prior += self.kappa_Q*self.Q_est + Q_min  # add process noise

        if True:  # add cross-covariance between the states and estimated process noise

            # P_xa_w = [[P_xw],
            #           [P_pw]]
            # print("sigmas raw:\n",
            #       f"{self.sigmas_raw_fx}\n")
            xa_in, Pa_in = UT(self.sigmas_raw_fx, self.Wm_a, self.Wc_a)
            # print(f"xa_in: {xa_in}\n",
            #       f"Pa_in: {Pa_in}\n",
            #       f"x_post: {self.x_post}\n",
            #       f"P_post: {self.P_post}\n",
            #       f"Pa_in: {Pa_in}\n",
            #       )

            # print("start P_x_w")
            P_xa_w = self.cross_covariance(self.sigmas_raw_fx[:, 0],
                                           self.w_mean,
                                           self.sigmas_raw_fx,
                                           self.sigmas_w,
                                           self.Wc_a)
            self.P_xw = P_xa_w[:self._dim_x, :]
            self.P_xa_w = P_xa_w
            # print("end P_x_w")

            # Calculate Ak=P_xk_xk1.T @ P_post^-1
            self.P_xk_xk1 = self.cross_covariance(self.x_post,
                                                  self.x_prior - self.w_mean,  # check this, if w_mean should be subtracted or not
                                                  sigmas_x,
                                                  self.sigmas_prop[:self._dim_x,
                                                                   :self._num_sigmas_x],
                                                  self.Wc_x)

            # #most readable option, worst numerically
            # P_post_inv = np.linalg.inv(self.P_post)
            # self.Ak = self.P_xk_xk1.T @ P_post_inv

            # best numerical solution
            self.Ak = np.linalg.solve(self.P_post, self.P_xk_xk1).T
            
            self.Q_xw = self.Ak @ self.P_xw

            self.P_prior += self.Q_xw + self.Q_xw.T
        
        
        