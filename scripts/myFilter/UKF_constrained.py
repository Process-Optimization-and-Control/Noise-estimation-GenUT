# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import (absolute_import, division)

from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import eye, zeros, dot, isscalar, outer
from scipy.linalg import cholesky
# import os
# import sys
# wdir = os.getcwd()
# module_path = os.path.join(wdir, "utils_filter")
# sys.path.append(module_path)

# from utils_filter import unscented_transform
# from utils_filter import stats
# from utils_filter import helpers
from . import unscented_transform
from . import stats
from . import helpers

import check_and_project_points_to_constraints

class UnscentedKalmanFilterConstrained(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=invalid-name
    r"""
    Implements the Scaled Unscented Kalman filter (UKF) as defined by
    Simon Julier in [1], using the formulation provided by Wan and Merle
    in [2]. This filter scales the sigma points to avoid strong nonlinearities.


    Parameters
    ----------

    dim_x : int
        Number of state variables for the filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.


    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.

        This is for convience, so everything is sized correctly on
        creation. If you are using multiple sensors the size of `z` can
        change based on the sensor. Just provide the appropriate hx function


    dt : float
        Time between steps in seconds.



    hx : function(x,**hx_args)
        Measurement function. Converts state vector x into a measurement
        vector of shape (dim_z).

    fx : function(x,dt,**fx_args)
        function that returns the state x transformed by the
        state transition function. dt is the time step in seconds.

    points : class
        Class which computes the sigma points and weights for a UKF
        algorithm. You can vary the UKF implementation by changing this
        class. For example, MerweScaledSigmaPoints implements the alpha,
        beta, kappa parameterization of Van der Merwe, and
        JulierSigmaPoints implements Julier's original kappa
        parameterization. See either of those for the required
        signature of this class if you want to implement your own.

    sqrt_fn : callable(ndarray), default=None (implies scipy.linalg.cholesky)
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix.
        Usually this will not matter to you; if so the default cholesky()
        yields maximal performance. As of van der Merwe's dissertation of
        2004 [6] this was not a well reseached area so I have no advice
        to give you.

        If your method returns a triangular matrix it must be upper
        triangular. Do not use numpy.linalg.cholesky - for historical
        reasons it returns a lower triangular matrix. The SciPy version
        does the right thing as far as this class is concerned.

    x_mean_fn : callable  (sigma_points, weights), optional
        Function that computes the mean of the provided sigma points
        and weights. Use this if your state variable contains nonlinear
        values such as angles which cannot be summed.

        .. code-block:: Python

            def state_mean(sigmas, Wm):
                x = np.zeros(3)
                sum_sin, sum_cos = 0., 0.

                for i in range(len(sigmas)):
                    s = sigmas[i]
                    x[0] += s[0] * Wm[i]
                    x[1] += s[1] * Wm[i]
                    sum_sin += sin(s[2])*Wm[i]
                    sum_cos += cos(s[2])*Wm[i]
                x[2] = atan2(sum_sin, sum_cos)
                return x

    z_mean_fn : callable  (sigma_points, weights), optional
        Same as x_mean_fn, except it is called for sigma points which
        form the measurements after being passed through hx().

    residual_x : callable (x, y), optional
    residual_z : callable (x, y), optional
        Function that computes the residual (difference) between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars. One is for the state variable,
        the other is for the measurement state.

        .. code-block:: Python

            def residual(a, b):
                y = a[0] - b[0]
                if y > np.pi:
                    y -= 2*np.pi
                if y < -np.pi:
                    y += 2*np.pi
                return y

    state_add: callable (x, y), optional, default np.add
        Function that subtracts two state vectors, returning a new
        state vector. Used during update to compute `x + K@y`
        You will have to supply this if your state variable does not
        suport addition, such as it contains angles.

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

    z : ndarray
        Last measurement used in update(). Read only.

    R : numpy.array(dim_z, dim_z)
        measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        process noise matrix

    K : numpy.array
        Kalman gain

    y : numpy.array
        innovation residual

    log_likelihood : scalar
        Log likelihood of last measurement update.

    likelihood : float
        likelihood of last measurment. Read only.

        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.

    mahalanobis : float
        mahalanobis distance of the measurement. Read only.

    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead:

        .. code-block:: Python

            kf.inv = np.linalg.pinv


    Examples
    --------

    Simple example of a linear order 1 kinematic filter in 2D. There is no
    need to use a UKF for this example, but it is easy to read.

    >>> def fx(x, dt):
    >>>     # state transition function - predict next state based
    >>>     # on constant velocity model x = vt + x_0
    >>>     F = np.array([[1, dt, 0, 0],
    >>>                   [0, 1, 0, 0],
    >>>                   [0, 0, 1, dt],
    >>>                   [0, 0, 0, 1]], dtype=float)
    >>>     return np.dot(F, x)
    >>>
    >>> def hx(x):
    >>>    # measurement function - convert state into a measurement
    >>>    # where measurements are [x_pos, y_pos]
    >>>    return np.array([x[0], x[2]])
    >>>
    >>> dt = 0.1
    >>> # create sigma points to use in the filter. This is standard for Gaussian processes
    >>> points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)
    >>>
    >>> kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)
    >>> kf.x = np.array([-1., 1., -1., 1]) # initial state
    >>> kf.P *= 0.2 # initial uncertainty
    >>> z_std = 0.1
    >>> kf.R = np.diag([z_std**2, z_std**2]) # 1 standard
    >>> kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01**2, block_size=2)
    >>>
    >>> zs = [[i+randn()*z_std, i+randn()*z_std] for i in range(50)] # measurements
    >>> for z in zs:
    >>>     kf.predict()
    >>>     kf.update(z)
    >>>     print(kf.x, 'log-likelihood', kf.log_likelihood)

    For in depth explanations see my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

    Also see the filterpy/kalman/tests subdirectory for test code that
    may be illuminating.

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

    def __init__(self, dim_x, dim_z, dt, hx, fx, points,
                 sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
                 residual_x=None,
                 residual_z=None,
                 state_add=None,
                 c = None,
                 d = None):
        """
        Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        """
        #define inequality constraints. They define c^T x >= d
        self.c = c
        self.d = d

        #pylint: disable=too-many-arguments

        self.x = zeros(dim_x)
        self.P = eye(dim_x)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.Q = eye(dim_x)
        self.R = eye(dim_z)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self._dt = dt
        self._num_sigmas = points.num_sigmas()
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        if sqrt_fn is None:
            self.msqrt = cholesky
        else:
            self.msqrt = sqrt_fn

        # weights for the means and covariances.
        self.Wm, self.Wc = points.Wm, points.Wc

        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x

        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z

        if state_add is None:
            self.state_add = np.add
        else:
            self.state_add = state_add

        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update

        self.sigmas_f = zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = zeros((self._num_sigmas, self._dim_z))

        self.K = np.zeros((dim_x, dim_z))    # Kalman gain
        self.y = np.zeros((dim_z))           # residual
        self.z = np.array([[None]*dim_z]).T  # measurement
        self.S = np.zeros((dim_z, dim_z))    # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))   # inverse system uncertainty

        self.inv = np.linalg.inv

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, dt=None, UT=None, fx=None, c = None, d = None, **fx_args):
        r"""
        Performs the predict step of the UKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P). '

        Important: this MUST be called before update() is called for the first
        time.

        Parameters
        ----------

        dt : double, optional
            If specified, the time step to be used for this prediction.
            self._dt is used if this is not provided.

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

        if dt is None:
            dt = self._dt

        if UT is None:
            UT = unscented_transform.unscented_transform
            
        if c is None:
            c = self.c 
        
        if d is None:
            d = self.d
            

        # calculate sigma points for given mean and covariance
        self.compute_process_sigmas(dt, fx, c = c, d = d, **fx_args)

        #and pass sigmas through the unscented transform to compute prior
        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q, #here, sigmas_f = fx(sigma points) - each point has gone through the nonlinear transformation already (constraints checked on the points before and after the transfomration fx)
                            self.x_mean, self.residual_x)

        # update sigma points to reflect the new variance of the points
        self.sigmas_f = self.points_fn.sigma_points(self.x, self.P) #these sigma points have not gone through any transformation - it is just the pure sigma points (not f(sigma points)). They are checked against the constraints in the update function


        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z, R=None, UT=None, hx=None, c = None, d = None, **hx_args):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        z : numpy.array of shape (dim_z)
            measurement vector

        R : numpy.array((dim_z, dim_z)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        hx : callable h(x, **hx_args), optional
            Measurement function. If not provided, the default
            function passed in during construction will be used.

        **hx_args : keyword argument
            arguments to be passed into h(x) after x -> h(x, **hx_args)
        """

        if z is None:
            self.z = np.array([[None]*self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform.unscented_transform #hak mod

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self._dim_z) * R
            
        if c is None:
            c = self.c
        else: #update attribute
            self.c = c
            
        if d is None:
            d = self.d
        else: #update attribute
            self.d = d
        
        # #update these attributes
        # self.c = c
        # self.d = d

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        sigmas_f = self.sigmas_f
        
        #HAK: project these sigma points to constraint surface here
        if not ((c is None) and (d is None)):

            sigmas_f_constrained, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(
                self.sigmas_f, c, d)
            # print(f"sigmas_f: {sigmas_f}\n",
            #       f"sigmas_fc: {sigmas_f_constrained}")
            sigmas_f = sigmas_f_constrained
        
        
        for s in sigmas_f:
            sigmas_h.append(hx(s, **hx_args))

        self.sigmas_h = np.atleast_2d(sigmas_h)
        
        # self.sigmas_h2 = np.atleast_2d(list(map(hx, sigmas_f, **hx_args)))


        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, sigmas_f, self.sigmas_h) #hak update - if I change self.sigmas_f to sigmas_f get "LinAlgError: 2-th leading minor of the array is not positive definite" when doing matrix square root (Cholesky decomposition)


        self.K = dot(Pxz, self.SI)        # Kalman gain
        self.y = self.residual_z(z, zp)   # residual

        # update Gaussian state estimate (x, P)
        self.x = self.state_add(self.x, dot(self.K, self.y))
        self.P = self.P - dot(self.K, dot(self.S, self.K.T))
        
        #HAK: check that the estimate does not violate the constraints
        if not ((c is None) and (d is None)):
            x_unconstrained = self.x
            #set correct dimensions
            x_unconstrained = x_unconstrained.reshape(1, -1)
            x_constrained, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(
                x_unconstrained, c, d)
            self.x = x_constrained.flatten()
            #but does not check annything regarding self.P! Is this correct?

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        """
        Compute cross variance of the state `x` and measurement `z`.
        """

        Pxz = zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = self.residual_x(sigmas_f[i], x)
            dz = self.residual_z(sigmas_h[i], z)
            Pxz += self.Wc[i] * outer(dx, dz)
        return Pxz

    def compute_process_sigmas(self, dt, fx=None, c = None, d = None, **fx_args):
        """
        computes the values of sigmas_f. Normally a user would not call
        this, but it is useful if you need to call update more than once
        between calls to predict (to update for multiple simultaneous
        measurements), so the sigmas correctly reflect the updated state
        x, P.
        """

        if fx is None:
            fx = self.fx
        if c is None:
            c = self.c 
        if d is None:
            d = self.d
        

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)
        #HAK: project these sigma points to constraint surface here
        if not ((c is None) and (d is None)):

            # print("projecting points to c^T x >= d")
            sigmas_constrained, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(sigmas, c, d)
            sigmas = sigmas_constrained
            
            
        # print(sigmas.shape)
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = fx(s, dt, **fx_args)
        #check and project these new points to the constraint surface as well
        if not ((c is None) and (d is None)):
            sigmas_f_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(self.sigmas_f, c, d)
            self.sigmas_f = sigmas_f_c
            

    def batch_filter(self, zs, Rs=None, dts=None, UT=None, saver=None):
        """
        Performs the UKF filter over the list of measurement in `zs`.

        Parameters
        ----------

        zs : list-like
            list of measurements at each time step `self._dt` Missing
            measurements must be represented by 'None'.

        Rs : None, np.array or list-like, default=None
            optional list of values to use for the measurement error
            covariance R.

            If Rs is None then self.R is used for all epochs.

            If it is a list of matrices or a 3D array where
            len(Rs) == len(zs), then it is treated as a list of R values, one
            per epoch. This allows you to have varying R per epoch.

        dts : None, scalar or list-like, default=None
            optional value or list of delta time to be passed into predict.

            If dtss is None then self.dt is used for all epochs.

            If it is a list where len(dts) == len(zs), then it is treated as a
            list of dt values, one per epoch. This allows you to have varying
            epoch durations.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch

        Returns
        -------

        means: ndarray((n,dim_x,1))
            array of the state for each time step after the update. Each entry
            is an np.array. In other words `means[k,:]` is the state at step
            `k`.

        covariance: ndarray((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.

        Examples
        --------

        .. code-block:: Python

            # this example demonstrates tracking a measurement where the time
            # between measurement varies, as stored in dts The output is then smoothed
            # with an RTS smoother.

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = ukf.batch_filter(zs, dts=dts)
            (xs, Ps, Ks) = ukf.rts_smoother(mu, cov)

        """
        #pylint: disable=too-many-arguments

        try:
            z = zs[0]
        except TypeError:
            raise TypeError('zs must be list-like')

        if self._dim_z == 1:
            if not(isscalar(z) or (z.ndim == 1 and len(z) == 1)):
                raise TypeError('zs must be a list of scalars or 1D, 1 element arrays')
        else:
            if len(z) != self._dim_z:
                raise TypeError(
                    'each element in zs must be a 1D array of length {}'.format(self._dim_z))

        z_n = np.size(zs, 0)
        if Rs is None:
            Rs = [self.R] * z_n

        if dts is None:
            dts = [self._dt] * z_n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = zeros((z_n, self._dim_x))
        else:
            means = zeros((z_n, self._dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((z_n, self._dim_x, self._dim_x))

        for i, (z, r, dt) in enumerate(zip(zs, Rs, dts)):
            self.predict(dt=dt, UT=UT)
            self.update(z, r, UT=UT)
            means[i, :] = self.x
            covariances[i, :, :] = self.P

            if saver is not None:
                saver.save()

        return (means, covariances)

    def rts_smoother(self, Xs, Ps, Qs=None, dts=None, UT=None):
        """
        Runs the Rauch-Tung-Striebel Kalman smoother on a set of
        means and covariances computed by the UKF. The usual input
        would come from the output of `batch_filter()`.

        Parameters
        ----------

        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.

        Ps : numpy.array
            array of the covariances of the output of a kalman filter.

        Qs: list-like collection of numpy.array, optional
            Process noise of the Kalman filter at each time step. Optional,
            if not provided the filter's self.Q will be used

        dt : optional, float or array-like of float
            If provided, specifies the time step of each step of the filter.
            If float, then the same time step is used for all steps. If
            an array, then each element k contains the time  at step k.
            Units are seconds.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        Returns
        -------

        x : numpy.ndarray
           smoothed means

        P : numpy.ndarray
           smoothed state covariances

        K : numpy.ndarray
            smoother gain at each step

        Examples
        --------

        .. code-block:: Python

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = kalman.batch_filter(zs)
            (x, P, K) = rts_smoother(mu, cov, fk.F, fk.Q)
        """
        #pylint: disable=too-many-locals, too-many-arguments

        if len(Xs) != len(Ps):
            raise ValueError('Xs and Ps must have the same length')

        n, dim_x = Xs.shape

        if dts is None:
            dts = [self._dt] * n
        elif isscalar(dts):
            dts = [dts] * n

        if Qs is None:
            Qs = [self.Q] * n

        if UT is None:
            UT = unscented_transform.unscented_transform

        # smoother gain
        Ks = zeros((n, dim_x, dim_x))

        num_sigmas = self._num_sigmas

        xs, ps = Xs.copy(), Ps.copy()
        sigmas_f = zeros((num_sigmas, dim_x))

        for k in reversed(range(n-1)):
            # create sigma points from state estimate, pass through state func
            sigmas = self.points_fn.sigma_points(xs[k], ps[k])
            for i in range(num_sigmas):
                sigmas_f[i] = self.fx(sigmas[i], dts[k])

            xb, Pb = UT(
                sigmas_f, self.Wm, self.Wc, self.Q,
                self.x_mean, self.residual_x)

            # compute cross variance
            Pxb = 0
            for i in range(num_sigmas):
                y = self.residual_x(sigmas_f[i], xb)
                z = self.residual_x(sigmas[i], Xs[k])
                Pxb += self.Wc[i] * outer(z, y)

            # compute gain
            K = dot(Pxb, self.inv(Pb))

            # update the smoothed estimates
            xs[k] += dot(K, self.residual_x(xs[k+1], xb))
            ps[k] += dot(K, ps[k+1] - Pb).dot(K.T)
            Ks[k] = K

        return (xs, ps, Ks)

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = stats.logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """"
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(float(dot(dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis

    def __repr__(self):
        return '\n'.join([
            'UnscentedKalmanFilter object',
            helpers.pretty_str('x', self.x),
            helpers.pretty_str('P', self.P),
            helpers.pretty_str('x_prior', self.x_prior),
            helpers.pretty_str('P_prior', self.P_prior),
            helpers.pretty_str('Q', self.Q),
            helpers.pretty_str('R', self.R),
            helpers.pretty_str('S', self.S),
            helpers.pretty_str('K', self.K),
            helpers.pretty_str('y', self.y),
            helpers.pretty_str('log-likelihood', self.log_likelihood),
            helpers.pretty_str('likelihood', self.likelihood),
            helpers.pretty_str('mahalanobis', self.mahalanobis),
            helpers.pretty_str('sigmas_f', self.sigmas_f),
            helpers.pretty_str('h', self.sigmas_h),
            helpers.pretty_str('Wm', self.Wm),
            helpers.pretty_str('Wc', self.Wc),
            helpers.pretty_str('residual_x', self.residual_x),
            helpers.pretty_str('residual_z', self.residual_z),
            helpers.pretty_str('msqrt', self.msqrt),
            helpers.pretty_str('hx', self.hx),
            helpers.pretty_str('fx', self.fx),
            helpers.pretty_str('x_mean', self.x_mean),
            helpers.pretty_str('z_mean', self.z_mean)
            ])

class UKF_reformulated_Kolaas(UnscentedKalmanFilterConstrained):
    
    def __init__(self, dim_x, dim_z, dt, hx, fx, points,
                 sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
                 residual_x=None,
                 residual_z=None,
                 state_add=None,
                 c = None, #for x
                 d = None, #for x
                 c_y = None, #for y
                 d_y = None, #for y
                 #CC = constraint candidate, see Kolaas 2009
                 CC1 = False, #sigma_prior
                 CC2 = False, #propagated sigma, f(sigma_prior)
                 CC3 = False, #mean, prior estimate
                 CC4 = False, #y_i of sigma point
                 CC5 = False, #y_i of sigma point (don't know difference)
                 CC6 = False, #estimated measurement
                 CC7 = False, #sigma_posterior
                 CC8 = False, #posterior estimate
                 recompute_sigmas = False, #recompute sigma points between predict and update
                 CC9 = False # constrain the sigma points based on P_prior
                 ):
        
        super().__init__(dim_x, dim_z, dt, hx, fx, points,
                 sqrt_fn=sqrt_fn, x_mean_fn=x_mean_fn, z_mean_fn=z_mean_fn,
                 residual_x=residual_x,
                 residual_z=residual_z,
                 state_add=state_add,
                 c = c,
                 d = d)
        
        #"new", reformulated sigma points initialization
        self.sigmas_posterior = np.zeros(self.sigmas_f.shape)
        
        #set new parameters to self.    
        self.CC1 = CC1
        self.CC2 = CC2
        self.CC3 = CC3
        self.CC4 = CC4
        self.CC5 = CC5
        self.CC6 = CC6
        self.CC7 = CC7
        self.CC8 = CC8
        self.CC9 = CC9
        self.recompute_sigmas = recompute_sigmas
        self.c_y = c_y
        self.d_y = d_y
        
    def compute_process_sigmas(self, dt, fx=None, c = None, d = None, CC1 = None, CC2 = None, **fx_args):
        """
        computes the values of sigmas_f. Normally a user would not call
        this, but it is useful if you need to call update more than once
        between calls to predict (to update for multiple simultaneous
        measurements), so the sigmas correctly reflect the updated state
        x, P.
        """

        if fx is None:
            fx = self.fx
        if c is None:
            c = self.c 
        if d is None:
            d = self.d
        if CC1 is None:
            CC1 = self.CC1
        if CC2 is None:
            CC2 = self.CC2
        

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)
        if CC1: #project these sigma points to constraint surface
            if not ((c is None) and (d is None)):
                # print("projecting points to c^T x >= d")
                sigmas_constrained, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(sigmas, c, d)
                sigmas = sigmas_constrained
            
            

        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = fx(s, dt, **fx_args)
        # self.sigmas_f2 = np.atleast_2d(list(map(fx, sigmas, dt, **fx_args)))
        
        if CC2: #check and project propagated sigma points to the constraint surface
            if not ((c is None) and (d is None)):
                sigmas_f_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(self.sigmas_f, c, d)
                self.sigmas_f = sigmas_f_c
                
                # sigmas_f2_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(self.sigmas_f2, c, d)
                # self.sigmas_f2 = sigmas_f2_c
    
    
    def predict(self, dt=None, UT=None, fx=None, c = None, d = None, CC1 = None, CC2 = None, CC3 = None, CC9 = None, recompute_sigmas = None, **fx_args):
        r"""
        Performs the predict step of the UKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P). '

        Important: this MUST be called before update() is called for the first
        time.

        Parameters
        ----------

        dt : double, optional
            If specified, the time step to be used for this prediction.
            self._dt is used if this is not provided.

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

        if dt is None:
            dt = self._dt

        if UT is None:
            UT = unscented_transform.unscented_transform
            
        if c is None:
            c = self.c 
        
        if d is None:
            d = self.d
        
        if CC1 is None:
            CC1 = self.CC1
        
        if CC2 is None:
            CC2 = self.CC2
        
        if CC3 is None:
            CC3 = self.CC3
        
        if CC9 is None:
            CC9 = self.CC9
        
        if recompute_sigmas is None:
            recompute_sigmas = self.recompute_sigmas

        # calculate sigma points for given mean and covariance
        self.compute_process_sigmas(dt, fx, c = c, d = d, CC1 = CC1, CC2 = CC2, **fx_args)
        print(f"self.Q: \n{self.Q}")

        #and pass sigmas through the unscented transform to compute prior
        # x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q, #here, sigmas_f = fx(sigma points) - each point has gone through the nonlinear transformation already (constraints checked on the points before and after the transfomration fx)
        #                     self.x_mean, self.residual_x)
        #and pass sigmas through the unscented transform to compute prior
        x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, noise_cov=self.Q, #here, sigmas_f = fx(sigma points) - each point has gone through the nonlinear transformation already (constraints checked on the points before and after the transfomration fx)
                            mean_fn=self.x_mean, residual_fn=self.residual_x)
        
        if CC3: #project mean estimate to sigma points
            
            if not ((c is None) and (d is None)):
                x = np.expand_dims(x, axis = 0) #check and project need 2d array
                x_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(x, c, d)
                x = x_c.flatten()
        self.x = x
        

        
        if recompute_sigmas: #recompute sigma points based on prior estimate
            self.sigmas_f = self.points_fn.sigma_points(self.x, self.P) #these sigma points have not gone through any transformation - it is just the pure sigma points (not f(sigma points)). They are checked against the constraints in the update function
            if CC9:
                if not ((c is None) and (d is None)):
                    sigmas_f_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(self.sigmas_f, c, d)
                    self.sigmas_f = sigmas_f_c
                

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        
    def update(self, z, R=None, UT=None, hx=None, c = None, d = None, c_y = None, d_y = None, CC4 = None, CC5 = None, CC6 = None, CC7 = None, CC8 = None, **hx_args):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        z : numpy.array of shape (dim_z)
            measurement vector

        R : numpy.array((dim_z, dim_z)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        hx : callable h(x, **hx_args), optional
            Measurement function. If not provided, the default
            function passed in during construction will be used.

        **hx_args : keyword argument
            arguments to be passed into h(x) after x -> h(x, **hx_args)
        """

        if z is None:
            self.z = np.array([[None]*self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform.unscented_transform

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self._dim_z) * R
        self.R = R
            
        if c is None:
            c = self.c
        else: #update attribute
            self.c = c
            
        if d is None:
            d = self.d
        else: #update attribute
            self.d = d
            
        if c_y is None:
            c_y = self.c_y
        else: #update attribute
            self.c_y = c_y
            
        if d_y is None:
            d_y = self.d_y
        else: #update attribute
            self.d_y = d_y
            
        if CC4 is None:
            CC4 = self.CC4
        
        if CC5 is None:
            CC5 = self.CC5
        
        if CC6 is None:
            CC6 = self.CC6
        
        if CC7 is None:
            CC7 = self.CC7
        
        if CC8 is None:
            CC8 = self.CC8
        
        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        sigmas_f = self.sigmas_f
        for s in sigmas_f:
            sigmas_h.append(hx(s, **hx_args))

        self.sigmas_h = np.atleast_2d(sigmas_h)
        
        #Can possibly do this instead? Not sure how it works if I actually give **hx_args though (it should be an iterable for map to work, but it is a dict)
        # self.sigmas_h2 = np.atleast_2d(list(map(hx, sigmas_f, **hx_args)))
        
        if (CC4 or CC5):
            if not ((c_y is None) and (d_y is None)):
                sigmas_h_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(self.sigmas_h, c_y, d_y)
                self.sigmas_h = sigmas_h_c

        # mean and covariance of prediction passed through unscented transform
        print(f"self.R: \n{self.R}")
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, self.R, self.z_mean, self.residual_z)
        self.SI = self.inv(self.S)
        if CC6: #project mean estimate to sigma points
            if not ((c_y is None) and (d_y is None)):
                zp = np.expand_dims(zp, axis=0)
                zp_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(zp, c_y, d_y)
                zp = zp_c.flatten()
        

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)


        self.K = dot(Pxz, self.SI)        # Kalman gain
        
        for i in range(self._num_sigmas): #or self.sigmas_f.shape[0]
            self.sigmas_posterior[i,:] = self.state_add(sigmas_f[i, :],
                                                        dot(self.K, self.residual_z(z, self.sigmas_h[i, :])))
            
        if CC7:
            if not ((c is None) and (d is None)):
                sigmas_posterior_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(self.sigmas_posterior, c, d)
                self.sigmas_posterior = sigmas_posterior_c
        print("Supposed to be None")    
        x, self.P = UT(self.sigmas_posterior, self.Wm, self.Wc, noise_cov = None, mean_fn = self.x_mean, residual_fn = self.residual_x)    
        
        if CC8: #project mean estimate to sigma points
            if not ((c is None) and (d is None)):
                x = np.expand_dims(x, axis=0)
                x_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(x, c, d)
                x = x_c.flatten()
        self.x = x
        
        # self.y = self.residual_z(z, zp)   # residual

        # # update Gaussian state estimate (x, P)
        # self.x = self.state_add(self.x, dot(self.K, self.y))
        # self.P = self.P - dot(self.K, dot(self.S, self.K.T))
        
        # #HAK: check that the estimate does not violate the constraints
        # if not ((c is None) and (d is None)):
        #     x_unconstrained = self.x
        #     #set correct dimensions
        #     x_unconstrained = x_unconstrained.reshape(1, -1)
        #     x_constrained, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(
        #         x_unconstrained, c, d)
        #     self.x = x_constrained.flatten()

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

# class UKF_reformulated_Kolaas_QP_update(UnscentedKalmanFilterConstrained):
    
#     def __init__(self, dim_x, dim_z, dt, hx, fx, points,
#                  sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
#                  residual_x=None,
#                  residual_z=None,
#                  state_add=None,
#                  c = None, #for x
#                  d = None, #for x
#                  c_y = None, #for y
#                  d_y = None, #for y
#                  #CC = constraint candidate, see Kolaas 2009
#                  CC1 = False, #sigma_prior
#                  CC2 = False, #propagated sigma, f(sigma_prior)
#                  CC3 = False, #mean, prior estimate
#                  CC4 = False, #y_i of sigma point
#                  CC5 = False, #y_i of sigma point (don't know difference)
#                  CC6 = False, #estimated measurement
#                  CC7 = False, #sigma_posterior
#                  CC8 = False, #posterior estimate
#                  recompute_sigmas = False, #recompute sigma points between predict and update
#                  CC9 = False # constrain the sigma points based on P_prior
#                  ):
        
#         super().__init__(dim_x, dim_z, dt, hx, fx, points,
#                  sqrt_fn=sqrt_fn, x_mean_fn=x_mean_fn, z_mean_fn=z_mean_fn,
#                  residual_x=residual_x,
#                  residual_z=residual_z,
#                  state_add=state_add,
#                  c = c,
#                  d = d)
        
#         #"new", reformulated sigma points initialization
#         self.sigmas_posterior = np.zeros(self.sigmas_f.shape)
        
#         #set new parameters to self.    
#         self.CC1 = CC1
#         self.CC2 = CC2
#         self.CC3 = CC3
#         self.CC4 = CC4
#         self.CC5 = CC5
#         self.CC6 = CC6
#         self.CC7 = CC7
#         self.CC8 = CC8
#         self.CC9 = CC9
#         self.recompute_sigmas = recompute_sigmas
#         self.c_y = c_y
#         self.d_y = d_y
        
#     def compute_process_sigmas(self, dt, fx=None, c = None, d = None, CC1 = None, CC2 = None, **fx_args):
#         """
#         computes the values of sigmas_f. Normally a user would not call
#         this, but it is useful if you need to call update more than once
#         between calls to predict (to update for multiple simultaneous
#         measurements), so the sigmas correctly reflect the updated state
#         x, P.
#         """

#         if fx is None:
#             fx = self.fx
#         if c is None:
#             c = self.c 
#         if d is None:
#             d = self.d
#         if CC1 is None:
#             CC1 = self.CC1
#         if CC2 is None:
#             CC2 = self.CC2
        

#         # calculate sigma points for given mean and covariance
#         sigmas = self.points_fn.sigma_points(self.x, self.P)
#         if CC1: #project these sigma points to constraint surface
#             if not ((c is None) and (d is None)):
#                 # print("projecting points to c^T x >= d")
#                 sigmas_constrained, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(sigmas, c, d)
#                 sigmas = sigmas_constrained
            
            

#         for i, s in enumerate(sigmas):
#             self.sigmas_f[i] = fx(s, dt, **fx_args)
#         # self.sigmas_f2 = np.atleast_2d(list(map(fx, sigmas, dt, **fx_args)))
        
#         if CC2: #check and project propagated sigma points to the constraint surface
#             if not ((c is None) and (d is None)):
#                 sigmas_f_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(self.sigmas_f, c, d)
#                 self.sigmas_f = sigmas_f_c
                
#                 # sigmas_f2_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(self.sigmas_f2, c, d)
#                 # self.sigmas_f2 = sigmas_f2_c
    
    
#     def predict(self, dt=None, UT=None, fx=None, c = None, d = None, CC1 = None, CC2 = None, CC3 = None, CC9 = None, recompute_sigmas = None, **fx_args):
#         r"""
#         Performs the predict step of the UKF. On return, self.x and
#         self.P contain the predicted state (x) and covariance (P). '

#         Important: this MUST be called before update() is called for the first
#         time.

#         Parameters
#         ----------

#         dt : double, optional
#             If specified, the time step to be used for this prediction.
#             self._dt is used if this is not provided.

#         fx : callable f(x, dt, **fx_args), optional
#             State transition function. If not provided, the default
#             function passed in during construction will be used.

#         UT : function(sigmas, Wm, Wc, noise_cov), optional
#             Optional function to compute the unscented transform for the sigma
#             points passed through hx. Typically the default function will
#             work - you can use x_mean_fn and z_mean_fn to alter the behavior
#             of the unscented transform.

#         **fx_args : keyword arguments
#             optional keyword arguments to be passed into f(x).
#         """

#         if dt is None:
#             dt = self._dt

#         if UT is None:
#             UT = unscented_transform.unscented_transform
            
#         if c is None:
#             c = self.c 
        
#         if d is None:
#             d = self.d
        
#         if CC1 is None:
#             CC1 = self.CC1
        
#         if CC2 is None:
#             CC2 = self.CC2
        
#         if CC3 is None:
#             CC3 = self.CC3
        
#         if CC9 is None:
#             CC9 = self.CC9
        
#         if recompute_sigmas is None:
#             recompute_sigmas = self.recompute_sigmas

#         # calculate sigma points for given mean and covariance
#         self.compute_process_sigmas(dt, fx, c = c, d = d, CC1 = CC1, CC2 = CC2, **fx_args)

#         #and pass sigmas through the unscented transform to compute prior
#         x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q, #here, sigmas_f = fx(sigma points) - each point has gone through the nonlinear transformation already (constraints checked on the points before and after the transfomration fx)
#                             self.x_mean, self.residual_x)
        
#         if CC3: #project mean estimate to sigma points
            
#             if not ((c is None) and (d is None)):
#                 x = np.expand_dims(x, axis = 0) #check and project need 2d array
#                 x_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(x, c, d)
#                 x = x_c.flatten()
#         self.x = x
        

        
#         if recompute_sigmas: #recompute sigma points based on prior estimate
#             self.sigmas_f = self.points_fn.sigma_points(self.x, self.P) #these sigma points have not gone through any transformation - it is just the pure sigma points (not f(sigma points)). They are checked against the constraints in the update function
#             if CC9:
#                 if not ((c is None) and (d is None)):
#                     sigmas_f_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(self.sigmas_f, c, d)
#                     self.sigmas_f = sigmas_f_c
                

#         # save prior
#         self.x_prior = np.copy(self.x)
#         self.P_prior = np.copy(self.P)
        
#     def update(self, z, R=None, UT=None, hx=None, c = None, d = None, c_y = None, d_y = None, CC4 = None, CC5 = None, CC6 = None, CC7 = None, CC8 = None, **hx_args):
#         """
#         Update the UKF with the given measurements. On return,
#         self.x and self.P contain the new mean and covariance of the filter.

#         Parameters
#         ----------

#         z : numpy.array of shape (dim_z)
#             measurement vector

#         R : numpy.array((dim_z, dim_z)), optional
#             Measurement noise. If provided, overrides self.R for
#             this function call.

#         UT : function(sigmas, Wm, Wc, noise_cov), optional
#             Optional function to compute the unscented transform for the sigma
#             points passed through hx. Typically the default function will
#             work - you can use x_mean_fn and z_mean_fn to alter the behavior
#             of the unscented transform.

#         hx : callable h(x, **hx_args), optional
#             Measurement function. If not provided, the default
#             function passed in during construction will be used.

#         **hx_args : keyword argument
#             arguments to be passed into h(x) after x -> h(x, **hx_args)
#         """

#         if z is None:
#             self.z = np.array([[None]*self._dim_z]).T
#             self.x_post = self.x.copy()
#             self.P_post = self.P.copy()
#             return

#         if hx is None:
#             hx = self.hx

#         if UT is None:
#             UT = unscented_transform.unscented_transform

#         if R is None:
#             R = self.R
#         elif isscalar(R):
#             R = eye(self._dim_z) * R
            
#         if c is None:
#             c = self.c
#         else: #update attribute
#             self.c = c
            
#         if d is None:
#             d = self.d
#         else: #update attribute
#             self.d = d
            
#         if c_y is None:
#             c_y = self.c_y
#         else: #update attribute
#             self.c_y = c_y
            
#         if d_y is None:
#             d_y = self.d_y
#         else: #update attribute
#             self.d_y = d_y
            
#         if CC4 is None:
#             CC4 = self.CC4
        
#         if CC5 is None:
#             CC5 = self.CC5
        
#         if CC6 is None:
#             CC6 = self.CC6
        
#         if CC7 is None:
#             CC7 = self.CC7
        
#         if CC8 is None:
#             CC8 = self.CC8
        
#         # pass prior sigmas through h(x) to get measurement sigmas
#         # the shape of sigmas_h will vary if the shape of z varies, so
#         # recreate each time
#         sigmas_h = []
#         sigmas_f = self.sigmas_f
#         for s in sigmas_f:
#             sigmas_h.append(hx(s, **hx_args))

#         self.sigmas_h = np.atleast_2d(sigmas_h)
        
#         #Can possibly do this instead? Not sure how it works if I actually give **hx_args though (it should be an iterable for map to work, but it is a dict)
#         # self.sigmas_h2 = np.atleast_2d(list(map(hx, sigmas_f, **hx_args)))
        
#         if (CC4 or CC5):
#             if not ((c_y is None) and (d_y is None)):
#                 sigmas_h_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(self.sigmas_h, c_y, d_y)
#                 self.sigmas_h = sigmas_h_c

#         # mean and covariance of prediction passed through unscented transform
#         zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)
#         self.SI = self.inv(self.S)
#         if CC6: #project mean estimate to sigma points
#             if not ((c_y is None) and (d_y is None)):
#                 zp = np.expand_dims(zp, axis=0)
#                 zp_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(zp, c_y, d_y)
#                 zp = zp_c.flatten()
        

#         # compute cross variance of the state and the measurements
#         Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)


#         self.K = dot(Pxz, self.SI)        # Kalman gain
        
#         for i in range(self._num_sigmas): #or self.sigmas_f.shape[0]
#             self.sigmas_posterior[i,:] = self.state_add(sigmas_f[i, :],
#                                                         dot(self.K, self.residual_z(z, self.sigmas_h[i, :])))
            
#         if CC7:
#             if not ((c is None) and (d is None)):
#                 sigmas_posterior_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(self.sigmas_posterior, c, d)
#                 self.sigmas_posterior = sigmas_posterior_c
            
#         x, self.P = UT(self.sigmas_posterior, self.Wm, self.Wc, noise_cov = None, mean_fn = self.x_mean, residual_fn = self.residual_x)    
        
#         if CC8: #project mean estimate to sigma points
#             if not ((c is None) and (d is None)):
#                 x = np.expand_dims(x, axis=0)
#                 x_c, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(x, c, d)
#                 x = x_c.flatten()
#         self.x = x
        
#         # self.y = self.residual_z(z, zp)   # residual

#         # # update Gaussian state estimate (x, P)
#         # self.x = self.state_add(self.x, dot(self.K, self.y))
#         # self.P = self.P - dot(self.K, dot(self.S, self.K.T))
        
#         # #HAK: check that the estimate does not violate the constraints
#         # if not ((c is None) and (d is None)):
#         #     x_unconstrained = self.x
#         #     #set correct dimensions
#         #     x_unconstrained = x_unconstrained.reshape(1, -1)
#         #     x_constrained, bool_unchanged = check_and_project_points_to_constraints.check_and_project_points_to_constraints(
#         #         x_unconstrained, c, d)
#         #     self.x = x_constrained.flatten()

#         # save measurement and posterior state
#         self.z = deepcopy(z)
#         self.x_post = self.x.copy()
#         self.P_post = self.P.copy()

#         # set to None to force recompute
#         self._log_likelihood = None
#         self._likelihood = None
#         self._mahalanobis = None
