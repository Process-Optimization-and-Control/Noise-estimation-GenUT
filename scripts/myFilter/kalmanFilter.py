# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:17:56 2022

@author: halvorak
"""

# from . import unscented_transform

# from copy import deepcopy
import numpy as np
import scipy.linalg


class KalmanFilter(object):
    r"""
    Implements the Kalman filter (linear)


    Parameters
    ----------

    dim_x : int
        Number of state variables for the filter.


    dim_y : int
        Number of of measurements




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


    """

    def __init__(self, F, G, H, Q, R,
                 name=None):
        """
        Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        """
        # check inputs
        dim_x = F.shape[0]
        dim_y = H.shape[0]
        # dim_w = L.shape[1]
        
        assert dim_x == F.shape[1]
        assert dim_x == H.shape[1] #H.shape== ((dim_y, dim_x))
        assert R.shape == ((dim_y, dim_y))


        self.x_prior = np.zeros((dim_x, 1))
        self.P_prior = np.eye(dim_x)
        self.x_post = np.copy(self.x_prior)
        self.P_post = np.copy(self.P_prior)
        self.Q = Q
        self.R = R
        self._dim_x = dim_x
        self._dim_y = dim_y
        # self._dim_w = dim_w
        
        #matrices for prediction: x_prior = F@x_post + G@uk + L@w
        self.F = F
        self.G = G
        # self.L = L
        #matrix for posterior: x_post = H@x_prior + v
        self.H = H 
        
        self._name = name  # object name, handy when printing from within class

        # # augmented state vector
        # self._dim_xa = self._dim_x + self._dim_w
        # self.xa_prior = np.vstack((self._x_post, np.zeros((self._dim_w))))
        # self.P_xw = np.zeros((self._dim_x, self._dim_w))
        # self.Pa_prior = np.
        # self.Pa_prior = scipy.linalg.block_diag([self.P_post, self.Q])
        
        self.K = np.zeros((dim_x, dim_y))    # Kalman gain
        self.y_res = np.zeros((dim_y, 1))           # residual
        self.y = np.array([[None]*dim_y]).T  # measurement


    def predict(self, u, F=None, G=None, Q = None):
        r"""
        Performs the predict step of the Kalman filter. On return, self.x_prior and
        self.P_prior contain the predicted state (x) and covariance (P). '

        Important: this MUST be called before update() is called for the first
        time.
        
        Solves the equation for mean and covariance:
        [[x_prior = F@x_post + G@uk + L@w],
         [w = w]]
         

        Parameters
        ----------

        u: control input
        """

        if F is None:
            F = self.F

        if G is None:
            G = self.G

        if Q is None:
            Q = self.Q
       

        x_prior = F @ self.x_post + G @ u
        P_prior = F @ self.P_post @ F.T #+ L @ Q @ L.T

        self.x_prior = x_prior
        self.P_prior = P_prior
        

    def update(self, y, H=None, R=None):
        """
        Update the UKF with the given measurements. On return,
        self.x_post and self.P_post contain the new mean and covariance of the filter.

        Parameters
        ----------

        y : numpy.array of shape (dim_y)
            measurement vector

        R : numpy.array((dim_y, dim_y)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.

        
        """

        if y is None:
            self.y = np.array([[None]*self._dim_y]).T
            self.x_post = self.x_prior.copy()
            self.P_post = self.P_prior.copy()
            return

        if H is None:
            H = self.H

      
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self._dim_y) * R
        
        # Kalman gain (K) by solving K@A = b ==> A.T @ K.T = b.T where A and b are dummy variables
        A = H @ self.P_prior @ H.T + R
        b = self.P_prior @ H.T
        # print(f"A: {A.shape}\n",
        #       f"b: {b.shape}")
        K = np.linalg.solve(A.T, b.T).T
        # K = b @ np.linalg.inv(A) #alternative (but slower)
        
        y_pred = H @ self.x_prior
        y_res = y - y_pred
        x_post = self.x_prior + K @ y_res
        
        #Joseph stabilized form of P_post
        a_term = np.eye(self._dim_x) - K @ H
        P_post = a_term @ self.P_prior @ a_term.T + K @ R @ K.T
        
        self.x_post = x_post
        self.P_post = P_post
        self.K = K

    
        
        