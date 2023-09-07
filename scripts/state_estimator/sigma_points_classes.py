# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:54:11 2021

@author: halvorak

NB: Documentation of the classes/functions may be outdated
"""

import numpy as np
import scipy.stats
import scipy.linalg

class SigmaPoints():
    """
    Parent class when sigma points algorithms are constructed. All points tru to estimate mean and covariance of Y, where Y=f(X)
    """
    def __init__(self, n, sqrt_method = scipy.linalg.sqrtm):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is scipy.linalg.sqrtm (principal matrix square root). Method to calculate the square root of a matrix. The other choice is typically np.linalg.cholesky or sqrt_method = lambda P: scipy.linalg.cholesky(P, lower = True)

        Returns
        -------
        None.

        """
        self.n = n
        
        #principal matrix square root or Cholesky factorization (only lower triangular factorization) is supported. Default for np.linalg.cholesky is lower factorization, default for scipy.linalg.cholesky is upper factorization
        if sqrt_method is scipy.linalg.cholesky:
            sqrt_method = lambda P: scipy.linalg.cholesky(P, lower = True)
        self.sqrt = sqrt_method
        
    def num_sigma_points(self):
        """
        Returns the number of sigma points. Most algorithms return (2n+1) points, can be overwritten by child class

        Returns
        -------
        TYPE int
            DESCRIPTION. dim_sigma, number of sigma points

        """
        return 2*self.n + 1
    
    def is_matrix_pos_def(self, a_matrix):
        """
        Checks if a matrix is positive definite by checking if all eigenvalues are positive

        Parameters
        ----------
        a_matrix : TYPE np.array((n,n))
            DESCRIPTION. A matrix

        Returns
        -------
        TYPE bool
            DESCRIPTION. True if the matrix is pos def, else False

        """
        return np.all(np.linalg.eigvals(a_matrix) > 0)
    

    
class JulierSigmaPoints(SigmaPoints):
    """
    Implement the sigma points as described by Julier's original paper. It assumes that the distribtions are symmetrical.
    
    @TECHREPORT{Julier96ageneral,
    author = {Simon Julier and Jeffrey K. Uhlmann},
    title = {A General Method for Approximating Nonlinear Transformations of Probability Distributions},
    institution = {},
    year = {1996}
}
    
    """
    def __init__(self, n, kappa = 0., sqrt_method = scipy.linalg.sqrtm):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is scipy.linalg.sqrtm (principal matrix square root). Method to calculate the square root of a matrix. The other choice is typically np.linalg.cholesky
        kappa : TYPE, optional float
            DESCRIPTION. The default is 0. If set to (n-3), you minimize error in higher order terms.


        Returns
        -------
        None.

        """
        super().__init__(n, sqrt_method = sqrt_method)
        # print("WARNING: This class has NOT been verified yet")
        # raise ValueError("This class has NOT been verified yet!")
        if not (kappa == np.max([(3-n), 0])):
            print(f"warning: kappa is not set to kappa = max([(3-n),0]) = max([{3-n},0]), which minimizes the fourth order mismatch. Proceeding with a value of kappa = {kappa}")
        self.kappa = kappa
        self.dim_sigma = self.num_sigma_points()
        self.Wm = self.compute_weights()
        self.Wc = self.Wm.copy()
        
    # def compute_weights(self)
    def compute_sigma_points(self, mu, P, P_sqrt = None):
        """
        Computes the sigma points based on Julier's paper

        Parameters
        ----------
        mu : TYPE np.array(n,)
            DESCRIPTION. Mean value of X 
        P : TYPE np.array(n,n)
            DESCRIPTION. Covariance matrix of X
        P_sqrt : TYPE np.array(n,n), optional
            DESCRIPTION. default is None. If supplied, algorithm does not compute sqrt(P).

        Raises
        ------
        ValueError
            DESCRIPTION. Shapes are wrong
        LinAlgError
            DESCRIPTION. P is not positiv definite and symmetric

        Returns
        -------
        sigmas : TYPE np.array(n, dim_sigma)
            DESCRIPTION. sigma points
        P_sqrt : TYPE np.array(n,n)
            DESCRIPTION. sqrt((n+kappa)P). Can be inspected if something goes wrong.

        """
        if not self.n == mu.shape[0]:
            raise ValueError(f" self.n = {self.n} while mu.shape = {mu.shape}. mu.shape[0] must match self.n!")
        
        if not ((self.n == P.shape[0]) and (self.n == P.shape[1])):
            raise ValueError(f"P.shape = {P.shape}, it must be ({self.n, self.n})")
        
        
        n = self.n
        dim_sigma = self.dim_sigma
        
        sigmas = np.zeros((n, dim_sigma))
        sigmas[:, 0] = mu
        
        try:
            sqrt_factor = np.sqrt(n+self.kappa)
            if P_sqrt is None:
                P_sqrt = self.sqrt(P)
            P_sqrt_weight = sqrt_factor*P_sqrt
        except np.linalg.LinAlgError as LinAlgError:
            print(f"(n+kappa)P is not positive definite. Current value is (n+kappa)P = {(n+self.kappa)*P}")
            raise LinAlgError
        
        for i in range(n):
            sigmas[:, 1 + i] = mu + P_sqrt_weight[:, i]
            sigmas[:, 1 + n + i] = mu - P_sqrt_weight[:, i]
        
        return sigmas, self.Wm, self.Wc, P_sqrt
        
    def compute_weights(self):
        """
        Computes the weights

        Returns
        -------
        weights : TYPE np.array(dim_sigma,)
            DESCRIPTION. Weights for every sigma points

        """
        n = self.n
        dim_sigma = self.dim_sigma
        
        weights = np.array([1/(2*(n + self.kappa)) for i in range(dim_sigma)])
        weights[0] = self.kappa/(n + self.kappa)
        return weights

class ScaledSigmaPoints(SigmaPoints):
    """
    From
    JULIER, S. J. The Scaled Unscented Transformation.  Proceedings of the American Control Conference, 2002 2002 Anchorage. 4555-4559 vol.6.

    """
    
    def __init__(self, n, alpha = 1e-3, beta = 2., kappa = 0., sqrt_method = scipy.linalg.sqrtm, suppress_init_warning = False, force_Wm_sum_zero = False):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is scipy.linalg.sqrtm (principal matrix square root). Method to calculate the square root of a matrix. The other choice is typically np.linalg.cholesky
        alpha : TYPE, optional float
            DESCRIPTION. The default is 1e-3. Determines the scaling of the sigma-points (how far away they are from the mean.
        kappa : TYPE, optional float
            DESCRIPTION. The default is 0. If set to (n-3), you minimize error in higher order terms.


        Returns
        -------
        None.

        """
        super().__init__(n, sqrt_method = sqrt_method)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.force_Wm_sum_zero = force_Wm_sum_zero
        self.lam = self.calculate_lam()
        if ((kappa != np.max([(3-n), 0])) and (suppress_init_warning == False)):
            print(f"warning: kappa is not set to kappa = max([(3-n),0]) = max([{3-n},0]), which minimizes the fourth order mismatch. Proceeding with a value of kappa = {kappa}")
        self.dim_sigma = self.num_sigma_points()
        self.Wm, self.Wc = self.compute_weights(force_Wm_sum_zero = force_Wm_sum_zero)
    
    def calculate_lam(self):
        lam = (self.alpha**2)*(self.n + self.kappa) - self.n
        return lam
        
    def compute_weights(self, force_Wm_sum_zero = False):
        """
        Computes the weights

        Returns
        -------
        weights : TYPE np.array(dim_sigma,)
            DESCRIPTION. Weights for every sigma points

        """
        n = self.n
        dim_sigma = self.dim_sigma
        alpha = self.alpha
        beta = self.beta
        
        lam = self.calculate_lam()
        
        Wm = np.array([1/(2*(n + lam)) for i in range(dim_sigma)])
        Wc = Wm.copy()
        
        if force_Wm_sum_zero:
            Wm[0] = 1 - Wm[1:].sum()
        else:
            Wm[0] = lam/(lam + n)
        Wc[0] = lam/(lam + n) + (1 - alpha**2 + beta) #this does not add up to 1 (Wm.sum()=1, Wm.sum() != 1)
        return Wm, Wc
    
    def compute_sigma_points(self, mu, P, P_sqrt = None):
        """
        Computes the sigma points based on Julier's paper

        Parameters
        ----------
        mu : TYPE np.array(n,)
            DESCRIPTION. Mean value of X 
        P : TYPE np.array(n,n)
            DESCRIPTION. Covariance matrix of X
        P_sqrt : TYPE np.array(n,n), optional
            DESCRIPTION. default is None. If supplied, algorithm does not compute sqrt(P).

        Raises
        ------
        ValueError
            DESCRIPTION. Shapes are wrong
        LinAlgError
            DESCRIPTION. P is not positiv definite and symmetric

        Returns
        -------
        sigmas : TYPE np.array(n, dim_sigma)
            DESCRIPTION. sigma points
        P_sqrt : TYPE np.array(n,n)
            DESCRIPTION. sqrt((n+kappa)P). Can be inspected if something goes wrong.

        """
        if not self.n == mu.shape[0]:
            raise ValueError(f" self.n = {self.n} while mu.shape = {mu.shape}. mu.shape[0] must match self.n!")
        
        if not ((self.n == P.shape[0]) and (self.n == P.shape[1])):
            raise ValueError(f"P.shape = {P.shape}, it must be ({self.n, self.n})")
        
        
        n = self.n
        dim_sigma = self.dim_sigma
        
        sigmas = np.zeros((n, dim_sigma))
        sigmas[:, 0] = mu
        
        try:
            sqrt_factor = np.sqrt(n+self.lam)
            if P_sqrt is None:
                P_sqrt = self.sqrt(P)
            P_sqrt_weight = sqrt_factor*P_sqrt
        except np.linalg.LinAlgError as LinAlgError:
            print(f"(n+kappa)P is not positive definite. Current value is (n+kappa)P = {(n+self.kappa)*P}")
            raise LinAlgError
        
        for i in range(n):
            sigmas[:, 1 + i] = mu + P_sqrt_weight[:, i]
            sigmas[:, 1 + n + i] = mu - P_sqrt_weight[:, i]
        
        return sigmas, self.Wm, self.Wc, P_sqrt
        
  
    
class GenUTSigmaPoints(SigmaPoints):
    """
    Implement the sigma points as described by Ebeigbe. Distributions does NOT need to be symmetrical.
    
    @article{EbeigbeDonald2021AGUT,
abstract = {The unscented transform uses a weighted set of samples called sigma points to propagate the means and covariances of nonlinear transformations of random variables. However, unscented transforms developed using either the Gaussian assumption or a minimum set of sigma points typically fall short when the random variable is not Gaussian distributed and the nonlinearities are substantial. In this paper, we develop the generalized unscented transform (GenUT), which uses adaptable sigma points that can be positively constrained, and accurately approximates the mean, covariance, and skewness of an independent random vector of most probability distributions, while being able to partially approximate the kurtosis. For correlated random vectors, the GenUT can accurately approximate the mean and covariance. In addition to its superior accuracy in propagating means and covariances, the GenUT uses the same order of calculations as most unscented transforms that guarantee third-order accuracy, which makes it applicable to a wide variety of applications, including the assimilation of observations in the modeling of the coronavirus (SARS-CoV-2) causing COVID-19.},
journal = {ArXiv},
year = {2021},
title = {A Generalized Unscented Transformation for Probability Distributions},
language = {eng},
address = {United States},
author = {Ebeigbe, Donald and Berry, Tyrus and Norton, Michael M and Whalen, Andrew J and Simon, Dan and Sauer, Timothy and Schiff, Steven J},
issn = {2331-8422},
}


    """
    def __init__(self, n, theta = 1 - 1e-8, sqrt_method = scipy.linalg.sqrtm, lbx = None, ubx = None):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is scipy.linalg.sqrtm (principal matrix square root). Method to calculate the square root of a matrix. The other choice is typically np.linalg.cholesky

        Returns
        -------
        None.

        """
        super().__init__(n, sqrt_method = sqrt_method)
        self.dim_sigma = self.num_sigma_points()
        self.theta = theta
        self.lbx = lbx #constraints for the sigma-points
        self.ubx = ubx
    
    # @staticmethod
    def compute_std_moments(self, P, S, K):
        std_dev = np.sqrt(np.diag(P))
        std_dev_inv = np.diag([1/si for si in std_dev])
        corr = std_dev_inv @ P @ std_dev_inv
        corr_sqrt = self.sqrt(corr)
        
        corr_sqrt_pow_3_inv = scipy.linalg.inv(np.power(corr_sqrt, 3))
        corr_sqrt_pow_4_inv = scipy.linalg.inv(np.power(corr_sqrt, 4))
        P_sqrt_pow3_inv = corr_sqrt_pow_3_inv @ np.power(std_dev_inv, 3)
        P_sqrt_pow4_inv = corr_sqrt_pow_4_inv @ np.power(std_dev_inv, 4)
        
        S_std = P_sqrt_pow3_inv @ S
        K_std = P_sqrt_pow4_inv @ K
        return S_std, K_std
        
    
    def compute_scaling_and_weights(self, P, S, K, u = None):
        """
        Computes the scaling parameters s and the weights w

        Parameters
        ----------
        P : TYPE np.array(n,n)
            DESCRIPTION. Covariance of X
        S : TYPE np.array(n,)
            DESCRIPTION. 3rd central moment of X. Can be computed by scipy.stats.moments(data, moment=3)
        K : TYPE np.array(n,)
            DESCRIPTION. 4th central moment of X. Can be computed by scipy.stats.moments(data, moment=4)
        u : TYPE, optional np.array(n,)
            DESCRIPTION. The default is None. First part of scaling arrays. s1> 0 for every element. If None, algorithm computes the suggested values in the article.

        Raises
        ------
        ValueError
            DESCRIPTION. Dimension mismatch

        Returns
        -------
        s : TYPE np.array(2n,)
            DESCRIPTION. Scaling values
        w : TYPE np.array(dim_sigma,)
            DESCRIPTION. Weights for every sigma points.

        """
        
        S_std, K_std = self.compute_std_moments(P, S, K)
        
        P_sqrt_pow3_inv = scipy.linalg.inv(np.power(self.P_sqrt, 3))
        P_sqrt_pow4_inv = scipy.linalg.inv(np.power(self.P_sqrt, 4))
        
        S_std = P_sqrt_pow3_inv @ S
        

        K_std = P_sqrt_pow4_inv @ K
        
        self.S_std = S_std
        self.K_std = K_std
        
        self.S_std2, self.K_std2 = self.compute_std_moments(P, S, K)
        
        if u is None: #create s (s.shape = (n,))
            # u = self.select_u_to_match_kurtosis(S_std, K_std)
            u = self.select_u(P, S, K)
        
        assert u.shape[0] == S.shape[0], "Dimension of u is wrong"
        
        #create the next values for s, total dim is 2n+1
        v = u + S_std
        
        w2 = 1 / v / (u + v)
        w1 = w2*v/u
        w = np.concatenate((np.array([0]), w1, w2))
        w[0] = 1 - np.sum(w[1:])
        return u, v, w
    
    def select_u_to_match_kurtosis(self, S_std, K_std):
        """
        Computes the first part of the scaling array by the method suggested in the paper.

        Parameters
        ----------
        S_std : TYPE np.array(n,)
            DESCRIPTION. Scaled 3rd central moment
        K_std : TYPE np.array(n,)
            DESCRIPTION. Scaled 4th central moment

        Returns
        -------
        s1 : TYPE np.array(n,)
            DESCRIPTION. First part of the scaling value array

        """
        val = 4*K_std - 3*np.square(S_std)
        u = .5*(-S_std + np.sqrt(val, where = val >= .0, out = np.zeros(val.shape[0])))
        u = np.maximum(u, np.abs(S_std) + 1e-1) #ensures that v>0
        # u = np.maximum(u, 1e-1*np.ones(u.shape[0]))
        assert (u > 0).all(), f"u should be > 0, now u={u}"
        return u
    
    def select_u(self, P, S, K):
        """
        Computes the first part of the scaling array by the method suggested in the paper.

        Parameters
        ----------
        S_std : TYPE np.array(n,)
            DESCRIPTION. Scaled 3rd central moment
        K_std : TYPE np.array(n,)
            DESCRIPTION. Scaled 4th central moment

        Returns
        -------
        s1 : TYPE np.array(n,)
            DESCRIPTION. First part of the scaling value array

        """
        sigma = np.sqrt(np.diag(P)) #standard deviation of each state
        S_std_univar = np.divide(S, np.power(sigma, 3))
        K_std_univar = np.divide(K, np.power(sigma, 4))
        
        P_sqrt_pow3_inv = scipy.linalg.inv(np.power(self.P_sqrt, 3))
        P_sqrt_pow4_inv = scipy.linalg.inv(np.power(self.P_sqrt, 4))
        
        S_std = P_sqrt_pow3_inv @ S
       
        K_std = P_sqrt_pow4_inv @ K
        
        self.S_std = S_std
        self.K_std = K_std
        self.S_std_univar = S_std_univar
        self.K_std_univar = K_std_univar
        
        
        val = 4*K_std - 3*np.square(S_std)
        u = .5*(-S_std + np.sqrt(val, where = val >= .0, out = np.zeros(val.shape[0])))
        u = np.maximum(u, np.abs(S_std) + .5) #ensures that v>0
        # u = np.maximum(u, 1e-1*np.ones(u.shape[0]))
        assert (u > 0).all(), f"u should be > 0, now u={u}"
        return u
    
    def compute_sigma_points(self, mu, P, S = None, K = None, u = None, sqrt_method = None, P_sqrt = None, lbx = None, ubx = None, theta = None):
        """
        Computes the sigma points

        Parameters
        ----------
        mu : TYPE np.array(n,)
            DESCRIPTION. Mean value of X 
        P : TYPE np.array(n,n)
            DESCRIPTION. Covariance matrix of X
        S : TYPE np.array(n,) if None, assumed symmetric distribution (0)
            DESCRIPTION. 3rd central moment of X. Can be computed by scipy.stats.moments(data, moment=3)
        K : TYPE np.array(n,) If none, assumed Gaussian distribution
            DESCRIPTION. 4th central moment of X. Can be computed by scipy.stats.moments(data, moment=4)
        u : TYPE, optional np.array(n,)
            DESCRIPTION. The default is None. First part of scaling arrays. u> 0 for every element. If None, algorithm computes the suggested values in the article.
        P_sqrt : TYPE np.array(n,n), optional
            DESCRIPTION. default is None. If supplied, algorithm does not compute sqrt(P).

        Raises
        ------
        ValueError
            DESCRIPTION. Shapes are wrong
        LinAlgError
            DESCRIPTION. P is not positiv definite and symmetric

        Returns
        -------
        sigmas : TYPE np.array(n, dim_sigma)
            DESCRIPTION. sigma points
        W : TYPE np.array(n,)
            DESCRIPTION. Weights for the sigma points
        P_sqrt : TYPE np.array(n,n)
            DESCRIPTION. sqrt(P). Can be inspected if something goes wrong.

        """
        
        if theta is None:
            theta = self.theta
        assert isinstance(theta, float), f"theta must be a float, now it is {type(theta)}"
        assert ((theta > 0) and (theta < 1)), f"theta must be in the range (0,1), excluding end-point values. Current value is {theta}"
        
        if not self.n == mu.shape[0]:
            raise ValueError(f" self.n = {self.n} while mu.shape = {mu.shape}. mu.shape[0] must match self.n!")
        
        if not ((self.n == P.shape[0]) and (self.n == P.shape[1])):
            raise ValueError(f"P.shape = {P.shape}, it must be ({self.n, self.n})")
        
        assert ((mu.ndim == 1) and (P.ndim == 2)), "Error in dimensions of mu and P"
        
        if sqrt_method is None:
            sqrt_method = self.sqrt
        
        n = self.n #dimension of x
        
        if S is None: #assumed symmetric distribution
            S = np.zeros((n,))
        else:
            assert S.ndim == 1, f"S.ndim should be 1, now it is {S.ndim}"
            assert S.shape[0] == n, f"S should contain {n} elements, its shape is {S.shape}"
        if K is None: #assume Gaussian distribution
            K = self.compute_cm4_isserlis_for_multivariate_normal(P)
        else:
            assert K.ndim == 1, f"K.ndim should be 1, now it is {K.ndim}"
            assert K.shape[0] == n, f"K should contain {n} elements, its shape is {K.shape}"
        
        if lbx is None:
            lbx = self.lbx
        if ubx is None:
            ubx = self.ubx
        
        if u is not None:
            assert (u > 0).all(), f"u should be positive, values are {u}"
            assert u.ndim == 1, f"u should be 1D, it is {u.ndim}"
            assert u.shape[0] == n, f"u should have {n} elements, its shape is {u.shape}"
            
        
        dim_sigma = self.dim_sigma
        
        sigmas = np.zeros((n, dim_sigma))
        sigmas[:, 0] = mu
        
        if P_sqrt is None:
            self.P_sqrt = sqrt_method(P)
        else:
            self.P_sqrt = P_sqrt
        
        #compute scaling and weights
        self.u, self.v, Wm = self.compute_scaling_and_weights(P, S, K, u = u)
        Wc = Wm.copy()
        
        for i in range(n):
            sigmas[:, 1 + i] = mu - self.u[i]*self.P_sqrt[:, i]
            sigmas[:, 1 + n + i] = mu + self.v[i]*self.P_sqrt[:, i]
        
            
        if lbx is not None:
            assert isinstance(lbx, (np.ndarray, float, int)), f"lbx is type {type(lbx)}"
            if (sigmas < lbx).any():
                
                assert (mu > lbx).all(), "When constraining sigma-points to be, the assumption is that the mean > lbx"
                self.sigmas_orig = sigmas.copy()
                # idx_neg = np.column_stack(np.where(sigmas[:, 1:1+n] < 0))
                idx_neg = np.column_stack(np.where(sigmas[:, 1:] < 0))
                self.r = np.unique(idx_neg[:, 0])
                self.c = np.unique(idx_neg[:, 1])
                self.idx_neg = idx_neg

                v_redefined = False
                for ci in self.c: #update u,v
                    assert (ci >= 0) and (ci <= 2*n), "Sth wrong here"
                    if ci < n:
                        self.u[ci] = theta*np.min(np.abs((mu - lbx) / self.P_sqrt[:, ci]))
                    else:# ci >= n
                        self.v[ci - n] = theta*np.min(np.abs((lbx - mu) / self.P_sqrt[:, ci - n]))
                        v_redefined = True
                    
                    if not v_redefined: #redefine v
                        P_sqrt_pow3_inv = scipy.linalg.inv(np.power(self.P_sqrt, 3))
                        # P_sqrt_pow4_inv = scipy.linalg.inv(np.power(self.P_sqrt, 4))
                        
                        S_std = P_sqrt_pow3_inv @ S
                        # K_std = P_sqrt_pow4_inv @ K
                    
                        self.v = self.u + S_std
                    
                    #recalculate the sigma points
                    for i in range(n):
                        sigmas[:, 1 + i] = mu - self.u[i]*self.P_sqrt[:, i]
                        sigmas[:, 1 + n + i] = mu + self.v[i]*self.P_sqrt[:, i]
                        
                    #update the weights - ubx should be implemented before this!
                    w2 = 1 / self.v / (self.u + self.v)
                    w1 = w2*self.v/self.u
                    Wm = np.concatenate((np.array([0]), w1, w2))
                    Wm[0] = 1 - np.sum(Wm[1:])
                    Wc = Wm.copy()
                
                self.sigmas = sigmas
                assert (sigmas >= lbx).all(), "Sth wrong when constraining sigma-points to be larger than lbx"
            
        if ubx is not None:
            raise ValueError("ubx not implemented yet")                
        self.sigmas = sigmas
        self.Wm = Wm
        self.Wc = Wc
        return sigmas, Wm, Wc, self.P_sqrt
    
    @staticmethod
    def compute_cm4_isserlis_for_multivariate_normal(P):
        """
        Calculate 4th central moment from Isserli's theorem based on Equation 2.42 in 
        Barfoot, T. (2017). State Estimation for Robotics. Cambridge: Cambridge University Press. doi:10.1017/9781316671528

        Parameters
        ----------
        P : TYPE np.array((dim_x, dim_x))
            DESCRIPTION. Covariance matrix

        Returns
        -------
        cm4 : TYPE np.array((dim_x,))
            DESCRIPTION. 4th central moment of an multivariate normal distribution ("diagonal" terms)
        """
        # dim_x = P.shape[0]
        # # print(Px.shape)
        # I = np.eye(dim_x)
        # cm4 = P @ (np.trace(P)*I + 2*P)
        # cm4 = np.diag(cm4)
        cm4 = 3*np.square(np.diag(P))
        return cm4


# def sqrt_method_robust(P, sqrt_method):
#     try:
#         return sqrt_method(P)
#     except np.linalg.LinAlgError:
#         #increase eigenvalues. Add identity matrix times epsilon on and when it is successfull we're ok
        
#         #obtain "original" standard deviation. If the NUKF is used, this is 1 and this step is redundant
#         std_dev_orig = np.sqrt(np.diag(P))
#         dim_P = P.shape[0]
#         success = False
#         epsilon = 1e-10
#         P2 = P + np.eye(dim_P)*epsilon
#         n_iter = 0
#         while not success:
#             print(f"P2 = P + I*{epsilon}")
#             try:
#                 P_sqrt = sqrt_method(P2)
#                 success = True
#             except np.linalg.LinAlgError:
#                 epsilon *= 10 #increase epsilon
#                 P2 += np.eye(dim_P)*epsilon #increase eigenvalues further
#                 n_iter += 1
#                 if n_iter > 8:
#                     raise ValueError(f"sqrt_method did not converge. Epsilon is now {epsilon}.")
        
#         #Now we know that we can take the sqrt_method. But we need to get correct standard deviations again since we added on the diagonal.
#         std_dev = np.sqrt(np.diag(P2))
#         std_dev_inv = np.diag(1/std_dev)
#         corr = std_dev_inv @ P2 @ std_dev_inv
#         corr_sqrt = sqrt_method(corr) #this should work now
#         print(f"Psqrt success")
#         #return with original standard deviation
#         return np.diag(std_dev_orig) @ corr_sqrt

