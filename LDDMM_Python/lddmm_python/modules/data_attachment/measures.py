from pylab import *
from scipy.spatial.distance import pdist, squareform, cdist

import theano
import theano.tensor as T

from .sinkhorn import sinkhorn_log, _sinkhorn_log, SinkhornOptions
from ..math_utils.kernels import _squared_distances, _gaussian_cross_kernels

class Measure :
	"""
	Encodes a Measure as a sum of weighted diracs.
	"""
	def __init__(self, points, weights = None) :
		self.points = points
		if weights is None :
			weights = ones(len(self.points)) / len(self.points) # Probability measure
		self.weights = weights
		self.dimension = self.points.shape[1]
		
class Measures :
	
	# Theano symbolic methods ======================================================
	@staticmethod
	def _kernel_matching(q1_x, q1_mu, xt_x, xt_mu, radius) :
		"""Theano symbolic method."""
		K_qq, K_qx, K_xx = _gaussian_cross_kernels(q1_x, xt_x, radius)
		
		q1_mu = q1_mu.dimshuffle(0,'x')  # column
		xt_mu = xt_mu.dimshuffle(0,'x') 
		return [.5 * (   T.sum(K_qq * q1_mu.dot(q1_mu.T)) \
		            +   T.sum(K_xx * xt_mu.dot(xt_mu.T)) \
		            - 2*T.sum(K_qx * q1_mu.dot(xt_mu.T))  ), []]
		
	
	@staticmethod
	def _sinkhorn_matching(q1_x, q1_mu, xt_x, xt_mu, sinkhorn_options) :
		"""Theano symbolic method."""
		rho = sinkhorn_options.rho
		C = _squared_distances(q1_x, xt_x) # Cost function
		mu = q1_mu
		nu = xt_mu
		if rho == inf : # Balanced transport : we normalize the total weights
			mu = mu / T.sum(mu)
			nu = nu / T.sum(nu)
			
		return _sinkhorn_log( mu, nu, C, sinkhorn_options)
		
	# Legacy Python methods ======================================================
	@staticmethod
	def kernel_matching(Q, Xt, s) :
		"""
		Implementation of the kernel data attachment term :
		
		d(Q, Xt) = .5 *   sum_{i,j} mu_i*nu_j* k( |  Q_i -  Q_j | )
				 - .5 * 2*sum_{i,j} mu_i*nu_j* k( |  Q_i - Xt_j | )
				 + .5 *   sum_{i,j} mu_i*nu_j* k( | Xt_i - Xt_j | )
		where
			Q  = sum_i mu_i \dirac_{ Q_i}
			Xt = sum_j nu_j \dirac_{Xt_j}
		and where k( d ) = exp( - d^2/(2*s^2) ) is a gaussian kernel
		with std = s.
		See the Phd thesis of Joan Glaunes, Chapter 4, for reference (2005).
		
		This is the most rudimentary tool for the matching of unlabelled data :
		Landmarks are simply seen as sums of dirac measures,
		with *same weight* and *total mass 1*.
		More sophisticated attachment terms such as 'varifold', 'currents'
		or 'optimal transport'/'gromov-wasserstein' are implemented by 
		the Curves2D class.
		"""	
		# We use a Gaussian kernel
		kernel   = lambda x :   exp(- x / (2* s ** 2)) # kernel is given |x|^2 as input
		kernelp  = lambda x : - exp(- x / (2* s ** 2)) / (2* s ** 2)
		q  =  Q.points
		xt = Xt.points
		mu =  Q.weights
		nu = Xt.weights
		q_dists     = squareform(pdist( q, 'sqeuclidean'))
		cross_dists = cdist(    q, xt,     'sqeuclidean')
		xt_dists    = squareform(pdist(xt, 'sqeuclidean'))
		# We store those two as they will be useful later
		k_qq  = kernel(q_dists)
		k_qxt = kernel(cross_dists)
		# Compute the products...
		K_qq   = (atleast_2d( mu).T * mu) * k_qq
		K_qxt  = (atleast_2d( mu).T * nu) * k_qxt
		K_xtxt = (atleast_2d( nu).T * nu) * kernel(xt_dists)
		# Total data attachment term :
		C = .5 * ( sum(K_qq) - 2*sum(K_qxt) + sum(K_xtxt) )
		
		# Computation of the directional derivatives
		# with respect to the dirac positions
		Kp_qq   = (atleast_2d( mu).T * mu) * kernelp(q_dists)
		Kp_qxt  = (atleast_2d( mu).T * nu) * kernelp(cross_dists)
		dq = zeros(q.shape)
		for d in range(q.shape[1]) :
			qi_min_qj  = atleast_2d(q[:,d]).T - atleast_2d( q[:,d])
			qi_min_xtj = atleast_2d(q[:,d]).T - atleast_2d(xt[:,d])
			# The factor 2 is here because we tale advantage of the kernel
			# symmetry : instead of summing on rows + columns, 
			# we take 2* the summation on rows.
			dq[:,d] = 2*( sum( qi_min_qj  * Kp_qq , 1) \
			            - sum( qi_min_xtj * Kp_qxt, 1) )
			            
		# with respect to the dirac weights
		dmu = k_qq.dot(mu) - k_qxt.dot(nu)
		
		# Combine the two
		dMu = Measure(dq, dmu)
		return (C, dMu)
		
		
	@staticmethod
	def sinkhorn_matching(Q, Xt, sinkhorn_options = None) :
		"""
		Implements the gradient of the wasserstein distance,
		using the Sinkhorn algorithm to get the transport plan.
		"""
		if sinkhorn_options is None :
			sinkhorn_options = SinkhornOptions( epsilon = 0.1,
												niter   = 1000,
												rho     = 1.,
												tau     = 0.
											   )
		rho = sinkhorn_options.rho
		def nablaC(X,Y) :
			M = zeros( (X.shape[1], len(X), len(Y)) )
			for d in range(X.shape[1]) :
				M[d] = atleast_2d(X[:,d]).T - Y[:,d]
			return M
		def C(X,Y) :
			M = (nablaC(X,Y))**2
			return .5 * sum(M, 0)
			
		mu = Q.weights
		nu = Xt.weights
		if rho == inf :
			munnorm = mu
			mutot = sum(mu)
			mu = mu / sum(mu)
			nu = nu / sum(nu)
			
		(u, v, gamma, cost, errs) = sinkhorn_log( mu, nu,
										    C(Q.points, Xt.points),
										    sinkhorn_options,
										    Measures.sinkhorn_matching.warm_restart)
		#print( sqrt( sum( (sum(gamma, 1) - mu) ** 2) ))
		#print( sqrt( sum( (sum(gamma, 0) - nu) ** 2) ))
		
		#print(errs)
		#Measures.sinkhorn_matching.warm_restart = (u,v)
		if rho == inf :
			u -= mean(u)
			nabla_mu = u
			nabla_mu = nabla_mu * (mutot - munnorm) / mutot**2
		else :
			nabla_mu = - rho * (exp( -u /rho ) - 1)
			
		gradC_q = nablaC(Q.points, Xt.points) # Size d * nq * nxt
		nabla_x = zeros(Q.points.shape)       # size nq * d
		for d in range(nabla_x.shape[1]) :
			nabla_x[:, d] = sum( gradC_q[d] * gamma, 1)
		#nabla_x = (nabla_x.T / sqrt(sum(nabla_x**2, 1))).T
		M = .5 * sum(nablaC( Q.points, Xt.points ) **2, 0)
		#print("Error in the iterative Sinkhorn scheme, from : ", errs[0])
		#print("                                          to : ", errs[-1])
		return (cost, Measure( nabla_x, nabla_mu), gamma )
		
		
Measures.sinkhorn_matching.warm_restart = None
		
		
		
		
		
		
