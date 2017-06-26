from pylab import *
from scipy.spatial.distance import pdist, squareform, cdist

import theano
import theano.tensor as T

from .sinkhorn import sinkhorn_log, _sinkhorn_log, SinkhornOptions
from ..math_utils.kernels import _squared_distances, _gaussian_cross_kernels

from collections import namedtuple
VarifoldOptions = namedtuple('VarifoldOptions', 'orientation_weight orientation_order')


class Varifold :
	"""
	Encodes a Varifold as a sum of weighted diracs on R^n x G_{n-1}(R^n) ~= R^n x (R^n/{R}) .
	\mu_Q = \sum_i weight_i \dirac_(point_i, normals_i^{\orth})
	"""
	def __init__(self, points, normals, weights) :
		assert (points.shape == normals.shape), "A varifold is given by an array of coordinates + an array of directions + a vector of weights."
		assert (isvector(weights) and (len(weights) == len(points)) ), "Points, normals and weights should have the same length."
		#assert ( all( abs(sqrt(sum( normals ** 2, 1 )) - 1) < 0.001 ) ), "Tangents should be unit vectors, as length is encoded in 'weights'."
		self.points   = points
		self.normals = normals
		self.weights  = weights
		self.dimension = self.points.shape[1]
		
class Varifolds :
	# Theano symbolic methods ======================================================
	@staticmethod
	def _kernel_matching(q1_x, q1_mu, q1_n, xt_x, xt_mu, xt_n, radius) :
		"""
		Theano symbolic implementation of the kernel_matching method.
		As of today, we only implemented the Cauchy-Binet angular kernel.
		"""
		K_qq, K_qx, K_xx = _gaussian_cross_kernels(q1_x, xt_x, radius)
		
		V_qq = (q1_n.dot(q1_n.T)) ** 4
		V_qx = (q1_n.dot(xt_n.T)) ** 4
		V_xx = (xt_n.dot(xt_n.T)) ** 4
		
		q1_mu = q1_mu.dimshuffle(0,'x')  # column
		xt_mu = xt_mu.dimshuffle(0,'x') 
		return [.5 * (   T.sum(K_qq * V_qq * q1_mu.dot(q1_mu.T)) \
		            +   T.sum(K_xx * V_xx * xt_mu.dot(xt_mu.T)) \
		            - 2*T.sum(K_qx * V_qx * q1_mu.dot(xt_mu.T))  ), 0.*q1_n.dot(xt_n.T)]
	
	@staticmethod
	def _sinkhorn_matching(q1_x, q1_mu, q1_n, xt_x, xt_mu, xt_n, cost_options, sinkhorn_options) :
		"""
		Theano symbolic implementation of the sinkhorn data attachment term.
		We use a cost ("quadratic distance") function of the form
		C( (x_i, n_i), (y_j, m_j) ) = .5 * |x_i-y_j|^2 * (1. + a * (1 - (n_i.m_j)^{k}) )
		where x_i, y_j are positions and n_i, m_j two orientations, encoded as unit-length vectors.
		Remember that two curve/surface elements will be "matched" by the sinkhorn algorithm
		if the associated pairwise cost is *small* : the adjunction of a factor
		 " a * (1 - (n_i.m_j)^{k}) "
		therefore allows us to match  preferentially shape elements whose orientations are "similar",
		i.e. such that (n_i.m_j)^{k} = cos^k(theta) ~ 1, where theta is the angle between the normals.
		
		'a' : cost_options.orientation_weight controls the orientation influence (a = 0 -> simple measure matching)
		'k' : cost_options.orientation_order  controls the angular selectivity.
		      k = 1 will result in a current-like data attachment term.
		      k = 2 is akin to a Cauchy-Binet kernel, which cannot distinguish two rotated cross
		            (cos^2(theta) + cos^2(pi/2+theta) = 1 for any value of theta).
		      We recommend using selective *even* values of k in the range 4-8, depending on your data.
		"""
		rho = sinkhorn_options.rho
		# Cost function :
		C = .5 * _squared_distances(q1_x, xt_x) * (1. + cost_options.orientation_weight * (1 - (q1_n.dot(xt_n.T))**cost_options.orientation_order ))
		#C = .5 * _squared_distances(q1_x, xt_x) + .5 * ( cost_options.orientation_weight * (1 - (q1_n.dot(xt_n.T))**cost_options.orientation_order ))
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
		
		d(Q, Xt) = .5 *   sum_{i,j} mu_i*mu_j * (v_i, v_j)^2 k( |  Q_i -  Q_j | )
				 - .5 * 2*sum_{i,j} mu_i*nu_j * (v_i, w_j)^2 k( |  Q_i - Xt_j | )
				 + .5 *   sum_{i,j} nu_i*nu_j * (w_i, w_j)^2 k( | Xt_i - Xt_j | )
		where
			Q  = sum_i mu_i \dirac_{ ( Q_i, v_i^{\orth}) }
			Xt = sum_j nu_i \dirac_{ (Xt_i, w_i^{\orth}) }
		and where k( d ) = exp( - d^2/(2*s^2) ) is a gaussian kernel
		with std = s.
		
		This can be seen as a ``quadratic'' matching tool between curves/surfaces :
		whereas the Current approach was comparing tangent spaces / normals
		making an angle 'theta'
		with the             'linear'    term cos  (theta),
		we are now using the 'quadratic' term cos^2(theta),
		which is orientation-independent.
		Given two segments [a->b] and [c->d], we have replaced
		  ( R_(-90)b-a , R_(-90)d-c )   = ( b-a, d-c )
		                                =  |b-a|*|d-c|*  cos(theta)
		with
		|b-a|*|d-c|*(n_{b-a},n_{d-c})^2 =  |b-a|*|d-c|*cos^2(theta)
		
		Instead of a quadratic kernel, one could also use a pseudo-gaussian kernel
		exp( -cos^2(theta) / (2 s^2) ),
		thus gaining a new parameter 's' (angular sensitivity)
		at a greater computational cost.
		"""	
		# We use a Gaussian kernel
		kernel   = lambda x :   exp(- x / (2* s ** 2)) # kernel is given |x|^2 as input
		kernelp  = lambda x : - exp(- x / (2* s ** 2)) / (2* s ** 2)
		
		# Simpler variable names...
		q  =  Q.points
		xt = Xt.points
		mu =  Q.weights
		nu = Xt.weights
		v  =  Q.normals
		w  = Xt.normals
		
		# Compute the squared distances between points in the euclidean space
		q_dists     = squareform(pdist( q, 'sqeuclidean'))
		cross_dists = cdist(    q, xt,     'sqeuclidean')
		xt_dists    = squareform(pdist(xt, 'sqeuclidean'))
		
		# Matrices of scalar products between normals ('SG' stands for Squared Grassmanian)
		Gvv  = v @ v.T
		Gvw  = v @ w.T
		Gww  = w @ w.T
		SGvv = Gvv**2
		SGvw = Gvw**2
		SGww = Gww**2
		
		# Matrices of products mu_i mu_j, ... ('T' stands for Tensor product)
		Tmumu = atleast_2d(mu).T * mu
		Tmunu = atleast_2d(mu).T * nu
		Tnunu = atleast_2d(nu).T * nu
		
		# We're gonna need those two for later calculations
		ker_qq  = kernel(q_dists)
		ker_qxt = kernel(cross_dists)
		
		K_qq   = Tmumu * SGvv * ker_qq
		K_qxt  = Tmunu * SGvw * ker_qxt
		K_xtxt = Tnunu * SGww * kernel(xt_dists)
		
		# Total data attachment term :
		C = .5 * ( sum(K_qq) - 2*sum(K_qxt) + sum(K_xtxt) )
		
		# Computation of the directional derivatives
		# with respect to the dirac positions
		Kp_qq   = Tmumu * SGvv * kernelp(q_dists)
		Kp_qxt  = Tmunu * SGvw * kernelp(cross_dists)
		dq = zeros(q.shape)
		for d in range(q.shape[1]) :
			qi_min_qj  = atleast_2d(q[:,d]).T - atleast_2d( q[:,d])
			qi_min_xtj = atleast_2d(q[:,d]).T - atleast_2d(xt[:,d])
			dq[:,d] =   ( sum( qi_min_qj  * Kp_qq , 1) \
			         - 2* sum( qi_min_xtj * Kp_qxt, 1) )
		
		# Computation of the directional derivatives
		# with respect to the normals v
		dv = zeros(v.shape)
		for d in range(v.shape[1]) :
			dv[:,d] = sum( 2 * (v[:,d] - atleast_2d(v[:,d]).T * Gvv) * Gvv * ker_qq , 1) \
			        - sum( 2 * (w[:,d] - atleast_2d(v[:,d]).T * Gvw) * Gvw * ker_qxt, 1)
			        
		# Computation of the directional derivatives
		# with respect to the weights mu
		dmu = zeros(mu.shape)
		dmu = sum( mu * SGvv * ker_qq , 1) \
		    - sum( nu * SGvw * ker_qxt, 1)
		
		
		dV = Varifold(dq, dv, dmu) 
		return (C, dV)
