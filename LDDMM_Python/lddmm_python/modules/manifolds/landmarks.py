from pylab import *
from plotly.tools import FigureFactory as FF
import plotly.graph_objs as go
from scipy.spatial.distance import pdist, squareform, cdist

from .riemannian_manifold import RManifold
from ..data_attachment.measures import Measures, Measure


class Landmarks(RManifold) :
	"""
	Encodes a Landmarks manifold : 
	self = {(x_1,...,x_n) in R^d, x_i != x_j} ~ R^(nd)
	endowed with an appropriate (kernel) metric.
	"""
	def __init__(self, npoints = 1, dimension = 2, kernel = ('gaussian', 1), dt=0.1) :
		"""
		Creates a Landmarks manifold.
		"""
		
		RManifold.__init__(self, npoints * dimension, g=None, dt=dt)
		self.npoints   = npoints
		self.dimension = dimension
		assert(kernel[0] == 'gaussian'), 'The gaussian kernel is the only one that is implemented yet.'
		if kernel[0] == 'gaussian' :
			self.kernel_scale = kernel[1]
			# These three functions will typically account for 90% of the overall computation time
			#self.kernel   = lambda x :   exp(- x / (2* self.kernel_scale ** 2)) # kernel is given |x|^2 as input
			#self.kernelp  = lambda x : - exp(- x / (2* self.kernel_scale ** 2)) / (2* self.kernel_scale ** 2)
			#self.kernelpp = lambda x : + exp(- x / (2* self.kernel_scale ** 2)) / (4* self.kernel_scale ** 4)
		

	def precompute_kernels(self, q) :
		"""
		Returns a tuple of kernel, kernel', kernel'' matrices at position q.
		"""
		x = q.reshape((self.npoints, self.dimension))
		dists = squareform(pdist(x, 'sqeuclidean'))
		K = exp(- dists / (2* self.kernel_scale ** 2))
		
		return (  K, 
		        - K / (2* self.kernel_scale ** 2), 
		          K / (4* self.kernel_scale ** 4))
		
	def K(self,q,p, kernels) :
		"""
		Kernel representation of a cotangent momentum p at position q
		in the tangent space.
		"""
		m = p.reshape((self.npoints, self.dimension))
		K = kernels[0]          # K_ij = k(|x_i-x_j|^2)
		# K = kron(K, eye(self.dimension)) # hugely inefficient, but whatever...
		# return p @ K
		Kq_p = zeros((self.npoints, self.dimension))
		for d in range(self.dimension) :
			Kq_p[:,d] = m[:,d] @ K  # v_nd = (Kq_p)_nd = sum_i k(|x_i-x_j|^2) p_i^d
		return Kq_p.ravel()
		
	def L2_repr_p(self,q,p, kernels) :
		"""
		Mapping from the cotangent plane endowed with Kernel metric
		to R^2 endowed with the standard dot product.
		 K(r, theta)^.5 = ...
		"""
		raise(NotImplementedError)
		
	def upP(self,q,p, kernels) :
		"""
		Returns an update step of the momentum p in the geodesic equations.
		 -.5*d_q (p, K_q p) = ...
		"""
		x = q.reshape((self.npoints, self.dimension))
		p = p.reshape((self.npoints, self.dimension))
		K = kernels[1]          # K_ij       = k'(|x_i-x_j|^2)
		L2prods = p @ p.T       # L2prods_ij = (p_i . p_j) : isotropic kernels
		pKqp_p = K * L2prods    # pKqp_p_ij  = (p_i . p_j) * k'(|x_i-x_j|^2)
		grad = zeros((self.npoints, self.dimension))
		for d in range(self.dimension) :
			diffs = atleast_2d(x[:,d]).T - x[:,d] # diffs_ij = x_i^d - x_j^d
			# grad_nd =        2*sum_i   (x_i^d - x_n^d) * (p_i . p_n) * k'(|x_i-x_n|^2)
			#         = -.5 * (  sum_j 2*(x_n^d - x_j^d) * (p_n . p_j) * k'(|x_n-x_j|^2)
			#                  - sum_i 2*(x_i^d - x_n^d) * (p_i . p_n) * k'(|x_i-x_n|^2) )
			grad[:,d] = 2*sum( diffs * pKqp_p, 0) 
		return grad.reshape((self.npoints * self.dimension,))
	def gradq_pKqz(self, p, q, z, kernels) :
		"""
		Useful for the adjoint integration scheme.
		d_q (p, K_q z) = ...
		"""
		x = q.reshape((self.npoints, self.dimension))
		p = p.reshape((self.npoints, self.dimension))
		z = z.reshape((self.npoints, self.dimension))
		K = kernels[1]           # K_ij       = k'(|x_i-x_j|^2)
		L2prods = p @ z.T        # L2prods_ij = (p_i . z_j) : isotropic kernels
		pKqp_z = K * L2prods     # pKqp_p_ij  = (p_i . z_j) * k'(|x_i-x_j|^2)
		grad = zeros((self.npoints, self.dimension))
		for d in range(self.dimension) :
			diffs = atleast_2d(x[:,d]).T - x[:,d] # diffs_ij = x_i^d - x_j^d
			# grad_nd = sum_i 2*(x_i^d - x_n^d) * (p_i . z_n) * k'(|x_i-x_n|^2)
			#         + sum_j 2*(x_n^d - x_j^d) * (p_n . z_j) * k'(|x_n-x_j|^2)
			grad[:,d] = - sum( 2*diffs * pKqp_z, 0) + sum( 2*diffs * pKqp_z, 1)
		return grad.reshape((self.npoints * self.dimension,))
		
	def dq_gradq_pKqp_a(self, q, p, a, kernels) :
		"""
		Useful for the adjoint integration scheme :
		d_q [ d_q (p, K_q p) ] . a = ...
		"""
		
		
		h = 1e-8
		Q0phA = q + h*a
		Q0mhA = q - h*a
		update_emp =  (  Landmarks.gradq_pKqz(self, p, Q0phA, p, Landmarks.precompute_kernels(self, Q0phA))
					  -  Landmarks.gradq_pKqz(self, p, Q0mhA, p, Landmarks.precompute_kernels(self, Q0mhA))) / (2*h)
		return update_emp
		
		"""
		x = q.reshape((self.npoints, self.dimension))
		p = p.reshape((self.npoints, self.dimension))
		a = a.reshape((self.npoints, self.dimension))
		L2prods = p @ p.T                           # L2prods_ij     = (p_i . p_j) : isotropic kernels
		
		grad = zeros((self.npoints, self.dimension))
		for d in range(self.dimension) :
			diffs = atleast_2d(x[:,d]).T - x[:,d]  # diffs_ij = x_i^d - x_j^d
			# K_ij    = 2*[ k'(|x_i-x_j|^2) + 2* (x_i^d - x_j^d)^2 * k''(|x_i-x_j|^2) ]
			K = 2*( kernels[1] \
				  + 2 * kernels[2] * (diffs**2)) # The two '2' come from the fact that  d(x-y)^2 / dx = 2 * (x-y)
			# We have :
			# [ d_q (p, K_q p) ]_nd = 2* sum_j  (p_n . p_j) * 2*(x_n^d - x_j^d) * k'(|x_n-x_j|^2)
			#                       = 2* sum_j  (p_n . p_j) * f(x_n^d, x_j)
			#                         --> the first factor '2' because we are actually 
			#                             doing a summation over i + a summation over j,
			#                             which can be identified by symmetry.
			# with :
			#     f(x_n^d, x_j) = 2* (x_n^d - x_j^d) * k'( |x_n - x_j|^2)
			#     df/d(x_n^d)   = 2* [ k'( |x_n - x_j|^2) + 2 * (x_n^d - x_j^d)^2 * k''( |x_n - x_j|^2) ]
			# If we note F(q,p) = [ d_q (p, K_q p) ], we have :
			# d_q [ d_q (p, K_q p) ] . a ~= (F(q + dt.a, p) - F(q,p)) / dt 
			# (Gateau derivative in the direction "a" over the variable "q")
			#
			#
			# So that :
			# grad_nd = a_nd * 2 * sum_j (p_n . p_j) * f'(x_n^d, x_j)
			# grad_nd = 2 * a_nd 
			#           * sum_i [ (p_i . p_j) * 2* (k'(|x_i-x_j|^2) 
			#                                       + 2* (x_i^d - x_j^d)^2 * k''(|x_i-x_j|^2) ) ]
			grad[:,d] = a[:,d] * 2 * sum( K * L2prods , 0 ) 
			# The factor '2' comes from the fact that we identify the summation over i with the summation over j
		return grad.reshape((self.npoints * self.dimension,))
		"""
	def dq_Kqp_a(self,q,p,a, kernels) :
		"""
		Useful for the adjoint integration scheme.
		d_q (K_q p) . a  = ...
		"""
		h = 1e-8
		Q0phA = q + h*a
		Q0mhA = q - h*a
		update_emp =  (  Landmarks.K(self, Q0phA, p, Landmarks.precompute_kernels(self, Q0phA))
					  -  Landmarks.K(self, Q0mhA, p, Landmarks.precompute_kernels(self, Q0mhA))) / (2*h)
		return update_emp
		
		"""x = q.reshape((self.npoints, self.dimension))
		p = p.reshape((self.npoints, self.dimension))
		a = a.reshape((self.npoints, self.dimension))
		dists = squareform(pdist(x, 'sqeuclidean')) # dists_ij       = |x_i-x_j|^2
		# We have :
		# [K_q p]_nd = sum_j { k(|x_n - x_j|^2) * p_j^d }
		#
		# So that :
		# grad_nd = a_nd * sum_j { 2 * (x_n^d - x_j^d) * k'(|x_n - x_j|^2) * p_j^d }
		grad = zeros((self.npoints, self.dimension))
		for d in range(self.dimension) :
			diffs = atleast_2d(x[:,d]).T - x[:,d]  # diffs_ij = x_i^d - x_j^d
			
			# K_ij = 2 * (x_i^d - x_j^d) * k'(|x_i - x_j|^2) * p_j^d
			K = 2 * dists * kernels[1] * p[:,d]
			# grad_nd =   a_nd * sum_j { 2 * (x_n^d - x_j^d) * k'(|x_n - x_j|^2) * p_j^d }
			grad[:,d] = a[:,d] * sum( K , 1 )
		return grad.reshape((self.npoints * self.dimension,))"""
	
	""" Distances """
	def squared_distance(self, Q, Xt, *args) :
		"""Returns 1/2 * |I(Q) - Xt|^2 and its Q-gradient."""
		return (.5*sum( (Q-Xt)**2) , Q - Xt)
	def distance(self, Q, Xt, *args) :
		"""Returns |I(Q) - Xt| and its Q-gradient."""
		raise(NotImplementedError)
	
	def kernel_matchings(self, start_scale, end_scale) :
		def curryfied (Q,Xt,progress) :
			return self.kernel_matching(Q, Xt, start_scale + (end_scale - start_scale) * progress ) # Coarse to fine scheme
		return curryfied
	def kernel_matching(self, Q, Xt, s = 0.3) :
		"""
		Implementation of the kernel data attachment term :
		
		d(Q, Xt) = .5 *   sum_{i,j} k( |  Q_i -  Q_j | ) / nobs^2
				 - .5 * 2*sum_{i,j} k( |  Q_i - Xt_j | ) / nobs^2
				 + .5 *   sum_{i,j} k( | Xt_i - Xt_j | ) / nobs^2
		
		where k( d ) = exp( - d^2/(2*s^2) ) is a gaussian kernel
		with std = s.
		See the Phd thesis of Joan Glaunes, Chapter 4, for reference (2005).
		
		This is the most rudimentary tool for the matching of unlabelled data :
		Landmarks are simply seen as sums of dirac measures,
		with *same weight* and *total mass 1*.
		More sophisticated attachment terms such as 'varifold', 'currents'
		or 'optimal transport'/'gromov-wasserstein' are implemented by 
		the Curves2D class.
		"""	
		(C, dMu) = Measures.kernel_matching( Measure( Q.reshape((self.npoints, self.dimension))), 
											 Measure(Xt.reshape((self.npoints, self.dimension))), 
											 s )
		return (C, dMu.points ) # throw away the information about the weights variations
		
	def sinkhorn_matchings(self, sinkhorn_options = None) :
		def curryfied (Q,Xt,progress) :
			return self.sinkhorn_matching(Q, Xt, sinkhorn_options )
		return curryfied
	def sinkhorn_matching(self, Q, Xt, sinkhorn_options) :
		(C, dMu) = Measures.sinkhorn_matching( Measure( Q.reshape((self.npoints, self.dimension))), 
											   Measure(Xt.reshape((self.npoints, self.dimension))), 
											 sinkhorn_options )
		return (C, dMu.points ) # throw away the information about the weights variations
	
	def I(self, q) :
		return vstack(q)
	def show(self, mode='', ax = None) :
		"Manifold display."
		self.layout = go.Layout(
			title='',
			width=800,
			height=800,
			legend = dict( x = .8, y = 1),
			xaxis = dict(range = [-3,3]),
			yaxis = dict(range = [-3,3])
		)
		
	def plot_traj(self, qt, **kwargs) :
		"Trajectory display. qt can be an array of coordinates, or a list of such arrays."
		if type(qt) is not list :
			qt = [qt]
		points = array([]).reshape((0,self.dimension)) # we should pre-allocate...
		separator = array([None]* self.dimension).reshape((1,self.dimension))
		for traj in qt :
			traj = atleast_2d(traj)
			ntimes = traj.shape[0]
			for landmark in range(self.npoints) :
				traj_landmark = traj[:, landmark*(self.dimension) : landmark*(self.dimension) + self.dimension]
				points = vstack((points, traj_landmark, separator))
		
		points = go.Scatter(x = array(points[:,0]), y = array(points[:,1]), mode = 'markers+lines', hoverinfo='name', **kwargs)
		self.current_axis.append(points)

	def quiver(self, qt, vt, **kwargs) :
		"Vector field display"
		self.marker(qt, **kwargs)
		
	def marker(self, q, **kwargs) :
		"""Marker field display"""
		q = atleast_2d(q)
		list_points = []
		separator = array([None]* self.dimension)
		for l in range(q.shape[0]) :
			list_points.append(q[l].reshape((self.npoints, self.dimension)))
			list_points.append( separator )
		points = vstack(list_points)
		points = go.Scatter(x = array(points[:,0]), y = array(points[:,1]), mode = 'markers', hoverinfo='name', **kwargs)
		self.current_axis.append(points)
