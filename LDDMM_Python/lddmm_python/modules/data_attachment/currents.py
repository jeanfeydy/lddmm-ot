from pylab import *
from scipy.spatial.distance import pdist, squareform, cdist


class Current :
	"""
	Encodes a Current as a sum of weighted vector diracs.
	\omega(v) = \sum_i (Omega_i, v(x_i))
	"""
	def __init__(self, points, normals) :
		assert (points.shape[1] == 2), "3D currents have not been implemented yet !"
		assert (points.shape == normals.shape), "A current is given by an array of coordinates + an array of directions"
		self.points = points
		self.normals = normals
		self.dimension = self.points.shape[1]
		
class Currents :
	@staticmethod
	def kernel_matching(Q, Xt, s) :
		"""
		Implementation of the kernel data attachment term :
		
		d(Q, Xt) = .5 *   sum_{i,j} (v_i, v_j) k( |  Q_i -  Q_j | )
				 - .5 * 2*sum_{i,j} (v_i, w_j) k( |  Q_i - Xt_j | )
				 + .5 *   sum_{i,j} (w_i, w_j) k( | Xt_i - Xt_j | )
		where
			Q  = sum_i (v_i, \dirac_{ Q_i}(.) )
			Xt = sum_j (w_j, \dirac_{Xt_j}(.) )
		and where k( d ) = exp( - d^2/(2*s^2) ) is a gaussian kernel
		with std = s.
		
		This can be seen as a ``linear'' matching tool between curves/surfaces :
		  - unlike the "Measures" tool, it takes the orientation into account,
		  - but it only does so in a linear way.
		Today, one may prefer to use Varifolds or Normal Cycles.
		"""	
		# We use a Gaussian kernel
		kernel   = lambda x :   exp(- x / (2* s ** 2)) # kernel is given |x|^2 as input
		kernelp  = lambda x : - exp(- x / (2* s ** 2)) / (2* s ** 2)
		q  =  Q.points
		xt = Xt.points
		v  =  Q.normals
		w  = Xt.normals
		q_dists     = squareform(pdist( q, 'sqeuclidean'))
		cross_dists = cdist(    q, xt,     'sqeuclidean')
		xt_dists    = squareform(pdist(xt, 'sqeuclidean'))
		
		# Matrices of scalar products between normals ('G' stands for Grassmanian)
		Gvv = v @ v.T
		Gvw = v @ w.T
		Gww = w @ w.T
		
		# We're gonna need those two for later calculations
		ker_qq  = kernel(q_dists)
		ker_qxt = kernel(cross_dists)
		
		K_qq   = Gvv * ker_qq
		K_qxt  = Gvw * ker_qxt
		K_xtxt = Gww * kernel(xt_dists)
		# Total data attachment term :
		C = .5 * ( sum(K_qq) - 2*sum(K_qxt) + sum(K_xtxt) )
		
		# Computation of the directional derivatives
		# with respect to the dirac positions
		Kp_qq   = Gvv * kernelp(q_dists)
		Kp_qxt  = Gvw * kernelp(cross_dists)
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
			dv[:,d] = sum( v[:,d] * ker_qq , 1) \
			        - sum( w[:,d] * ker_qxt, 1)
		
		dOmega = Current(dq, dv) 
		return (C, dOmega)
