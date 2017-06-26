from pylab import *
import plotly.graph_objs as go
from ..io.my_iplot import my_iplot

class RManifold :
	"Encodes a Riemannian Manifold."
	def __init__(self, d, g=None, dt=0.1) :
		"""
		Creates a Riemannian Manifold.
		
		N.B. : This class implements the geodesic forward and adjoint backward equations
		       by flowing a discretized Hamiltonian flow.
		       However, sometimes, one may need to handle some variables differently,
		       using a "tangential" "shooting" scheme :
		       - the "signal" part of fshapes, for which the "tangential"
		         denomination has been coined,
		         which is typically very costly to implement "properly".
		         
		       - the "thickness" signal for a ThickCurve or ThickSurface, 
		         which can be restricted to lie in the segment [0,1] 
		         through the use of a sigmoidal function (see future works).
		         As a discrete scheme may bring us out of bounds,
		         I find it safer to implement it "explicitely"
		         (this can be done in the spatially decorrelated case).
		         
		       Such manipulations are made possible here through the use
		       of two overloaded methods : shoot_tangential and backward_scheme_tangential.
		
		Arguments :
		d -- dimension of the Manifold, typically 2 or 3
		g -- metric field : @(x) -> g(x), a d-by-d symmetric >0 array
		dt -- step used by the shooting routine
		"""
		self.d = d
		self.g = g
		self.dt = dt # 0.1 by default
		self.current_axis = []
	
	def precompute_kernels(self, q) :
		"""
		Returns None, or a tuple of kernel, kernel', kernel'' matrices at position q.
		"""
		return None # The default behavior is to precompute nothing. Landmarks manifold overrides that.
	
	def shoot(self, q0, p0) :
		"""
		Geodesic shooting on the manifold.
		
		Arguments :
		q0 -- start point coordinates, d-by-1 array
		p0 -- initial momentum, d-by-1 array
		"""
		#assert (q0.shape == (self.d, )), 'Wrong dimension of the starting point.'
		#assert (p0.shape == (self.d, )), 'Wrong dimension of the shooting momentum.'
		(q0,p0) = self.shoot_tangential(q0, p0) # Handle tangential shooting at the beginning
		
		nsteps = round(1/self.dt) + 1
		q = q0 # current position
		p = p0 # current momentum
		
		# We could do without all this precomputation stuff,
		# but we would end-up recomputing identical kernel matrices ~5-6 times...
		# at a consequent computational cost.
		# All in all, adding a few lines of "precomputation" here and there 
		# is definitely worth it.
		prec_q = self.precompute_kernels(q) 
		
		v = self.K(q,p, prec_q) # current speed
		qt      = [q0]
		pt      = [p0]
		vt      = [v]
		prec_qt = [prec_q]
		
		# Simplistic Euler scheme.
		# The use of higher order methods such as 'ode45' should be implemented ~soon.
		for it in range(1, nsteps) :
			dq = self.dt * v              # q' = K_q * p
			dp = self.dt * self.upP(q, p, prec_q) # p' = - .5 * d_q (p, K_q*p)
			q = q + dq      # current position
			p = p + dp      # current momentum
			prec_q = self.precompute_kernels(q) # current kernel matrices
			v = self.K(q,p, prec_q) # current speed
			
			qt.append(q)
			pt.append(p)
			vt.append(v)
			prec_qt.append(prec_q)
			
		return (q, qt, pt, vt, prec_qt) # we are often interested in the sole end point : q1 = q
	def shoot_tangential(self, q, p) :
		"""
		Output a new phase point (q,p) which reflects the "tangential shooting"
		that can be done before the regular "Hamiltonian" one.
		"""
		return (q, p) # By default, everything is handled by the discretized Hamiltonian scheme
		
	def backward_scheme(self, dq1, dp1, qt, pt, prec_qt) :
		"""
		Implementation of the adjoint equations, backward euler scheme.
		If (qt, pt) is a geodesic trajectory in phase space, 
		and if (dq1, dp1) is the gradient of a functional g(q1,p1) in phase space,
		then the output (dq0, dp0) is the gradient of g with respect to the 't=0' variables
		(q0,p0), as (q1,p1) is computed from them by integration of the Hamiltonian
		flow in phase space on the time interval t = [0,1].
		"""
		nsteps = len(qt)
		assert nsteps == round(1/self.dt) + 1, 'Trajectories don t have the right number of steps'
		
		z = dq1
		a = dp1
		for it in range(0, (nsteps-1)) :
			col = nsteps - it - 1 # col = -1, -2, ..., 1
			# We work on the time interval 1 - it/(nsteps-1) = t -> t - dt
			q = qt[col]
			p = pt[col]
			prec_q = prec_qt[col] # Our precomputed kernel matrices...
			dz = - self.gradq_pKqz(p,q,z, prec_q) + .5 * self.dq_gradq_pKqp_a(q,p,a, prec_q)
			da = - self.K(q, z, prec_q)           +      self.dq_Kqp_a(q,p,a, prec_q)
			z = z - self.dt * dz # remember we're going BACKWARDS
			a = a - self.dt * da # remember we're going BACKWARDS
		
		(z,a) = self.backward_scheme_tangential(z,a, dq1, dp1, qt, pt, prec_qt)
			
		return (z, a)
	def backward_scheme_tangential(self, dq0, dp0, dq1, dp1, qt, pt, prec_qt) :
		"""
		Modify the gradient term (dq0, dp0) to reflect the evolution
		of the variables which are handled "tangentially".
		"""
		return (dq0, dp0) # By default, everything is treated by the regular backward scheme
		
	def zero_momentum(self) :
		"""
		Default origin of the cotangent plane. 
		This method should be overriden by complex manifolds which use non-vectorial momentums.
		"""
		return zeros(self.d)
	def zero_position(self) :
		"""
		Default origin of the space / the tangent plane. 
		This method should be overriden by complex manifolds which use non-vectorial positions.
		"""
		return zeros(self.d)
	def sum_position(self, Q0, dQ0p) :
		"""
		(Arithmetic) sum of a list of position updates. 
		This method should be overriden by complex manifolds which use non-vectorial positions.
		"""
		return sum(dQ0p, 0)
	def displacement_norm(self, dQ0p) :
		"""
		~norm of dQ0p, used by FreeAtlas to normalize template updates.
		This method should be overriden by complex manifolds which use non-vectorial positions.
		"""
		return sqrt(mean(array(dQ0p)**2))
	def norm(self,q,v, prec_q) :
		"Returns the norm of a tangent vector v at position q."
		v = atleast_2d(v)
		nvects = v.shape[0] # 1st dimension
		n = zeros(nvects)
		for i in range(0, nvects) :
			n[i] = sqrt( v[i] @ self.g(q) @ v[i] )
		return n
		
	def norm_p(self,q,p, prec_q) :
		"Returns the norm of a cotangent momentum p at position q."
		if isinstance(p, ndarray) :
			p = atleast_2d(p)
		elif (type(p) is not list):
			p = [p]
		nvects = len(p) 
		n = zeros(nvects)
		for (i, p_i) in enumerate(p) :
			n[i] = sqrt( p_i @ self.K(q, p_i, prec_q).ravel() )
		return n
		
	"""The following two routines are useful for quasi-Newton optimisation."""
	def L2_product_tangent(self, q, dq1, dq2) :
		return dot(dq1.ravel(), dq2.ravel())
	def L2_product_cotangent(self, q, p1, p2) :
		return dot(p1.ravel(), p2.ravel())
		
		
	def momentum_quadratic_cost(self, q, p, prec_q) :
		"""
		Typically, outputs .5*(p, K_q p). 
		But it can be overridden by manifolds which use tangential variables.
		"""
		return .5 * ( p @ self.K(q, p, prec_q).ravel() )
	def d_momentum_quadratic_cost(self, q, p, prec_q) :
		"""
		Typically, outputs (K_q p, d_q [.5*(p, K_q p)] ). 
		But it can be overridden by manifolds which use tangential variables.
		"""
		return (self.K(q, p, prec_q), - self.upP(q,p, prec_q) ) # upP = -.5 d_q (p,Kq p) )
		
		
	def unit_v(self,q,v) :
		"Normalizes a tangent vector v at position q."
		# vn = v ./ repmat(M.norm(q, v), [size(v,1), 1]);
		return linalg.solve(sqrt(self.g(q)), v)
	def unit_p(self,q,p, prec_q) :
		"Normalizes a cotangent momentum p at position q."
		return p / self.norm_p(q, p, prec_q)
	def K(self,q,p, prec_q) :
		"""
		Kernel representation of a cotangent momentum p at position q
		in the tangent space.
		'prec_q' (which stands for 'precomputed_q' is an object, 
		which gives useful data to accelerate the computation.
		Typically, 'prec_q' will be None (if no acceleration is deemed necessary),
		or a tuple of pre-computed kernel matrices.
		"""
		print('Not tested yet ! Very inefficient kernel computation !')
		return linalg.solve(self.g(q), p)
		
		
		
	"""To be implemented by successors :"""
	def L2_repr_p(self,q,p, prec_q) :
		"""
		Mapping from the cotangent plane endowed with Kernel metric
		to R^2 endowed with the standard dot product.
		 K(r, theta)^.5 = ...
		"""
		raise(NotImplementedError)
		
	def upP(self,q,p, prec_q) :
		"""
		Returns an update step of the momentum p in the geodesic equations.
		- .5*d_(r,theta) (p, K_(r,theta) p) = ...
		"""
		raise(NotImplementedError)
	def gradq_pKqz(self, p, q, z, prec_q) :
		"""
		Useful for the adjoint integration scheme.
		d_(r,theta) (p, K_(r,theta) z) = ...
		"""
		raise(NotImplementedError)
	def dq_gradq_pKqp_a(self, q, p, a, prec_q) :
		"Useful for the adjoint integration scheme."
		raise(NotImplementedError)
	def dq_Kqp_a(self,q,p,a, prec_q) :
		"""
		Useful for the adjoint integration scheme.
		d_(r,theta) (K_(r,theta) p) . a  = ...
		"""
		raise(NotImplementedError)
	
	""" Distances """
	def squared_distance(self, Q, Xt, *args) :
		"""Returns 1/2 * |I(Q) - Xt|^2 and its Q-gradient."""
		raise(NotImplementedError)
	def distance(self, Q, Xt, *args) :
		"""Returns |I(Q) - Xt| and its Q-gradient."""
		raise(NotImplementedError)
		
	def iplot(self, title) :
		"Interactive manifold display "
		self.layout['title'] = title
		return my_iplot(go.Figure(data=self.current_axis, layout=self.layout))
		
	def show(self, mode='', ax = None) :
		"Manifold display."
		raise(NotImplementedError)
		
	def plot_traj(self, qt, **kwargs) :
		"Trajectory display. qt can be an array of coordinates, or a list of such arrays."
		raise(NotImplementedError)

	def quiver(self, qt, vt, **kwargs) :
		"Vector field display"
		raise(NotImplementedError)
		
	def marker(self, q, **kwargs) :
		"""Marker field display"""
		raise(NotImplementedError)

