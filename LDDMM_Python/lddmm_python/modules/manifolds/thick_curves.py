from pylab import *
from plotly.tools import FigureFactory as FF
import plotly.graph_objs as go
from scipy.spatial.distance import pdist, squareform, cdist

from ..data_attachment.measures import Measures, Measure
from ..data_attachment.currents import Currents, Current
from ..data_attachment.varifolds import Varifolds, Varifold

from .landmarks import Landmarks
from .curves import Curves, Curve

class ThickCurve(Curve) :
	"""
	Encodes a 2D curve with an additional thickness field,
	which allows one to "remove" some segments.
	This class paves the way to the diffeomorphometry of
	partially observed curves.
	
	At the low-level (i.e. this class), thicknesses are considered to be L2 functions :
	they are bluntly added/substracted pointwise.
	However, just like one may choose to put a kernel metric on the "Landmarks"
	information self.points (instead of the basic L2 metric),
	ThickCurves manifolds may choose to penalize thicknesses variations
	so that they can be considered to be "sigmoids of a L2 function", for instance.
	(For more details, please wait a few months...)
	"""
	def __init__(self, points, connectivity, thicknesses, dimension) :
		assert (isvector(thicknesses) and (len(connectivity) == len(thicknesses))), "Thicknesses should be given as a vector, one scalar per segment."
		Curve.__init__(self, points, connectivity, dimension)
		self.thicknesses = thicknesses
		
	"Operations used to update current models."
	def __add__(self, dv) :
		return ThickCurve(self.points      + dv.points,      self.connectivity, 
						  self.thicknesses + dv.thicknesses, self.dimension)
	def __sub__(self, dv) :
		return ThickCurve(self.points      - dv.points,      self.connectivity, 
						  self.thicknesses - dv.thicknesses, self.dimension)
	def __rmul__(self, dt) :
		return ThickCurve(dt * self.points,      self.connectivity, 
						  dt * self.thicknesses, self.dimension)
	def __truediv__(self, n) :
		return ThickCurve(self.points      / n, self.connectivity, 
						  self.thicknesses / n, self.dimension)
	def __neg__(self) :
		return ThickCurve( - self.points,      self.connectivity, 
						   - self.thicknesses, self.dimension)
	def __pos__(self) :
		return ThickCurve(self.points, self.connectivity, self.thicknesses, self.dimension)
		
	def ravel(self) :
		return self
	def to_measure(self) :
		"""
		Same as for Curve, but we modulate the weight of each segment
		according to its thickness.
		"""
		mu = Curve.to_measure(self)
		mu.weights *= self.thicknesses
		return mu
	
	def to_varifold(self) :
		"""
		Same as for Curve, but we modulate the weight of each segment
		according to its thickness.
		"""
		mu = Curve.to_varifold(self)
		mu.weights *= self.thicknesses
		return mu
	
	def to_current(self) :
		"""
		Same as for Curve, but we modulate the length of each segment normal
		according to its thickness.
		"""
		omega = Curve.to_current(self)
		omega.normals *= self.thicknesses
		return omega
	
	@staticmethod
	def distribute_gradient(grad, Q) :
		"""
		Outputs a thick curve "dC" - which should be added to the curve Q as an 'update' -
		with the appropriate connectivity,
		representing in the variable "q" (i.e. the vertices of the curve)
		the gradient "grad", given for the variables ...
		(centers of the segments + lengths + directions)
		"""
		assert (len(grad.points) == len(Q.connectivity)), "Grad.points should be a 2d-array, not a vector."
		dC = zeros( Q.array_shape() )
		for (dx, segment) in zip(grad.points, Q.connectivity) :
			for ind in segment :
				dC[ind] += dx / float(len(segment))
		
		if type(grad) is Current :
			for (dn, segment) in zip(grad.normals, Q.connectivity) :
				# Normal( [a -> b] ) = R_(-90) ( b-a )
				#                    = [ (b-a)_1 ]
				#                      [-(b-a)_0 ]
				#                 dl =   db - da
				#                    = [ - dn_1 ]
				#                      [   dn_0 ]
				#    db += dl/2, da -= dl/2
				dC[segment[1]] += [ -.5*dn[1], .5*dn[0] ]
				dC[segment[0]] -= [ -.5*dn[1], .5*dn[0] ]
		elif type(grad) is Varifold :
			Qvar = Q.to_varifold()
			
			for (dn, dmu, n, mu, segment) in zip(grad.normals, grad.weights, Qvar.normals, Qvar.weights, Q.connectivity) :
				# Normal( [a -> b] ) = R_(-90) ( b-a )
				#                    = [ (b-a)_1 ]
				#                      [-(b-a)_0 ]
				# mun = mu * n -> dn = dmu * n + mu * dn
				#
				#                 dl =   db - da
				#                    = [ - dmun_1 ]
				#                      [   dmun_0 ]
				#    db += dl/2, da -= dl/2
				dmun = dmu * n + mu * dn
				dC[segment[1]] += [ -.5*dmun[1], .5*dmun[0] ]
				dC[segment[0]] -= [ -.5*dmun[1], .5*dmun[0] ]
				
		return Curve( dC.ravel(), Q.connectivity, grad.dimension )
	
	def to_plot(self) :
		"""
		Returns an array of points which can be plotted by Plot.ly.
		Size of the return array ~= ((2+1)*nsegments)-by-dimension
		"""
		separator = array([None]* self.dimension)
		points = self.to_array()
		ret = []
		for segment in self.connectivity :
			ret += [ points[segment[0]], points[segment[1]], separator ]
		return vstack(ret)
		


class ThickCurves(Curves) :
	"""
	Encodes a ThickCurves manifold.
	ThickCurves are seen as Curves with an additionnal thickness scalar parameter
	per segment.
	
	As of today, thicknesses are treated as L2 functions. This means that :
	 - There is no correlation between different segments.
	 - Thicknesses are unconstrained, and can freely move on the real line.
	
	One could implement a "decorrelated + sigmoid" version very simply,
	to enforce "thickness \in [0,1]", for instance.
	
	We treat thicknesses as 'spatially decorrelated sigmoids of L2 functions'.
	This means that, given a pointwoise "sigmoid" function 'sigm',
	which is identified to the induced application
	sigm : L2(X, R)     ->  L2(X, im(sigm))
	       (x -> f(x))  ->  ( x -> sigm(f(x)) )
	we treat a thickness field q as an image sigm( isigm( q ) ),
	where isigm(q) is a L2 function.
	An infinitesimal variation dq is therefore penalized through "isigm'" :
	
	Cost(q -> q + dq ) = Cost_*( isigm(q) -> isigm(q+dq) )
	                   = \int_X |isigm'(q(x)) * dq(x)|^2 dx
	Otherwise said, on the space of thicknesses L2(X, im(sigm)),
	we are using the pushforward metric sigm_* (L2(X, R)).
	
	As the Curve/Surface X is given by a collection of segments/triangles,
	we can simply factor out the lengths/areas of the curve/surface elements,
	and consider our functions to be vectors with as many coordinates
	as there are curve/surface elements.
	The thickness-related cost is then given by
	Cost(q -> q + dq ) = sum_i ( isigmp(q_i) * dq_i )^2
	
	L_q : v -> v * isigmp(q)^2 = p    (pointwise)
	K_q : p -> p / isigmp(q)^2 = v    (pointwise)
	
	Pointwise,      q_i  \in im(sigm) endowed with our metric is isometric to
	          isigm(q_i) \in R        endowed with the standard uniform euclidean metric.
	Momentums p_i can therefore be seen as functions/vectors in L2,
	with
	
	
	However, introducing correlation terms between thicknesses on neighbour segments
	(thus enforcing some kind of smoothness on the thickness stencil)
	would be quite costly, as it would basically involve the computation
	of yet another convolution / kernel matrix.
	"""
	def __init__(self, connectivity = None, npoints = 1, dimension = 2, kernel = ('gaussian', 1), dt=0.1) :
		"""
		Creates a ThickCurves manifold.
		Thicknesses evolutions are handled "tangentially".
		"""
		Curves.__init__(self, connectivity, npoints, dimension, kernel, dt)
		
		if 1 : # Simplistic L2 manifold structure on 'thicknesses'.
			self.sigm   = lambda x : x
			self.isigmp = lambda x : 1
			self.isigm  = lambda x : x
		else : # The metric on thicknesses is the push-forward by tanh of a L2 metric.
			self.sigm   = lambda x : tanh(x)
			self.isigmp = lambda x : 1 / (1 - x**2)
			self.isigm  = lambda x : arctanh(x)
		
	def K(self,q,p, kernels) :
		return ThickCurve( Landmarks.K( self, q.points, p.points, kernels ), q.connectivity , 
						   zeros(len(self.connectivity)),           self.dimension )
		
	def upP(self,q,p, kernels) :
		return ThickCurve( Landmarks.upP( self, q.points, p.points, kernels ), q.connectivity , self.dimension )
	def gradq_pKqz(self, p, q, z, kernels) :
		return ThickCurve( Landmarks.gradq_pKqz( self, p.points, q.points, z.points, kernels ), q.connectivity , self.dimension )
	def dq_gradq_pKqp_a(self, q, p, a, kernels) :
		return ThickCurve( Landmarks.dq_gradq_pKqp_a( self, q.points, p.points, a.points, kernels ), q.connectivity , self.dimension )
	def dq_Kqp_a(self,q,p,a, kernels) :
		return ThickCurve( Landmarks.dq_Kqp_a( self, q.points, p.points, a.points, kernels ), q.connectivity , self.dimension )
	
	# We're gonna have to overload norm_p, and the other cost-related functions which were using K !!!
	
	def shoot_tangential(self, q, p) :
		"""
		Output a new phase point (q,p) which reflects the "tangential shooting"
		that can be done before the regular "Hamiltonian" one.
		
		The "Spatial/Landmarks" part of p is left unchanged.
		However, we will transfer all its "thickness change information" to q.
		"""
		
		q.thicknesses = self.sigm( self.isigm(q.thicknesses) + p.thicknesses )
		
		# We have transferred the thickness change from p to q : no leftover should be kept !
		p.thicknesses = zeros(len(p.connectivity)) 
		
		return (q, p)
		
	def backward_scheme_tangential(self, dq0, dp0, dq1, dp1, qt, pt, kernels_qt) :
		"""
		Modify the gradient term (dq0, dp0) to reflect the evolution
		of the variable which is handled "tangentially" : thicknesses.
		
		As far as the thicknesses '(q/p)i.thicknesses' are concerned,
		we have the explicit shooting equation :
			(q1, p1) = ( sigm( isigm(q0) + p0 ) , p0 )
			
		So that :
			dq1 = 
		
		( As of today, we make a choice : assume that dp1 is indifferent,
		and put all the responsibility of dq1 on dp0.
		In practice, this will imply that the "mean thickness" is never going to change.
		
		This is not very principled, and not very coherent with our atlas estimation
		framework, where the "Frechet" mean is typically more or less free to move
		(the density prior on the template shape, encoded as a flat (Free template) 
		or gaussian (Shooted template) cost term, is never taken punctual,
		except for appariement problems).
		
		However, in our case, this may make sense : we know the topology
		of the complete brain/heart, which is encoded in its "connectivity" matrix :
		allowing thicknesses to delete segments/triangles on the template makes little sense.)
		"""
		# The only line that we really matter in the end
		dp0.thicknesses += dq1.thicknesses * self.isigmp( qt[-1].thicknesses )
		# The other ones are here for the sake of completeness,
		# or for later models
		dp0.thicknesses += 
		dq0.thicknesses += dq1.thicknesses
		
		return (dq0, dp0)
		
		
	"""Non vectorial stuff."""
	def zero_momentum(self) :
		"""
		Default origin of the cotangent plane. 
		"""
		return ThickCurve( zeros(self.d), self.connectivity, zeros(len(self.connectivity)), 2)
	def zero_position(self) :
		"""
		Default origin of the space / the tangent plane. 
		"""
		return ThickCurve( zeros(self.d), self.connectivity, zeros(len(self.connectivity)), 2)
	def mean_position(self, Q0, dQ0p) :
		"""
		(Arithmetic) mean of a list of position updates. 
		"""
		return ThickCurve(                                 mean(vstack([ dQ0.points      for dQ0 in dQ0p]), 0),           self.connectivity, 
						   self.isigmp(Q0.thicknesses) * ( mean(vstack([ dQ0.thicknesses for dQ0 in dQ0p]), 0).ravel() ), 2                  )
						   
	""" Distances """
	def squared_distance(self, Q, Xt) :
		print('ThickCurves cannot be compared pointwise.')
		raise(NotImplementedError)
		
	def distance(self, Q, Xt) :
		print('ThickCurves cannot be compared pointwise.')
		raise(NotImplementedError)
		
		
	def kernel_matching(self, Q, Xt, s = 0.3) :
		(C, dX) = Measures.kernel_matching(Q.to_measure(), Xt.to_measure(), s)
		return (C, ThickCurve.distribute_gradient(dX, Q))
		
	def current_matching(self, Q, Xt, s = 0.3) :
		(C, dX) = Currents.kernel_matching(Q.to_current(), Xt.to_current(), s)
		return (C, ThickCurve.distribute_gradient(dX, Q))
		
	def varifold_matching(self, Q, Xt, s = 0.3) :
		(C, dX) = Varifolds.kernel_matching(Q.to_varifold(), Xt.to_varifold(), s)
		return (C, ThickCurve.distribute_gradient(dX, Q))
		
		
	"""Plot.ly display"""
	def plot_traj(self, qts, **kwargs) :
		"Trajectory display. qt can be an array of coordinates, or a list of such arrays."
		if type(qts[0]) is not list :
			qts = [qts]
		points = []
		separator = array([None]* self.dimension).reshape((1,self.dimension))
		for qt in qts :
			for curve in qt :
				points.append( curve.to_plot() )
				points.append( separator )
		points = vstack(points)
		points = go.Scatter(x = array(points[:,0]), y = array(points[:,1]), mode = 'lines', hoverinfo='name', **kwargs)
		self.current_axis.append(points)
		
	def quiver(self, qt, vt, **kwargs) :
		"Vector field display"
		self.marker(qt, **kwargs)
		
	def marker(self, q, **kwargs) :
		"""Marker field display"""
		if type(q) is not list :
			q = [q]
		points = []
		separator = array([None]* self.dimension)
		for curve in q :
			points.append( curve.to_plot() )
			points.append( separator )
		points = vstack(points)
		points = go.Scatter(x = array(points[:,0]), y = array(points[:,1]), mode = 'lines', hoverinfo='name', **kwargs)
		self.current_axis.append(points)
		
		
		
		
		
		
		
		
		
		
