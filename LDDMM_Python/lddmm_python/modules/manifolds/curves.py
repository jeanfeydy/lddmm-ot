from pylab import *
from plotly.tools import FigureFactory as FF
import plotly.graph_objs as go
from scipy.spatial.distance import pdist, squareform, cdist

from pyvtk import *
from ..io.read_vtk import ReadVTK

from .landmarks import Landmarks
from ..data_attachment.measures import Measures, Measure
from ..data_attachment.currents import Currents, Current
from ..data_attachment.varifolds import Varifolds, Varifold
from ..data_attachment.normal_cycles import NormalCycles, NormalCycle, Cylinders, Spherical

class Curve :
	"""
	Encodes a 2D/3D curve.
	Pythonic class which is especially useful thanks to its io methods.
	"""
	def __init__(self, points, connectivity, dimension) :
		assert (dimension == 2), "3D Curves/Surfaces have not been implemented yet !"
		assert isvector(points), "points should be a npoints*dimension vector."
		self.points = points
		self.connectivity = connectivity
		self.dimension = dimension
	
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
		
	@staticmethod
	def from_file(fname, offset = [0, 0]) :
		a = ReadVTK(fname)
		points = (array(a[0])[:,0:2]) + offset
		connec = array(a[1])
		return Curve(points.ravel(), connec, 2)
	def to_file(self, fname) :
		points = hstack( (self.to_array(), zeros((len(self.to_array()), 1))) )
		vtk = VtkData( PolyData(points = points, polygons = self.connectivity))
		vtk.tofile(fname, 'ascii')
	
	def mean_std(self) :
		"Returns the standard deviation of the mass repartition, which is useful in scripts."
		M = self.to_measure()
		w = (M.weights / sum(M.weights))[:, newaxis]
		points = M.points
		moy    = sum(multiply(points, w), 0)
		return (moy, sqrt( sum( ( (points - moy)**2) * w) ) )
	def translate_rescale(self, m, s) :
		points = self.to_array()
		points = (points - m) / s
		self.points = points.ravel()
		
	def normalize(self) :
		m, s = self.mean_std()
		self.translate_rescale(m,s)
	
	"Operations used to update current models."
	def __add__(self, dv) :
		return Curve(self.points + dv.points, self.connectivity, self.dimension)
	def __sub__(self, dv) :
		if self.points.shape == dv.points.shape :
			return Curve(self.points - dv.points, self.connectivity, self.dimension)
		else :
			return None
	def __rmul__(self, dt) :
		return Curve(dt * self.points, self.connectivity, self.dimension)
	def scale(self, weights) :
		"Row wise multiplication, useful for pointwise density normalization."
		return Curve((multiply(self.to_array(), weights[:, np.newaxis])).ravel(), self.connectivity, self.dimension)
	def __matmul__(self, curve2) :
		"Used in the norm computation..."
		return sum(self.points * curve2.points)
	def __truediv__(self, n) :
		return Curve(self.points / n , self.connectivity, self.dimension)
	def __neg__(self) :
		return Curve(-self.points, self.connectivity, self.dimension)
	def __pos__(self) :
		return Curve(self.points, self.connectivity, self.dimension)
	def array_shape(self) :
		return ( int(self.points.size / self.dimension), self.dimension)
	def to_array(self) :
		"""
		Reshapes self.points from vector to npoint-by-dimension array.
		"""
		return self.points.reshape(self.array_shape()) # This is a view, not a copy !!
	def ravel(self) :
		return self
	def to_measure(self) :
		"""
		Outputs the sum-of-diracs measure associated to the curve.
		Each segment from the connectivity matrix self.c
		is represented as a weighted dirac located in its center,
		with weight equal to the segment length.
		"""
		points = self.to_array()
		centers = zeros((len(self.connectivity), self.dimension))
		lengths = zeros(len(self.connectivity))
		for (i, segment) in enumerate(self.connectivity) :
			centers[i] =           (points[segment[0]] + points[segment[1]]) /2
			lengths[i] = sqrt(sum( (points[segment[1]] - points[segment[0]])**2 ) )
		return Measure( centers, lengths )
	
	def to_current(self) :
		"""
		Outputs the current associated to the curve.
		Each segment [a, b] is represented as a weighted dirac at
		the location (a+b)/2 with direction (b-a), i.e. oriented normal R_(-90) (b-a).
		"""
		points = self.to_array()
		centers = zeros((len(self.connectivity), self.dimension))
		normals = zeros((len(self.connectivity), self.dimension))
		for (i, segment) in enumerate(self.connectivity) :
			centers[i] =    (points[segment[0]] + points[segment[1]]) /2
			
			# Normal( [a -> b] ) = R_(-90) ( b-a )
			#                    = [ (b-a)_1 ]
			#                      [-(b-a)_0 ]
			normals[i] = [ + points[segment[1]][1] - points[segment[0]][1],
						   - points[segment[1]][0] + points[segment[0]][0]  ]
		return Current( centers, normals )
		
	def to_varifold(self) :
		"""
		Outputs the varifold measure associated to the curve.
		Each segment [a, b] is represented as a weighted dirac at
		the location ( (a+b)/2, b-a ) \in R^n x G_2(R^n),
		with weight equal to the segment length.
		"""
		points = self.to_array()
		centers = zeros((len(self.connectivity), self.dimension))
		normals = zeros((len(self.connectivity), self.dimension))
		lengths = zeros(len(self.connectivity))
		for (i, segment) in enumerate(self.connectivity) :
			centers[i] =    (points[segment[0]] + points[segment[1]]) /2
			lengths[i] = sqrt(sum( (points[segment[1]] - points[segment[0]])**2 ) )
			# Normal( [a -> b] ) = R_(-90) ( b-a )
			#                    = [ (b-a)_1 ]
			#                      [-(b-a)_0 ]
			# + renormalization, as the length is given independently
			normals[i] = [ (+ points[segment[1]][1] - points[segment[0]][1]) / lengths[i],
						   (- points[segment[1]][0] + points[segment[0]][0]) / lengths[i] ]
		
		return Varifold( centers, normals, lengths )
		
	def to_normalcycles(self) :
		"""
		Outputs a representation of the normal cycle associated to the curve,
		seen as an union of segments.
		This is a straightforward implementation of the method proposed in
		"Kernel Metrics on Normal Cycles and Application to Curve Matching",
		Pierre Roussillon and Joan Alexis Glaunes, 2015.
		"""
		points = self.to_array()
		cyl_centers = zeros((len(self.connectivity), self.dimension))
		cyl_normals = zeros((len(self.connectivity), self.dimension))
		cyl_lengths = zeros(len(self.connectivity))
		
		sph_points     = points
		sph_directions = [ [] for p in points ]
		sph_weights    = [ [] for p in points ]
		
		for (i, segment) in enumerate(self.connectivity) :
			cyl_centers[i] =    (points[segment[0]] + points[segment[1]]) /2
			cyl_lengths[i] = sqrt(sum( (points[segment[1]] - points[segment[0]])**2 ) )
			# Normal( [a -> b] ) = R_(-90) ( b-a )
			#                    = [ (b-a)_1 ]
			#                      [-(b-a)_0 ]
			# + renormalization, as the length is given independently
			cyl_normals[i] = [ (+ points[segment[1]][1] - points[segment[0]][1]) / cyl_lengths[i],
						       (- points[segment[1]][0] + points[segment[0]][0]) / cyl_lengths[i] ]
			
			# sph_directions[i] is a list of all the unit "to be continued" direction vectors at point i.
			direction_01 = array( [ points[segment[1]][0] - points[segment[0]][0], 
			                        points[segment[1]][1] - points[segment[0]][1] ] ) / cyl_lengths[i]
			# See the definition of N(\tilde{C}) :
			sph_directions[segment[0]].append( + direction_01 )
			sph_weights[segment[0]].append( - 1 )
			sph_directions[segment[1]].append( - direction_01 )
			sph_weights[segment[1]].append( - 1 )
		
		
		
		return NormalCycle(
			Cylinders( 
				centers = cyl_centers,
				normals = cyl_normals,
				weights = cyl_lengths
			),
			Spherical(
				points     = sph_points,
				directions = sph_directions,
				weights    = sph_weights
			)
		)
		
		
		
	# Legacy Python method ===============================================================
	@staticmethod
	def distribute_gradient(grad, Q) :
		"""
		Outputs a curve "dC" - which should be added to the curve Q as an 'update' -
		with the appropriate connectivity,
		representing in the variable "q" (i.e. the vertices of the curve)
		the gradient "grad", given for the variables ...
		(centers of the segments + lengths + directions)
		
		This is nothing but a simple chain rule.
		"""
		grad_sph = None
		if type(grad) is NormalCycle :
			grad_sph = grad.spherical
			# We will treat the cylinders part as a simple varifold term
			grad = Varifold( grad.cylinders.centers, grad.cylinders.normals, grad.cylinders.weights)
			
		assert (len(grad.points) == len(Q.connectivity)), "Grad.points should be a 2d-array, not a vector."
			
		
		dC = zeros( Q.array_shape() )
		
		# Everybody agrees on how to distribute the centers' positions' gradients :
		for (dx, segment) in zip(grad.points, Q.connectivity) :
			# The center is given by
			# x = (C[segment[0]] + ... + C[segment[end]]) / len(segment)
			# i.e. x = (C[seg[0]] + C[seg[1]]) / 2 in the case of curves. 
			#
			# Hence we have 
			# d f / d C[seg[i]] = df/dx * dx/dC[seg[i]] = df/dx / (len(segment))
			for ind in segment :
				dC[ind] += dx / float(len(segment))
				
		# How to distribute the weights variations is more subtle, though :
		if type(grad) is Measure :
			for (dmu, segment) in zip(grad.weights, Q.connectivity) :
				# mu = | C[segment[0]] - C[segment[1]] |
				# d mu / dC[segment[0]] = unit(C[segment[0]] - C[segment[1]])
				# d mu / dC[segment[1]] = unit(C[segment[1]] - C[segment[0]])
				points = Q.to_array()
				u_01 = points[segment[1]] - points[segment[0]]
				u_01 = u_01 / (sqrt(sum( u_01 ** 2)) + 1e-10)
				dC[segment[0]] -= dmu * u_01
				dC[segment[1]] += dmu * u_01
		elif type(grad) is Current :
			for (dn, segment) in zip(grad.normals, Q.connectivity) :
				# Normal( [a -> b] ) = R_(-90) ( b-a )
				#                    = [ (b-a)_1 ]
				#                      [-(b-a)_0 ]
				#                 dl =   db - da
				#                    = [ - dn_1 ]
				#                      [   dn_0 ]
				#    db += dl, da -= dl
				dC[segment[1]] += [ -dn[1], dn[0] ]
				dC[segment[0]] -= [ -dn[1], dn[0] ]
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
				#    db += dl, da -= dl
				dmun = dmu * n + mu * dn
				dC[segment[1]] += [ -dmun[1], dmun[0] ]
				dC[segment[0]] -= [ -dmun[1], dmun[0] ]
		
		if grad_sph is not None :
			Qnor = Q.to_normalcycles()
			
			for (p, dp) in enumerate(grad_sph.points) :
				dC[p] += dp
			who_is_who = [ [] for p in grad_sph.points ]
			for (i, seg) in enumerate(Q.connectivity) :
				who_is_who[seg[0]].append((-1, i))
				who_is_who[seg[1]].append((+1, i))
			for (p, dU_is) in enumerate(grad_sph.directions) :
				for (i, dU_i) in enumerate(dU_is) :
					(eps, edge) = who_is_who[p][i] # We now know what dU_i stands for
					a = Q.connectivity[edge][0]
					b = Q.connectivity[edge][1]
					w = Qnor.spherical.weights[p][i] * Qnor.cylinders.weights[edge] # !!!!
					dC[a] -= eps * w * dU_i / 2
					dC[b] += eps * w * dU_i / 2
				
		return Curve( dC.ravel(), Q.connectivity, grad.dimension )
	
		

# Legacy Python Class. Use TheanoCurves instead =======================================

class Curves(Landmarks) :
	"""
	Encodes a Curves manifold.
	Curves are seen as Landmarks with respect to deformations,
	hence the inheritance structure :
	Curves manifold are implemented as overlays
	of Landmarks manifolds, endowed with appropriate (kernel) metrics
	which manage deformations.
	
	Yet, curves are also endowed with a connectivity structure :
	as Data attachment terms (Measure, Current, Varifold, Normal Cycles)
	are computed with respect to the *segments* of the curves,
	gradient updates will be *distributed* from the segment centers
	to the landmarks.
	"""
	def __init__(self, connectivity = None, npoints = 1, dimension = 2, kernel = ('gaussian', 1), dt=0.1) :
		"""
		Creates a Curves manifold.
		"""
		Landmarks.__init__(self, npoints, dimension, kernel, dt)
		self.connectivity = connectivity
		
	def precompute_kernels(self,q) :
		return Landmarks.precompute_kernels(self, q.points)
		
	def K(self,q,p, kernels) :
		#assert (q.connectivity == p.connectivity), "Momentums should have same connectivity as the base point."
		return Curve( Landmarks.K( self, q.points, p.points, kernels ), q.connectivity , self.dimension )
		
	def upP(self,q,p, kernels) :
		return Curve( Landmarks.upP( self, q.points, p.points, kernels ), q.connectivity , self.dimension )
	def gradq_pKqz(self, p, q, z, kernels) :
		return Curve( Landmarks.gradq_pKqz( self, p.points, q.points, z.points, kernels ), q.connectivity , self.dimension )
	def dq_gradq_pKqp_a(self, q, p, a, kernels) :
		return Curve( Landmarks.dq_gradq_pKqp_a( self, q.points, p.points, a.points, kernels ), q.connectivity , self.dimension )
	def dq_Kqp_a(self,q,p,a, kernels) :
		return Curve( Landmarks.dq_Kqp_a( self, q.points, p.points, a.points, kernels ), q.connectivity , self.dimension )
		
	"""Non vectorial stuff."""
	def zero_momentum(self) :
		"""
		Default origin of the cotangent plane. 
		"""
		return Curve( zeros(self.d), self.connectivity, 2)
	def zero_position(self) :
		"""
		Default origin of the space / the tangent plane. 
		"""
		return Curve( zeros(self.d), self.connectivity, 2)
	def sum_position(self, Q0, dQ0p) :
		"""
		(Arithmetic) mean of a list of position updates. 
		"""
		return Curve( sum(vstack([dQ0.points for dQ0 in dQ0p]), 0), self.connectivity, 2)
	def displacement_norm(self, dQ0p) :
		"""
		~norm of dQ0p, used by FreeAtlas to normalize template updates.
		"""
		return sqrt(mean(vstack([dQ0.points for dQ0 in dQ0p])**2))
		
	"""The following two routines are useful for quasi-Newton optimisation."""
	def L2_product_tangent(self, q, dq1, dq2) :
		return dot(dq1.points.ravel(), dq2.points.ravel())
	def L2_product_cotangent(self, q, p1, p2) :
		return dot(p1.points.ravel(), p2.points.ravel())
		
		
	""" Distances """
	def squared_distance(self, Q, Xt) :
		#print('Curves cannot be compared pointwise.')
		#raise(NotImplementedError)
		#print('This method is only provided for testing purposes.')
		(C, dQ) = Landmarks.squared_distance(self, Q.points, Xt.points)
		print(C)
		return (C, Curve(dQ, Q.connectivity, Q.dimension))
		
	def distance(self, Q, Xt) :
		print('Curves cannot be compared pointwise.')
		raise(NotImplementedError)
		
		
	def kernel_matching(self, Q, Xt, s = 0.3) :
		(C, dX) = Measures.kernel_matching(Q.to_measure(), Xt.to_measure(), s)
		return (C, Curve.distribute_gradient(dX, Q))
	
	def sinkhorn_matchings(self, sinkhorn_options = None) :
		def curryfied (Q,Xt,progress) :
			return self.sinkhorn_matching(Q, Xt, sinkhorn_options )
		return curryfied
	def sinkhorn_matching(self, Q, Xt, sinkhorn_options) :
		(C, dMu, trans) = Measures.sinkhorn_matching(  Q.to_measure(), 
											    Xt.to_measure(), 
											    sinkhorn_options )
		return (C, Curve.distribute_gradient(dMu, Q), trans )
		
	def current_matchings(self, start_scale, end_scale) :
		def curryfied (Q,Xt,progress) :
			return self.current_matching(Q, Xt, start_scale + (end_scale - start_scale) * progress ) # Coarse to fine scheme
		return curryfied
	def current_matching(self, Q, Xt, s = 0.3) :
		(C, dX) = Currents.kernel_matching(Q.to_current(), Xt.to_current(), s)
		return (C, Curve.distribute_gradient(dX, Q))
		
	def varifold_matchings(self, start_scale, end_scale) :
		def curryfied (Q,Xt,progress) :
			return self.varifold_matching(Q, Xt, start_scale + (end_scale - start_scale) * progress ) # Coarse to fine scheme
		return curryfied
	def varifold_matching(self, Q, Xt, s = 0.3) :
		(C, dX) = Varifolds.kernel_matching(Q.to_varifold(), Xt.to_varifold(), s)
		return (C, Curve.distribute_gradient(dX, Q))
		
	def normalcycles_matchings(self, start_scale, end_scale) :
		def curryfied (Q,Xt,progress) :
			return self.normalcycles_matching(Q, Xt, start_scale + (end_scale - start_scale) * progress ) # Coarse to fine scheme
		return curryfied
	def normalcycles_matching(self, Q, Xt, s = 0.3) :
		(C, dX) = NormalCycles.kernel_matching(Q.to_normalcycles(), Xt.to_normalcycles(), s)
		return (C, Curve.distribute_gradient(dX, Q))
		
	def I(self, q) :
		return q
		
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
		
	def show_transport(self, transports, **kwargs) :
		"""Display of a wasserstein transport plan."""
		
		points = []
		separator = array([None]* self.dimension)
		for trans in transports :
			(gamma, q, xt) = trans
			Q  = q.to_measure()
			Xt = xt.to_measure()
			xtpoints = xt.to_array()
			for (a, mui, gi) in zip(Q.points, Q.weights, gamma) :
				# gi = sum(Q.weights) * gi / mui
				gi = gi / mui
				for (seg, gij) in zip(xt.connectivity, gi) :
					mass_per_line = 0.25
					if gij >= mass_per_line :
						#points += [a, b]
						#points.append( separator )
						nlines = floor(gij / mass_per_line)
						ts = linspace(.35, .65, nlines)
						for t in ts :
							b = (1-t) * xtpoints[seg[0]] + t * xtpoints[seg[1]]
							points += [a, b]
							points.append( separator )
		if points != [] :
			points = vstack(points)
			points = go.Scatter(x = array(points[:,0]), y = array(points[:,1]), mode = 'lines', hoverinfo='name', **kwargs)
		else :
			points = go.Scatter(x = array([0]), y = array([0]), mode = 'lines', hoverinfo='name', **kwargs)
		self.current_axis.append(points)
		
		
		
		
		
		
