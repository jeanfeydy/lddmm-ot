
import numpy as np
import theano
import theano.tensor as T
from   theano import pp, config

from pyvtk import VtkData, PolyData, Vectors, PointData, Scalars

from .curves import Curve
from ..data_attachment.measures  import Measures
from ..data_attachment.varifolds import Varifolds



class ShapesManifold :
	"""
	Class which handles all the shape-specific data attachment methods
	+ the file io methods (marker, plot_traj, ...).
	It should be inherited by any manifold class which handles sparse (codim >= 1)
	data.
	"""
	def __init__(self, Q0, 
	                   data_attachment       = ('measure-kernel', ('gaussian', 1))
		               ) :
		
		self.connectivity    = Q0.connectivity
		self.array_shape     = Q0.array_shape()
		self.dimension       = self.array_shape[1]
		
		self.data_attachment = data_attachment
		self.att_param      = data_attachment[1]
		
		#===============================================================
		# Theano functions can only handle tensor data - not even tuples.
		# Before compilation, we must therefore specify the way the target
		# Xt (\tilde{X}) shall be given.
		self.attachment_type = self.data_attachment[0]
		if self.attachment_type == 'measure-kernel' or \
		   self.attachment_type == 'measure-sinkhorn' :
			self.embedding_type = 'measure'
			
		elif self.attachment_type == 'varifold-kernel' or \
		   self.attachment_type == 'varifold-sinkhorn' :
			self.embedding_type = 'varifold'
			
		elif self.attachment_type == 'current-kernel' :
			self.embedding_type = 'current'
			
		#===============================================================
		# Store the original weights :
		self.transport_as_measure = False # be sure self.to_measure gives the accurate result !
		q0_mes = Q0.to_measure()
		self.initial_weights = q0_mes.weights.astype(config.floatX)
		
		self.transport_as_measure = False
		
	# Symbolic Hamiltonian functions ==========================================================================
	# The Hamiltonian dynamics is implemented by the super class "HamiltonianClouds" :
	# "Shapes" only takes care of the data attachment term.
		        
	# Appendix : Collection of data attachment terms -----------------------------------------------
	def _data_attachment(self, q1, *args) :
		"""Selects the appropriate data attachment routine, depending on self's attributes."""
		if self.attachment_type == 'measure-kernel' :
			return self._att_measure_kernel(q1, *args)
		elif self.attachment_type == 'measure-sinkhorn' :
			return self._att_measure_sinkhorn(q1, *args)
		elif self.attachment_type == 'varifold-kernel' :
			return self._att_varifold_kernel(q1, *args)
		elif self.attachment_type == 'varifold-sinkhorn' :
			return self._att_varifold_sinkhorn(q1, *args)
		else :
			raise(NotImplementedError)
	
	def to_measure(self, q) :
		raise(NotImplementedError)
		
	def to_varifold(self, q) :
		raise(NotImplementedError)
		
		
	def _att_measure_kernel(self, q1, xt_x, xt_mu) :
		q1_x, q1_mu = self.to_measure(q1)
		return Measures._kernel_matching(q1_x, q1_mu, xt_x, xt_mu, self.att_param[1])
	def _att_measure_sinkhorn(self, q1, xt_x, xt_mu) :
		q1_x, q1_mu = self.to_measure(q1)
		return Measures._sinkhorn_matching(q1_x, q1_mu, xt_x, xt_mu, self.att_param[1])
		
	def _att_varifold_kernel(self, q1, xt_x, xt_mu, xt_n) :
		q1_x, q1_mu, q1_n = self.to_varifold(q1)
		return Varifolds._kernel_matching(q1_x, q1_mu, q1_n, xt_x, xt_mu, xt_n, self.att_param[1])
	def _att_varifold_sinkhorn(self, q1, xt_x, xt_mu, xt_n) :
		q1_x, q1_mu, q1_n = self.to_varifold(q1)
		return Varifolds._sinkhorn_matching(q1_x, q1_mu, q1_n, xt_x, xt_mu, xt_n, self.att_param[0], self.att_param[1])
	
	
	
	# Bindings from a (more or less legacy) pythonic interface to the compiled theano routines ====
	def shooting_cost(self, *args, target=None) :
		"""
		Method used by the end-user.
		Given q0 and p0 as numpy arrays,
		and Xt as an appropriate object (Measure, Varifold, ...),
		We output a tuple containing the cost and its two derivatives :
		(cost, dq0_cost, dp0_cost)
		"""
		if self.embedding_type == 'measure' :
			return self.opt_shooting_cost(*args, target.points, target.weights)
		elif self.embedding_type == 'varifold' :
			return self.opt_shooting_cost(*args, target.points, target.weights, target.normals)
		else :
			raise(NotImplementedError)
	
	def grid_trajectory(self, q0, p0, ranges, nlines = 11) :
		"""
		Returns a list of Curves, each of them being a grid carried along
		a geodesic shooting.
		ranges is a list of (min, max), 
		"""
		np_per_lines = (nlines-1) * 10 + 1
		
		assert len(ranges) == self.dimension
		
		x_l = [np.linspace(min_r, max_r, nlines      ) for (min_r,max_r) in ranges]
		x_d = [np.linspace(min_r, max_r, np_per_lines) for (min_r,max_r) in ranges]
		
		s0 = []
		connectivity = []
		
		i = 0
		if self.dimension == 2 :
			for x in x_l[0] :
				s0.append([x,x_d[1][0]] )
				for y in x_d[1][1:] :
					s0.append([x,y])
					connectivity.append([i,i+1])
					i += 1
				i += 1
			for y in x_l[1] :
				s0.append([x_d[0][0],y] )
				for x in x_d[0][1:] :
					s0.append([x,y])
					connectivity.append([i,i+1])
					i += 1
				i += 1
		s0 = np.vstack(s0)
		
		st = self.hamiltonian_trajectory_carrying(q0, p0, s0)[2]
		ct = [Curve(s.ravel(), connectivity, self.dimension) for s in st]
		return ct
		
		
	
	
	# Methods used by atlas & Co. to be manifold-independent =======================================
	"""Non vectorial stuff."""
	def zero_momentum(self) :
		return np.zeros(self.array_shape)
	def zero_position(self) :
		return np.zeros(self.array_shape)
	def sum_position(self, Q0, dQ0p) :
		"""(Arithmetic) mean of a list of position updates. """
		return reduce(add, dQ0p)
	def displacement_norm(self, dQ0p) :
		"""~norm of dQ0p, used by FreeAtlas to normalize template updates."""
		return sqrt(mean(vstack([dQ0 for dQ0 in dQ0p])**2))
		
	"""The following two routines are useful for quasi-Newton optimisation."""
	def L2_product_tangent(self, q, dq1, dq2) :
		return dq1.ravel().dot(dq2.points.ravel())
	def L2_product_cotangent(self, q, p1, p2) :
		return p1.points.ravel().dot(p2.points.ravel())
		
		
	# Input-Output through VTK files ===============================================================
	def file_marker(self, q, name = '', **kwargs) :
		self.to_file(q, self.foldername + name + '.vtk')
		
	def file_marker_target(self, q, name = '', **kwargs) :
		q.to_file(self.foldername + name + '.vtk')
		
	def file_plot_grids(self, gts, name = ['grid'], **kwargs) :
		if type(gts[0]) is not list :
			gts = [gts]
		if type(name) is not list :
			name = [name]
		for gt, n in zip(gts, name) :
			for (i, g) in enumerate(gt) :
				self.file_marker_target(g, n + '_' + str(i))
				
	def file_plot_traj(self, qts, name = [''], **kwargs) :
		if type(qts[0]) is not list :
			qts = [qts]
		if type(name) is not list :
			name = [name]
		for qt, n in zip(qts, name) :
			for (i, q) in enumerate(qt) :
				self.file_marker(q, n + '_' + str(i))
				
	def file_quiver(self, q, p, name=' ', **kwargs) :
		if q.shape[1] == 2 :
			q = np.hstack((q, np.zeros((q.shape[0],1))))
		if p.shape[1] == 2 :
			p = np.hstack((p, np.zeros((p.shape[0],1))))
		pd  = PolyData(points = q)
		vec = PointData(Vectors(p, name='momentum'))
		vtk = VtkData( pd, vec )
		vtk.tofile(self.foldername + name + '.vtk', 'ascii')
		
	def interactive_plot_momentums(self, *args, **kwargs) :
		None
		
	def file_plot_momentums(self, qts, pts, name = [''], **kwargs) :
		if type(qts[0]) is not list :
			qts = [qts]
			pts = [pts]
		if type(name) is not list :
			name = [name]
		for qt, pt, n in zip(qts, pts, name) :
			for (i, (q, p)) in enumerate(zip(qt,pt)) :
				self.file_quiver(q, p, n + '_' + str(i))
		
	def file_show_transport(self, q, xt, gamma, name='plan') :
		points       = []
		connectivity = []
		curr_id      = 0
		
		q = Curve(q.ravel(), self.connectivity, self.dimension)
		
		# Extract the centers + areas
		Q  = q.to_measure()
		Xt = xt.to_measure()
		xtpoints = xt.to_array()
		for (a, mui, gi) in zip(Q.points, Q.weights, gamma) :
			# gi = sum(Q.weights) * gi / mui
			gi = gi / mui # gi[j] = fraction of the mass from "a" which goes to xtpoints[j]
			for (seg, gij) in zip(xt.connectivity, gi) :
				mass_per_line = 0.1
				if gij >= mass_per_line :
					nlines = np.floor(gij / mass_per_line)
					ts = np.linspace(.35, .65, nlines)
					for t in ts :
						b = (1-t) * xtpoints[seg[0]] + t * xtpoints[seg[1]]
						points       += [a, b]
						connectivity += [[curr_id, curr_id + 1]]
						curr_id      += 2
		# Write to file :
		Points = np.vstack(points)
		Connec = np.vstack(connectivity)
		Plan   = Curve(Points.ravel(), Connec, self.dimension)
		Plan.to_file(self.foldername + name + '.vtk')
			
	def I(self, q) :
		raise(NotImplementedError)
		
		
		
