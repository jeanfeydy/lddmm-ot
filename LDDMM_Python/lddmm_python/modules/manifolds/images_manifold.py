
import numpy as np
import theano
import theano.tensor as T
from   theano import pp, config

from pyvtk import VtkData, PolyData, Vectors, PointData, Scalars # Will be useful to export the control points
import scipy.misc
#from .images import Image



class ImagesManifold :
	"""
	Class which handles all the image-specific data attachment methods
	+ the file io methods (marker, plot_traj, ...).
	It should be inherited by any manifold class which handles dense data.
	"""
	def __init__(self, i0, 
	                   data_attachment       = ('L2', []),
	                   image_dimension       = 2
		               ) :
		
		self.image_dimension = image_dimension
		excess_dims = len(i0.shape) - image_dimension
		if excess_dims == 0 : # real-valued image
			print("We're using simple images.")
			self.image_shape     = i0.shape
			self.nimage_fields   = 0
		elif excess_dims == 1 : # proba measures...
			self.image_shape     = i0.shape[:-1]
			self.nimage_fields   = i0.shape[-1]
			print("We're using vector-valued images.")
		else :
			raise(NotImplementedError)
		
		self.data_attachment = data_attachment
		self.attachment_type = data_attachment[0]
		self.att_param       = data_attachment[1]
		
		
		
		
	# Symbolic Hamiltonian functions ==========================================================================
	# The Hamiltonian dynamics is implemented by the super class "HamiltonianClouds" :
	# "Shapes" only takes care of the data attachment term.
		        
	# Appendix : Collection of data attachment terms -----------------------------------------------
	def _data_attachment(self, s1, *args) :
		"""Selects the appropriate data attachment routine, depending on self's attributes."""
		if self.attachment_type == 'L2' :
			return self._att_L2(s1, *args)
		else :
			raise(NotImplementedError)
	
	def to_measure(self, q) :
		raise(NotImplementedError)
		
	def _att_L2(self, s1, xt) :
		return .5 * T.sum( (s1-xt)**2 )
	
	# Bindings from a (more or less legacy) pythonic interface to the compiled theano routines ====
	def shooting_cost(self, *args, target=None) :
		"""
		Method used by the end-user.
		Given q0 and p0 as numpy arrays,
		and Xt as an array,
		We output a tuple containing the cost and its two derivatives :
		(cost, dq0_cost, dp0_cost)
		"""
		return self.opt_shooting_cost(*args, target)
	
	
	# Methods used by atlas & Co. to be manifold-independent =======================================
	"""Non vectorial stuff."""
	def zero_momentum(self) :
		raise(NotImplementedError)
	def zero_position(self) :
		raise(NotImplementedError)
	def sum_position(self, Q0, dQ0p) :
		"""(Arithmetic) mean of a list of position updates. """
		raise(NotImplementedError)
	def displacement_norm(self, dQ0p) :
		"""~norm of dQ0p, used by FreeAtlas to normalize template updates."""
		raise(NotImplementedError)
		
	"""The following two routines are useful for quasi-Newton optimisation."""
	def L2_product_tangent(self, q, dq1, dq2) :
		raise(NotImplementedError)
	def L2_product_cotangent(self, q, p1, p2) :
		raise(NotImplementedError)
		
		
	# Input-Output through VTK files ===============================================================
	def to_file(self, im, fname) :
		scipy.misc.imsave(fname, np.kron(im[::-1,:], np.ones((16,16))) )
	
	def file_marker(self, q, name = '', **kwargs) :
		self.to_file(q, self.foldername + name + '.png')
		
	def file_marker_target(self, q, name = '', **kwargs) :
		self.to_file(q, self.foldername + name + '.png')
		
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
		
	def file_show_transport(self, *args, **kwargs) :
		raise(NotImplementedError)
			
	def I(self, q) :
		raise(NotImplementedError)
		
		
		
