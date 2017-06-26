# Import of the relevant tools
import time
import numpy as np
import theano
import theano.tensor as T
from   theano import pp, config

import os

from plotly.tools import FigureFactory as FF
import plotly.graph_objs as go

from ..io.read_vtk import ReadVTK
from ..data_attachment.measures  import Measures
from ..data_attachment.varifolds import Varifolds
from ..math_utils.kernels import _squared_distances, _gaussian_kernel

# a TheanoShapes manifold will be created from a regular Curve/Surface, 
# with  information about connectivity and number of points.
# Basically, a TheanoShapes is an efficient implementation of
# a shape orbit.
from .riemannian_manifold import RManifold

class TheanoHamiltonianClouds(RManifold) :
	"""
	Abstract class which implements the symbolic Hamiltonian dynamic.
	"""
	def __init__(self, kernel                = ('gaussian', 1), 
	                   weights               = (0.01, 1), # gamma_V, gamma_W
	                   dt                    = 0.1,
		               plot_interactive      = False,
		               plot_file             = True,
		               foldername            = 'results/'
		               ) :		
		
		# We are going to re-implement the RManifold shoot and backward methods,
		# so we may as well start from scratch.
		# no need for RManifold.__init__(self, npoints, dimension, kernel, dt)
		self.kernel          = kernel
		self.kernel_radius   = kernel[1]
		self.dt              = dt
		self.weight_regularization = weights[0]
		self.weight_attachment     = weights[1]
		self.current_axis = []
		self.plot_interactive = plot_interactive
		self.plot_file        = plot_file
		self.foldername       = foldername
		
		def assert_folder(dname) :
			if not os.path.exists(dname):
				os.makedirs(dname)
		assert_folder(self.foldername)
		assert_folder(self.foldername + '/Descent/')
		assert_folder(self.foldername + '/Descent/Models/')
		assert_folder(self.foldername + '/Descent/Momentums/')
		assert_folder(self.foldername + '/Descent/Plans/')
		assert_folder(self.foldername + '/Grid/')
		assert_folder(self.foldername + '/Momentums/')
		assert_folder(self.foldername + '/Shoot/')
	
	
	# We don't need all those legacy routines :    ================================================
	def K(self,q,p, kernels) :
		raise(NotImplementedError)
	def upP(self,q,p, kernels) :
		raise(NotImplementedError)
	def gradq_pKqz(self, p, q, z, kernels) :
		raise(NotImplementedError)
	def dq_gradq_pKqp_a(self, q, p, a, kernels) :
		raise(NotImplementedError)
	def dq_Kqp_a(self,q,p,a, kernels) :
		raise(NotImplementedError)
		
	# Symbolic Hamiltonian functions ==========================================================================
	
	# Part 1 : cometric on the space of landmarks, kinetic energy on the phase space (Hamiltonian)-----
	def _kq(self, q): # Computes the standard gaussian kernel matrix of variance kernel_radius
		return _gaussian_kernel(q, q, self.kernel_radius)
		
	def _Kq(self, q) :
		k = self._kq(q)
		return k  
		
	def _Hqp(self, q, p) :
		"""Hamiltonian."""
		pKqp =  self._Kq(q) * (p.dot(p.T))  
		return .5 * T.sum(pKqp)             # H(q,p) = (1/2) * sum_ij K(x_i,x_j) p_i.p_j
		
		
	# Part 2 : Geodesic shooting ------------------------------------------------------------------
	# The partial derivatives of the Hamiltonian are automatically computed !
	def _dq_Hqp(self, q,p) : 
		return T.grad(self._Hqp(q,p), q)
	def _dp_Hqp(self, q,p) :
		return T.grad(self._Hqp(q,p), p)
	
	
		
	def _hamiltonian_step(self, q,p) :        # The "math" part of the code :
		return [q + self.dt * self._dp_Hqp(q,p) ,  # Simplistic euler scheme
				p - self.dt * self._dq_Hqp(q,p) ]
				
	def _HamiltonianTrajectory(self, q, p) :
		# Here, we use the "scan" theano routine, which  can be understood as a "for" loop
		result, updates = theano.scan(fn            = lambda x,y : self._hamiltonian_step(x,y),
									  outputs_info  = [q,p],
									  n_steps       = int(np.round(1/self.dt) ))
		return result
		
	def _HamiltonianShooting(self, q, p) :
		# Here, we use the "scan" theano routine, which  can be understood as a "for" loop
		result, updates = theano.scan(fn            = lambda x,y : self._hamiltonian_step(x,y),
									  outputs_info  = [q,p],
									  n_steps       = int(np.round(1/self.dt) ))
		final_result = [result[0][-1], result[1][-1]]  # We do not store the intermediate results
		return final_result                            # and only return the final state + momentum
		
	# Part 2bis : Action on the ambiant space. -------------------------------------------------------
	# This is useful to visualize  the grid deformation, or in the control points setting
	
	def _carry(self, q, p, s, dt) :
		"""
		Defines the infinitesimal action of a momentum p located at q
		on the theano variable s.
		"""
		return s + dt * _gaussian_kernel(s, q, self.kernel_radius).dot(p)
			
	def _hamiltonian_step_carrying(self, q,p, s) :        # The "math" part of the code :
		return [q + self.dt * self._dp_Hqp(q,p) ,  # Simplistic euler scheme
				p - self.dt * self._dq_Hqp(q,p) ,
				self._carry(q, p, s, self.dt)   ]
	
	def _HamiltonianTrajectoryCarrying(self, q, p, s) :
		# Here, we use the "scan" theano routine, which  can be understood as a "for" loop
		result, updates = theano.scan(fn            = lambda x,y,z : self._hamiltonian_step_carrying(x,y,z),
									  outputs_info  = [q,p,s],
									  n_steps       = int(np.round(1/self.dt) ))
		return result
		
	def _HamiltonianShootingCarrying(self, q, p, s) :
		# Here, we use the "scan" theano routine, which  can be understood as a "for" loop
		result = self._HamiltonianTrajectoryCarrying(q, p, s)
		final_result = [result[0][-1], result[1][-1], result[2][-1]]  # We do not store the intermediate results
		return final_result                            # and only return the final state + momentum
		
		
		
	# Part 3 : Cost function and derivatives -------------------------------------------------------
	def _cost(self, q, p, *args) :
		cost_reg = self._Hqp(q,p)
		cost_att = self._data_attachment(self._HamiltonianShooting(q,p)[0], *args) # C(q_0, p_0) = A(q_1, x_t)
		return  self.weight_regularization * cost_reg + self.weight_attachment * cost_att[0], cost_att[1]
		
	# The discrete backward scheme is automatically computed :
	def _dcost_q0(self, q,p, *args) :           # Useful for template estimation
		return T.grad(self._cost(q,p,*args)[0], q) # The gradients wrt. q_0 is automatically computed
	def _dcost_p0(self, q,p,*args) :            # Useful in a matching problem
		return T.grad(self._cost(q,p,*args)[0], p) # The gradients wrt. p_0 is automatically computed
		
	def _opt_shooting_cost(self, q0, p0, *args) : # Wrapper
		cost_info = self._cost(     q0, p0, *args)
		return [cost_info[0] ,  # the actual cost
		        q0,#self._dcost_q0( q0, p0, *args) , 
		        self._dcost_p0( q0, p0, *args) ,
		        self._HamiltonianShooting(q0,p0)[0],
		        cost_info[1]]   # Additional information (transport plan, etc.)
		        
	# Appendix : Collection of data attachment terms -----------------------------------------------
	def _data_attachment(self, q1, *args) :
		"""Selects the appropriate data attachment routine, depending on self's attributes."""
		raise(NotImplementedError)
		
		
	# Input-Output =================================================================================
	def show(self, *args, **kwargs) :
		if self.plot_interactive :
			self.interactive_show(*args, **kwargs)
	def interactive_show(self) :
		raise(NotImplementedError)
		
	def marker(self, *args, **kwargs) :
		if self.plot_interactive :
			self.interactive_marker(*args, **kwargs)
		if self.plot_file :
			self.file_marker(*args, **kwargs)
	
	def marker_target(self, *args, **kwargs) :
		if self.plot_interactive :
			self.interactive_marker_target(*args, **kwargs)
		if self.plot_file :
			self.file_marker_target(*args, **kwargs)
			
	def plot_traj(self, *args, **kwargs) :
		if self.plot_interactive :
			self.interactive_plot_traj(*args, **kwargs)
		if self.plot_file :
			self.file_plot_traj(*args, **kwargs)
			
	def quiver(self, *args, **kwargs) :
		if self.plot_interactive :
			self.interactive_quiver(*args, **kwargs)
		if self.plot_file :
			self.file_quiver(*args, **kwargs)
	def plot_momentums(self, *args, **kwargs) :
		if self.plot_interactive :
			self.interactive_plot_momentums(*args, **kwargs)
		if self.plot_file :
			self.file_plot_momentums(*args, **kwargs)
	def show_transport(self, *args, **kwargs) :
		#if self.plot_interactive :
		#	self.interactive_show_transport(*args, **kwargs)
		if self.plot_file :
			self.file_show_transport(*args, **kwargs)
			
		
		
