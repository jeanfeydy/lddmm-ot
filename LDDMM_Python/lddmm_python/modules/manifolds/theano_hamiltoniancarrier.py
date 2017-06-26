# Import of the relevant tools
import time
import numpy as np
import theano
import theano.tensor as T
from   theano import pp, config


from plotly.tools import FigureFactory as FF
import plotly.graph_objs as go

from ..io.read_vtk import ReadVTK
from ..data_attachment.measures  import Measures
from ..data_attachment.varifolds import Varifolds
from ..math_utils.kernels import _squared_distances, _gaussian_kernel


from .theano_hamiltonianclouds import TheanoHamiltonianClouds

class TheanoHamiltonianCarrier(TheanoHamiltonianClouds) :
	"""
	Superseeds the regular Hamiltonian cost, which is not computed wrt to
	the final state q1, but with respect to a carried "shape" s1.
	This is the class that shall be inherited by any "control points" manifold.
	"""
	def __init__(self, **kwargs) :
		
		TheanoHamiltonianClouds.__init__(self, **kwargs)
		
		
	# Symbolic Hamiltonian functions ==========================================================================
		
	# Part 3 : Cost function and derivatives -------------------------------------------------------
	def _cost(self, q, p, s, *args) :
		cost_reg = self._Hqp(q,p)
		cost_att = self._data_attachment(self._HamiltonianShootingCarrying(q,p,s)[2], *args) # C(q_0, p_0, s_0) = A(s_1, x_t)
		return  self.weight_regularization * cost_reg + self.weight_attachment * cost_att[0], cost_att[1]
		
	# The discrete backward scheme is automatically computed :
	def _dcost_q0(self, q,p, s, *args) :           # Useful for template estimation
		return T.grad(self._cost(q,p,s, *args)[0], q) # The gradients wrt. q_0 is automatically computed
	def _dcost_p0(self, q,p,s, *args) :            # Useful in a matching problem
		return T.grad(self._cost(q,p,s, *args)[0], p) # The gradients wrt. p_0 is automatically computed
		
	def _opt_shooting_cost(self, q0, p0, s0, *args) : # Wrapper
		cost_info = self._cost(     q0, p0, s0, *args) # cost + additional information
		return [cost_info[0] , # Actual cost 
		        q0,#self._dcost_q0( q0, p0, s0, *args) , 
		        self._dcost_p0( q0, p0, s0, *args) ,
		        self._HamiltonianShootingCarrying(q0,p0,s0)[2],
		        cost_info[1] ] # Additional information (transport plan, etc.)
		        
	# Appendix : Collection of data attachment terms -----------------------------------------------
	def _data_attachment(self, s1, *args) :
		"""Selects the appropriate data attachment routine, depending on self's attributes."""
		raise(NotImplementedError)
		
		
		
