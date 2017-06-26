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

# a TheanoShapes manifold will be created from a regular Curve/Surface, 
# with  information about connectivity and number of points.
# Basically, a TheanoShapes is an efficient implementation of
# a shape orbit.
from .theano_hamiltonianclouds import TheanoHamiltonianClouds
from .shapes_manifold          import ShapesManifold

class TheanoShapes(ShapesManifold, TheanoHamiltonianClouds) :
	"""
	Combines the Hamiltonian dynamics with the data attachment + io methods
	of the ShapesManifold class.
	This is the class which should be used to handle curves and surfaces
	with full momentum.
	"""
	def __init__(self, Q0, 
	                   kernel                = ('gaussian', 1), 
	                   data_attachment       = ('measure-kernel', ('gaussian', 1)), 
	                   weights               = (0.01, 1), # gamma_V, gamma_W
	                   dt                    = 0.1,
		               compatibility_compile = False,
		               plot_interactive      = False,
		               plot_file             = True,
		               foldername            = 'results/'
		               ) :
		"""
		Creates a TheanoCurves/Surfaces manifold.
		Compilation takes place here.
		"""
		
		TheanoHamiltonianClouds.__init__(self, kernel           = kernel,
		                                       weights          = weights,
		                                       dt               = dt,
		                                       plot_interactive = plot_interactive,
		                                       plot_file        = plot_file,
		                                       foldername       = foldername)
		ShapesManifold.__init__(self,          Q0,
		                                       data_attachment)
		
		
		#===============================================================
		# Before compiling, we assign types to the teano variables
		q0    = T.matrix('q0')
		p0    = T.matrix('p0')
		s0    = T.matrix('s0')
		xt_x  = T.matrix('xt_x')
		xt_mu = T.vector('xt_mu')
		xt_n  = T.matrix('xt_n')
		
		# Compilation. Depending on settings specified in the ~/.theanorc file or explicitely given
		# at execution time, this will produce CPU or GPU code.
		
		
		if not compatibility_compile : # With theano, it's better to let the compilator handle the whole forward-backward pipeline
			print('Compiling the shooting_cost routine...')
			time1 = time.time()
			
			if self.embedding_type == 'measure' :
				self.opt_shooting_cost = theano.function([q0, p0, xt_x, xt_mu],      # input
									   self._opt_shooting_cost(q0, p0, xt_x, xt_mu), # output
									   allow_input_downcast=True)           # GPU = float32 only, whereas numpy uses
																			# float64 : we allow silent conversion
			elif self.embedding_type == 'varifold' :
				self.opt_shooting_cost = theano.function([q0, p0, xt_x, xt_mu, xt_n],      # input
									   self._opt_shooting_cost(q0, p0, xt_x, xt_mu, xt_n), # output
									   allow_input_downcast=True)           # GPU = float32 only, whereas numpy uses
																			# float64 : we allow silent conversion
				
				
			time2 = time.time()   
			print('Compiled in : ', '{0:.2f}'.format(time2 - time1), 's')
			
			# First, the hamiltonian_trajectory routine, that shall be used in the visualization
			print('Compiling the hamiltonian_trajectory visualization routine...')
			time1 = time.time()
			self.opt_hamiltonian_trajectory = theano.function([q0,p0],                     # input
												  self._HamiltonianTrajectory(q0, p0),      # output
												  allow_input_downcast=True)         # GPU = float32 only, whereas numpy uses
																					 # float64 : we allow silent conversion
			time2 = time.time()   
			print('Compiled in : ', '{0:.2f}'.format(time2 - time1), 's')
			
			
			# The hamiltonian_trajectory routine, that shall be used in the visualization of the grid
			print('Compiling the hamiltonian_trajectory_carrying visualization routine...')
			time1 = time.time()
			self.opt_hamiltonian_trajectory_carrying = theano.function([q0,p0,s0],                     # input
												  self._HamiltonianTrajectoryCarrying(q0, p0, s0),      # output
												  allow_input_downcast=True)         # GPU = float32 only, whereas numpy uses
																					 # float64 : we allow silent conversion
			time2 = time.time()   
			print('Compiled in : ', '{0:.2f}'.format(time2 - time1), 's')
	
	def hamiltonian_trajectory(self, q0,p0) :
		[qt, pt] = self.opt_hamiltonian_trajectory(q0,p0)
		return [np.append([q0], qt, axis=0), np.append([p0], pt, axis=0)]
	def hamiltonian_trajectory_carrying(self, q0,p0,s0) :
		[qt, pt, st] = self.opt_hamiltonian_trajectory_carrying(q0,p0,s0)
		return [np.append([q0], qt, axis=0), np.append([p0], pt, axis=0), np.append([s0], st, axis=0)]
