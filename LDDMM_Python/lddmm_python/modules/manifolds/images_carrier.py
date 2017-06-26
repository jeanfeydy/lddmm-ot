# Import of the relevant tools
import time
import numpy as np
import theano
import theano.tensor as T
from   theano import pp, config

from numpy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq

from .theano_hamiltoniancarrier import TheanoHamiltonianCarrier
from .images_manifold           import ImagesManifold



class ImagesCarrier(ImagesManifold, TheanoHamiltonianCarrier) :
	"""
	Combines the control points framework with the data attachment + io methods
	of the ImagesManifold class.
	"""
	
	def __init__(self, i0, 
	                   kernel                = ('gaussian', 1), 
	                   data_attachment       = ('L2', []), 
	                   weights               = (0.01, 1), # gamma_V, gamma_W
	                   dt                    = 0.1,
		               compatibility_compile = False,
		               plot_interactive      = False,
		               plot_file             = True,
		               foldername            = 'results/',
		               image_dimension       = 2
		               ) :
		"""
		Creates a Images2D/3DCarrier manifold.
		Compilation takes place here.
		"""
		
		TheanoHamiltonianCarrier.__init__(self, kernel           = kernel,
		                                        weights          = weights,
		                                        dt               = dt,
		                                        plot_interactive = plot_interactive,
		                                        plot_file        = plot_file,
		                                        foldername       = foldername)
		ImagesManifold.__init__(self,           i0,
		                                        data_attachment,
		                                        image_dimension )
		
		# ==============================================================
		# Build the filters in advance
		sigma  = kernel[1] # Spatial kernel size
		
		# Sum-of-gaussians are implemented, and used through sigma = list of (weights, radius) couples.
		if type(sigma) is not list :
			sigma = [(1., sigma)]
		
		smallest_radius = np.min( [ x[1] for x in sigma ] )
		# Approximation used :
		# exp( - x^2/ (2 * s^2) ) = 0
		# if x > 4 * s.
		N      = np.minimum( (4 * np.array(self.image_shape)) // (2 * np.pi * smallest_radius), self.image_shape).astype(int)
		print('We use a kernel of size ', sigma, ' on an image of shape ', self.image_shape, '.')
		print('The cutoff frequency will be 2*pi*', N, '/', self.image_shape, '.')
		
		# Pre-compute the pulsations vector, which is to be used to explicitely
		# give the truncated FFT of a dirac which does not lie on the grid.
		pulsations = []
		for d in range(self.image_dimension) :
			l = np.hstack( ([0], 2*np.pi * np.linspace(-N[d], N[d], 2*N[d]+1) / self.image_shape[d]))
			pulsations.append( ifftshift(l) )
		print('We use frequency filters of size ', [len(i) for i in pulsations], '.')    
		
		puls_norm = np.atleast_2d(pulsations[0]**2).T + pulsations[1]**2
		
		# Implementing the sum of gaussians kernel :
		f_Ks = []
		for (weight, radius) in sigma :
			f_K_r = weight * np.exp(- radius**2 * puls_norm / 2)
			f_K_r   *= np.sqrt(2 * np.pi * radius**2) ** (self.image_dimension)
			f_Ks.append(f_K_r)
		f_K = np.sum( np.stack( f_Ks , axis=-1 ), axis=-1 )
		
		# Kill the  cutoff frequency. This may only work on even images...
		f_K[f_K.shape[0]//2,:] = 0
		f_K[:,f_K.shape[0]//2] = 0
		renorm = np.prod(f_K.shape) / np.prod(self.image_shape)
		f_K   *= renorm
		
		self.sampled_field_shape = f_K.shape
		
		#---------------------------------------------------------------
		# Adjustements needed because of theano's real-valued FFT
		# First, we have to truncate the spectral domain
		# to fit into the real-valued inverse fft :
		nfreqs          = f_K.shape
		self.pulsations = [puls[:(nfreq//2 + 1)].astype(config.floatX) for (puls, nfreq) in zip(pulsations, nfreqs)]
		self.f_K        = f_K[:(nfreqs[0]//2 + 1), :(nfreqs[1]//2 + 1)].astype(config.floatX)
		
		
		#===============================================================
		# Before compiling, we assign types to the teano variables
		q0    = T.matrix('q0')
		p0    = T.matrix('p0')
		if self.image_dimension == 2 :
			phi_inv_0 = T.tensor3('phi_inv_0')
			I0        = T.matrix('I0')
			I1        = T.matrix('I1')
		
		identity = self.dense_grid()
		print(identity.dtype)
		# Compilation. Depending on settings specified in the ~/.theanorc file or explicitely given
		# at execution time through environment variables, this will produce CPU or GPU code.
		
		if not compatibility_compile : # With theano, it's better to let the compilator handle the whole forward-backward pipeline
			print('Compiling the shooting_cost routine...')
			time1 = time.time()
			
			self.opt_shooting_cost = theano.function([q0, p0, I0, I1],      # input
								   self._opt_shooting_cost(q0, p0, I0, I1), # output
								   allow_input_downcast=True)           # GPU = float32 only, whereas numpy uses
																		# float64 : we allow silent conversion
			time2 = time.time()   
			print('Compiled in : ', '{0:.2f}'.format(time2 - time1), 's')
			
			
			# The hamiltonian_trajectory routine, that shall be used in the visualization
			print('Compiling the hamiltonian_trajectory visualization routine...')
			time1 = time.time()
			self.hamiltonian_trajectory = theano.function([q0,p0,I0],                                   # input
												  self._HamiltonianTrajectoryCarrying(q0, p0, I0),      # output
												  allow_input_downcast=True)         # GPU = float32 only, whereas numpy uses
																					 # float64 : we allow silent conversion
			time2 = time.time()   
			print('Compiled in : ', '{0:.2f}'.format(time2 - time1), 's')
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
