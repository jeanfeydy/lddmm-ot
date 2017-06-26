# Import of the relevant tools
import time
import numpy as np
import theano
import theano.tensor as T
from   theano import pp, config

from ..math_utils.real_fft import _rfft2d, _irfft2d

from .images_carrier import ImagesCarrier
#from .images2D_manifold import Images2DManifold

class Images2DCarrier(ImagesCarrier) :
	"""
	Images2D + HamiltonianDynamics with control points.
	"""
	def __init__(self, *args, **kwargs) :
		"""
		Creates a Images2DCarrier manifold.
		Compilation takes place here.
		"""
		
		ImagesCarrier.__init__(self, *args, image_dimension = 2, **kwargs)
	
	def _linear_interp_field(self, field, points, mode = 'dense', field_dims = 2, rescaling = np.array([1., 1.], dtype=config.floatX) ) :
		if mode == 'downsampled' :
			max_coords = np.array(self.sampled_field_shape, dtype='int64')
		elif mode == 'dense' :
			max_coords = np.array(self.image_shape        , dtype='int64')
			
		if rescaling is not None :
			points = points * rescaling
		#===============================================================
		# Slight Hack, which may not be necessary depending on your version of
		# Theano. The thing is : until recently, "fancy" dynamic indexing
		# (which is necessary for bilinear interpolation) was not supported
		# on GPU for "dim > 1"-indexing.
		# Classical fix (included here just in case) consists in
		# "flattening" the image we're interpolating.
		if field_dims > 0 : # Is there a 3rd dimension, i.e. at each location, are we considering a vector ?
			field = field.dimshuffle(2,0,1)
			field = field.flatten(ndim=2)
			field = field.dimshuffle(1,0)
		elif field_dims == 0 : # or a scalar ?
			field = field.flatten(ndim=1)
			
		def indices_in_flatten_array(ndim, shape, *args): # Fix taken from stackoverflow
			"""
			We expect that all args can be broadcasted together.
			So, if we have some array A with ndim&shape as given,
			A[args] would give us a subtensor.
			We return the indices so that A[args].flatten()
			and A.flatten()[indices] are the same.
			"""
			assert ndim > 0
			assert len(args) == ndim
			indices_per_axis = [args[i] for i in range(ndim)]
			for i in range(ndim):
				for j in range(i + 1, ndim):
					indices_per_axis[i] *= shape[j]
			indices = indices_per_axis[0]
			for i in range(1, ndim):
				indices += indices_per_axis[i]
			return indices
		def flat_ind(i0, i1) :
			return indices_in_flatten_array(2, max_coords, i0, i1)
		#===============================================================
		coords = points
		
		lambdas = (coords - T.floor(coords)).T
		ind_f = (T.floor(coords)).astype('int64')
		ind_c = (ind_f + 1      ).astype('int64')
		
		ind_f = (ind_f % max_coords).T.astype('int64')
		ind_c = (ind_c % max_coords).T.astype('int64')
		
		"""
		field_00 = field[ind_f[0], ind_f[1]].T
		field_01 = field[ind_f[0], ind_c[1]].T
		field_10 = field[ind_c[0], ind_f[1]].T
		field_11 = field[ind_c[0], ind_c[1]].T
		"""
		#===============================================================
		field_00 = field[flat_ind(ind_f[0], ind_f[1])].T
		field_01 = field[flat_ind(ind_f[0], ind_c[1])].T
		field_10 = field[flat_ind(ind_c[0], ind_f[1])].T
		field_11 = field[flat_ind(ind_c[0], ind_c[1])].T
		#===============================================================
		
		# Simplistic bilinear interpolation scheme :
		res = (1-lambdas[0]) * (1-lambdas[1]) * field_00 \
			+ (1-lambdas[0]) *    lambdas[1]  * field_01 \
			+    lambdas[0]  * (1-lambdas[1]) * field_10 \
			+    lambdas[0]  *    lambdas[1]  * field_11
		return res.T
		
	def _linear_interp_downsampledfield(self, field, points) :
		"""Allows us to interpolate a downsampled velocity field to the dense image grid."""
		resc = (np.array(self.sampled_field_shape) / np.array(self.image_shape)).astype(config.floatX)
		return self._linear_interp_field(field, points, mode = 'downsampled', rescaling = resc)
		
	def _phase_shifts(self, point) :
		"""
		Returns the phase shifts corresponding to a dirac at position 'point'.
		As Theano only implements real-valued FFT on the GPU, we will actually need
		more than the "complex" phase i*(phase_0 + phase_1).
		"""
		phase_0 = -(self.pulsations[0] * point[0]).dimshuffle(0, 'x')
		phase_1 = -(self.pulsations[1] * point[1]).dimshuffle('x', 0)
		return [(phase_0 + phase_1), (phase_0 - phase_1)]
	
	def _dirac_truncated_rfft(self, point) :
		"""
		Returns the truncated real FFT of a dirac at position 'point',
		as a (2+1)-d array of size "K.shape//2+1" + (4,),.
		See real_fft._irfft_2d to understand the format of the output.
		The code may seem quite circonvoluted but hey, it's not my fault
		if theano forces us to use real-valued FFT...
		"""
		su, di = self._phase_shifts(point)
		re_re = T.cos(di) + T.cos(su) # 2 cos(a)cos(b) = cos(a-b) + cos(a+b)
		re_im = T.sin(su) + T.sin(di) # 2 sin(a)cos(b) = sin(a+b) + sin(a-b)
		im_re = T.sin(su) - T.sin(di) # 2 cos(a)sin(b) = sin(a+b) - sin(a-b)
		im_im = T.cos(di) - T.cos(su) # 2 sin(a)sin(b) = cos(a-b) - cos(a+b)
		return .5 * T.stack([re_re, re_im, im_re, im_im], axis=2) # Don't forget the .5 !
	
	def _downsampled_velocity_field(self, q, p) :    
		"""
		Input : control points positions 'q' and momentums 'p'
		given as theano matrices of size n-by-d, where n is the number
		of control points and d the dimension of the ambient space.
		The point of this routine is to return the LDDMM global vector field
		v = K_q . p in the most efficient way.
		In this implementation, we try to leverage a simple observation :
		the kernel K is a lowpass filter, and as a result, the infinitesimal
		deformation v*dt is extremely smooth.
		It could therefore be computed on a "sparse" grid, and linearly interpolated
		without too much error.
		"""     
		# First, accumulate the Fourier transform of the field...
		def add_control_point(point, momentum, f_v0, f_v1) :
			"""
			Acts on the FFT on the vector field components
			f_v0 and f_v1, by representing the addition of
			a control point at position 'point' with momentum 'momentum'.
			Note that the f_vi are not smoothed yet, so that it would be
			cleaner to call them "f_pi" : they are the fourier transforms
			of atomic vector-valuled measures.
			"""
			rfft_point = self._dirac_truncated_rfft(point)
			f_v0 += momentum[0] * rfft_point
			f_v1 += momentum[1] * rfft_point
			return [f_v0, f_v1]
			
			
		# First, we build the real-valued FFT of the momentum,
		# seen as a vector-valued sum of diracs :
		null_freq = np.zeros(self.f_K.shape + (4,), dtype=config.floatX)
		results, _ = theano.scan(fn           = add_control_point,
								 outputs_info = [null_freq, null_freq], # Start with two null FFTs, one for each component.
								 sequences    = [q, p])                 # We'll iterate on the control points.
		# We can then get the velocity field by convolution with the LDDMM kernel,
		# i.e. multiplication in the Fourier domain.
		f_K_full = T.as_tensor_variable(self.f_K) # Array of same size as the image//2 +1. (real-valued FFT and all that...)
		f_K_full = f_K_full.dimshuffle(0,1,'x')   # Make f_K broadcastable in the last dimension (i.e. allow it to multiply each slice of the real fft).
		# Convolution with the LDDMM kernel :
		f_v0, f_v1 = (f_K_full * results[0][-1],  
					  f_K_full * results[1][-1])
		# Finally, apply the inverse Fourier transform...
		v0 = _irfft2d(f_v0)
		v1 = _irfft2d(f_v1)
		# And concatenate the components :
		return T.stack([v0, v1], axis = 2)
		
	def _semi_lagrangian_displacement(self, v_sampled, grid_points, dt) :
		"""
		Semi-Lagrangian scheme.
		Given a downsampled velocity field v (which  will be linearly interpolated),
		we find "where the information came from", i.e. numerically invert its
		flow during a time-step dt on the 'grid_points'.
		To do so, we simply solve the fixed point equation
		a(y)/2 = (dt/2) * v( y - a(y)/2 )
		by an "Picard-like" iterative scheme,
		where y is a grid point, and -a(y) the corresponding "backward" vector.
		"""
		def f(r) :
			return .5 * dt * self._linear_interp_downsampledfield(v_sampled, grid_points - r)
		# Theano on GPU requires float32, i.e. explicit downcast from numpy float64 type :
		r_0 = np.zeros((np.prod(self.image_shape), self.image_dimension), dtype = config.floatX) 
		result, updates = theano.scan(fn            = f,          # Iterated routine
									  outputs_info  = [r_0],      # Starting estimate for r
									  n_steps       = 5)          # Number of iterations, sufficient in practice
		r_inf = result[-1]  # We only keep the end result
		return 2. * r_inf   # displacement "alpha"
	
	def dense_grid(self) :
		"""
		Outputs the dense image 'meshgrid'.
		"""
		x = np.arange(self.image_shape[1])
		y = np.arange(self.image_shape[0])
		X =  np.ones((self.image_shape[0], 1)) * x
		Y = (np.ones((self.image_shape[1], 1)) * y).T
		return np.stack([Y, X], axis=2).astype(theano.config.floatX) # np.ones = float64 -> float32, the only type supported by Theano on GPU
		
	def unfolded_dense_grid(self) :
		"""
		List of points on the dense image grid.
		"""
		g = self.dense_grid()
		Y = g[:,:,0]
		X = g[:,:,1]
		return (np.vstack( (Y.ravel(), X.ravel()) )).T
		
	def _carry(self, q, p, phi_inv, dt) :
		"""
		Defines the infinitesimal action of a momentum p located at q
		on the theano variable phi_inv.
		phi_inv(y) represents the displacement field such that
		phi^{v_t}_{t -> 0} (y) = y + phi_inv(y)   -- in the notation of the 2005 Beg et Al. paper.
		This routine is about going from
		phi^{v_t}_{t -> 0} to phi^{v_t}_{t + dt -> 0},
		i.e. from 
		phi_inv^t to phi_inv^{t+dt}.
		
		As the grid is fixed, note that phi_inv is simply encoded as an array  of shape
		"img_size + (img_dim,)".
		"""
		# Note : wrt Theano, those two arrays are "numeric constants" :
		dense_grid          = self.dense_grid()           
		unfolded_dense_grid = self.unfolded_dense_grid()
		
		v         = self._downsampled_velocity_field(q, p)  # (downsampled) velocity vector field on the ambient space, v = K_q p.
		a         = self._semi_lagrangian_displacement(v, unfolded_dense_grid, dt) # invert its flow on the dense grid, for a timestep dt.
		n_phi_inv = self._linear_interp_field(phi_inv, unfolded_dense_grid - a, mode = 'dense') - a
		res       = n_phi_inv.reshape(self.image_shape + (2,)) # reshape it.
		return res
		
	def _image_circ_diffeo(self, I0, id_plus_phi_inv) :
		"""
		Given an image I0 and a diffeomorphism phi^{-1} encoded as a vector field id_plus_phi_inv such that
		phi^{-1} (y) = y + phi_inv(y) = id_plus_phi_inv(y),
		outputs the transported image
		phi . I0 = I0 \circ phi^{-1}
		"""
		InvPhi_list = id_plus_phi_inv.reshape(self.unfolded_dense_grid().shape)
		return self._linear_interp_field(I0, InvPhi_list, mode = 'dense', field_dims=self.nimage_fields).reshape(self.image_shape)
		
	def _hamiltonian_step_carrying(self, q,p, It, phi_inv, I0) : 
		"""
		Output updated values for the control points/momentums q and p,
		as well as the carried image It = phi_{0->t} . I0 = I0 \circ phi_{t->0} and
		the backward displacement field phi_inv.
		"""
		n_phi_inv = self._carry(q, p, phi_inv, self.dt) # update the displacement field n_phi_inv
		return [q + self.dt * self._dp_Hqp(q,p), # In the classical LDDMM + control points framework,
				p - self.dt * self._dq_Hqp(q,p), # the dynamic is not affected by the transported data.
				self._image_circ_diffeo(I0, self.dense_grid() + n_phi_inv), # Interpolate I0 \circ (Id + n_phi_inv)
				n_phi_inv]
	def _hamiltonian_step_carrying2(self, q,p, phi_inv) : 
		"""
		Same as _hamiltonian_step_carrying, but don't bother computing I0 \circ (Id + n_phi_inv).
		"""
		return [q + self.dt * self._dp_Hqp(q,p),
				p - self.dt * self._dq_Hqp(q,p),
				self._carry(q, p, phi_inv, self.dt)]
	
	def _HamiltonianTrajectoryCarrying(self, q, p, i0) :
		"""
		Given initial control points/momentums q0 and p0 given as n-by-d matrices,
		and a "template" image i0, outputs the trajectories q_t, p_t, I_t = I0 \circ phi_{t->0}.
		"""
		identity = T.as_tensor_variable(0. * self.dense_grid()) # We encode the identity as a null displacement field.
		
		# Here, we use the "scan" theano routine, which  can be understood as a "for" loop
		result, updates = theano.scan(fn            = lambda v,w,x,y,z : self._hamiltonian_step_carrying(v,w,x,y,z),
									  outputs_info  = [q,p,i0, identity],
									  non_sequences = i0,
									  n_steps       = int(np.round(1/self.dt) ))
		return result[0:3] # We keep phi_inv private
		
	def _HamiltonianShootingCarrying(self, q, p, i0) :
		"""
		Given initial control points/momentums q0 and p0 given as n-by-d matrices,
		and a "template" image i0, outputs the trajectories q_t, p_t, I_t = I0 \circ phi_{t->0}.
		"""
		# Here, we use the "scan" theano routine, which  can be understood as a "for" loop
		identity = T.as_tensor_variable(0. * self.dense_grid())  # We encode the identity as a null displacement field.
		
		# Here, we use the "scan" theano routine, which  can be understood as a "for" loop
		result, updates = theano.scan(fn            = lambda x,y,z : self._hamiltonian_step_carrying2(x,y,z),
									  outputs_info  = [q,p, identity],
									  n_steps       = int(np.round(1/self.dt) ))
		
		phi_inv_1 = result[2][-1]  # We do not store the intermediate results
		I1 = self._image_circ_diffeo(i0, self.dense_grid() + phi_inv_1)  # instead of interpolating the images It at all timesteps, we only do it in the end.
		return [result[0][-1], result[1][-1], I1]                       # and only return the final state + momentum + image


