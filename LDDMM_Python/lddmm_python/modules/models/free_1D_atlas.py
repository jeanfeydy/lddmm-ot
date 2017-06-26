from pylab import *

from .free_atlas import FreeAtlas

class Free1DAtlas(FreeAtlas) :
	"""Atlas with a free mean Q0 and rank-1 constraint on P0"""
	def __init__(self, *args, **kwargs) :
		FreeAtlas.__init__(self, *args, **kwargs)
		
		# the direction, a single momentum; we choose to constrain its norm
		self.E = self.M.unit_p(self.Q0, array([1,0])) # arbitrary choice.
		self.L = zeros((self.nobs,1)) # coefficients, 1xnobs real-valued matrix
		self.P = self.L @ atleast_2d(self.E) # it is more pythonic to work with the "transposes"
		
		# true (default) if we use the kernel matrix K
		# at the template to compute the scalar
		# products between momentums; this is the
		# natural choice, but it can be costly in
		# practice for shape manifolds.
		#
		# False to use a simple L2 metric.
		self.intrinsic_scalar_product = False
	
	"""Updates"""
	def update_P(self, dP) :
		# models update
		self.normalize_basis() # just in case the user is toying with the scalar product
		(scals, residuals) = self.decompose(dP)
		assert(self.L.shape == scals.shape)
		assert(self.E.shape == (residuals.shape[1],))
		self.L = self.L + scals
		
		# scals = scals ./ (abs(scals) + eps)
		# residuals = self.M.unit_p(self.Q0, residuals)
		self.E = self.E + mean(residuals, 0) # !!!
		
		self.normalize_basis() # Don't forget to stay in the Stiefel manifold !
		self.P = self.L @ atleast_2d(self.E)
	
	def normalize_basis(self) :
		# This is the simplest way  to do it; in practice,
		# we may use more accurate methods :
		if self.intrinsic_scalar_product :
			self.E = self.M.unit_p(self.Q0, self.E)
		else :
			self.E = self.E / sqrt(self.E @ self.E) # just in case the user is toying with the scalar product
	
	def decompose(self, dP) :
		"""
		Decomposes the update dP in a :
		 - longitudinal part along self.E, given by the coeff. update 'scals' 
		 - orthogonal part, given by 'residuals'
		self.E is assumed to be an orthonormal family
		"""
		if self.intrinsic_scalar_product :
			scals = self.M.K( self.Q0, dP ) @ self.E
		else :
			scals = dP @ self.E
		scals = atleast_2d(scals).T
		residuals = dP - scals @ atleast_2d(self.E) # normal component
		# rescaling depending on the distance from model to the template :
		residuals = residuals * (self.weights())
		# scals = scals ./ max(.2, abs(scals));
		# scals = .01 * scals ./ abs(scals);
		residuals = atleast_2d(residuals)
		return (scals, residuals)
		
	def weights(self) :
		"""
		Returns the weight associated to each point in the steering of
		the direction self.E. Just like with Frechet means, the choice
		of this function will change the behaviour of our model, from
		'angular median' to 'angular mean'.
		Note that the correct quadratic gradient-based formula is '1 / L'.
		"""
		return sign(self.L) #1/(self.L + 0.00001) 
	"""
	def show_inner_model(self, Xt, ax) :
		[points, Q, C, dQ] = show_inner_model@HypertemplateAtlas(Mod, Xt, ax);
		[dQ0p, dP] = self.training_step(Xt, 1, true); % simulation = true : there won't be any update !
		[scals, residuals] = self.decompose(dP);
		
		longitudinals = self.E * scals;
		% For ease of explanation as a steering process :
		normals = residuals.* sign(repmat(self.L, [size(residuals, 1), 1])); 
		if self.intrinsic_scalar_product
			dP = self.M.L2_repr_p(self.Q0, dP);
			longitudinals = self.M.L2_repr_p(self.Q0, longitudinals);
			normals = self.M.L2_repr_p(self.Q0, residuals);
		end
		
		basis = self.M.L2_repr_p(self.Q0, self.E);
		hold(ax, 'on');
		%plot(ax, self.L, self.L, '-x');
		quiver(ax, 0, 0, basis(2,:), basis(1, :), 0);
		quiver(ax, points(2,:), points(1,:), longitudinals(2,:), longitudinals(1,:), 1, 'y');
		quiver(ax, points(2,:), points(1,:), normals(2,:), normals(1,:), 1, 'g');
		%quiver(ax, points(2,:), points(1,:), dQ0p(2,:), dQ0p(1,:), 1, 'r');
		quiver(ax, points(2,:), points(1,:), dP(2,:), dP(1,:), 1, 'r');
		quiver(ax, 0, 0, mean(dQ0p(2,:)), mean(dQ0p(1,:)), 1, 'Color', [0,0,.5], 'Linewidth', 2);
		hold(ax, 'off');
		
	"""

