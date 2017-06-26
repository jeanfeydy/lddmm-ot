from pylab import *

from .atlas import Atlas
from ..manifolds.curves import Curve

class FreeAtlas(Atlas):
	"Atlas with a Free mean Q0 and no dimensional constraint."
	def __init__(self, *args, **kwargs):
		Atlas.__init__(self, *args, **kwargs)
	def after_step_Q0(self, dQ0) :
		return self.Q0 + dQ0  # Free templates
	def after_step_P(self, dP) :
		return [ p + dp for (p,dp) in zip(self.P, dP)]   # No dimensional constraint
	def template_cost(self, *args) :
		return 0  # Free template
	def normalize_and_regularize_template_updates(self, dQ0p, iteration) :
		dQ0p = self.normalize_template_updates(dQ0p, self.M.displacement_norm(dQ0p), iteration)
		dQ0 =  self.M.sum_position(self.Q0, dQ0p)
		
		# This whole Free template stuff is more or less a hack, 
		# as it is not a well defined problem (flat prior on an unbounded space),
		# so it's put here without too much care.
		if self.reg_template_gradient is not None :
			if self.reg_template_gradient[0] == 'gaussian' and type(self.reg_template_gradient[1]) is float :
				if self.reg_template_gradient[1] > 0 :
					if type(dQ0) is ndarray :
						None
					elif type(dQ0) is Curve :
						None
					else :
						print("I don't know how to regularize this type of gradient.")
						raise NotImplementedError
			else :
				print('This regularization type for the free template is not implemented : ', self.reg_template_gradient)
				raise NotImplementedError
		
		return dQ0
	def regularize_template_gradient(self, grad) :
		"Free template gradient regularization."
		if self.reg_template_gradient[0] == 'gaussian' and type(self.reg_template_gradient[1]) is float :
			if self.reg_template_gradient[1] > 0 :
				K = self.M.precompute_kernels(self.Q0)[0]
				if type(grad) is ndarray :
					grad_p = grad.reshape((self.M.npoints, self.M.dimension))
					return (K @ grad_p).ravel()
				elif type(grad) is Curve :
					C = self.Q0 # assume it's a curve
					M = C.to_measure()
					
					mutilde = zeros(C.array_shape()[0])
					for (i,s) in enumerate(C.connectivity) :
						mutilde[s[0]] += .5 * M.weights[i]
						mutilde[s[1]] += .5 * M.weights[i]
					
					K = K * (atleast_2d(K.dot(mutilde)).T * mutilde)
					grad.points = (K @ grad.to_array()).ravel()
					return grad
				else :
					print("I don't know how to regularize this type of gradient.")
					raise NotImplementedError
		else :
			print('This regularization type for the free template is not implemented : ', self.reg_template_gradient)
			raise NotImplementedError
				
	def show_HT(self) :
		None # nothing to show...
	def get_frame(self, f) :
		# Five plotly traces per frame : 'Frechet Mean', 'Shootings', 'Targets', 'Models', 'Transports'
		list1 = str([ 1 + 5*self.current_frame, 2 + 5*self.current_frame, 3 + 5*self.current_frame, 4 + 5*self.current_frame, 5 + 5*self.current_frame])[1:-1]
		self.current_frame = f
		list2 = str([ 1 + 5*self.current_frame, 2 + 5*self.current_frame, 3 + 5*self.current_frame, 4 + 5*self.current_frame, 5 + 5*self.current_frame])[1:-1]
		return (self.frames[f] , [list1, list2])
