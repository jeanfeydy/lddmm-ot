from collections import namedtuple
from pylab import *

from .model import Model
#LBGFS_iHessian = namedtuple('LBGFS_iHessian', 'prev_updates, last_gradient')


class AtlasVariables :
	def __init__(self, Q0, P) :
		self.Q0 = Q0
		self.P  = P
		
	def __add__(self, dv) :
		return AtlasVariables(self.Q0 + dv.Q0, [p + vp for (p,vp) in zip(self.P, dv.P)])
	def __sub__(self, dv) :
		return AtlasVariables(self.Q0 - dv.Q0, [p - vp for (p,vp) in zip(self.P, dv.P)])
	def __rmul__(self, dt) :
		return AtlasVariables(dt * self.Q0, [dt * p for p in self.P])
	def __neg__(self) :
		return AtlasVariables(-self.Q0, [- p for p in self.P])

class ScalarAtlasVariables :
	def __init__(self, Q0, P) :
		self.Q0 = Q0
		self.P  = P
		
	def __add__(self, dv) :
		return ScalarAtlasVariables(self.Q0 + dv.Q0, [p + vp for (p,vp) in zip(self.P, dv.P)])
	def __sub__(self, dv) :
		return ScalarAtlasVariables(self.Q0 - dv.Q0, [p - vp for (p,vp) in zip(self.P, dv.P)])
	def __mul__(self, var) :
		if isinstance(var, ScalarAtlasVariables) :
			return ScalarAtlasVariables(self.Q0 * var.Q0, [p * varp for (p,varp) in zip(self.P, var.P)])
		elif isinstance(var, AtlasVariables) :
			return AtlasVariables(self.Q0 * var.Q0, [p * varp for (p,varp) in zip(self.P, var.P)])
	def __neg__(self) :
		return ScalarAtlasVariables(-self.Q0, [- p for p in self.P])
	def __truediv__(self, v) :
		return ScalarAtlasVariables(self.Q0 / v.Q0, [p / vp for (p,vp) in zip(self.P, v.P)])
	def inverse(self) :
		return  ScalarAtlasVariables(1/self.Q0, [1/p for p in self.P])


class Atlas(Model):
	"""Generic abstract atlas"""
	
	def __init__(self, Manifold, DataAttachment, nobs = 0, FrechetExponent = 2, weights = (0,0.001,1), Q0 = None, reg_template_gradient = None) :
		
		Model.__init__(self, Manifold, DataAttachment, nobs)
		self.Q0 = Q0
		self.P = [self.M.zero_momentum()] * self.nobs
		self.FrechetExponent = FrechetExponent
		(gamma_V0, gamma_V, gamma_W) = weights
		self.gamma_V0 = gamma_V0
		self.gamma_V  = gamma_V
		self.gamma_W  = gamma_W
		
		self.reg_template_gradient = reg_template_gradient
		
	def current_state(self) :
		return AtlasVariables(self.Q0, self.P)
		
	def d_regularization_term(self, q, p, prec_q) :
		"Differential of the term associated to gamma_V :"
		if self.FrechetExponent == 1 :
			# sqrt( (p, Kq p) )
			dp =   self.M.K(q, p, prec_q)
			dq = - self.M.upP(q,p, prec_q) # upP = -.5 d_q (p,Kq p)
			dp = dp / (self.M.norm_p(q, p, prec_q) + 0.000000001) #!!!
			dq = dq / (self.M.norm_p(q, p, prec_q) + 0.000000001) #!!!
		elif self.FrechetExponent == 2 :
			# .5 * (p, Kq p)
			c  =   self.M.momentum_quadratic_cost(q, p, prec_q)
			dp =   self.M.K(q, p, prec_q)
			dq = - self.M.upP(q,p, prec_q) # upP = -.5 d_q (p,Kq p)
		return (c, dq, dp)
		
	def shooting_gradient(self, state, Xt, iteration, progress) :
		"""
		N.B. : the returned value for dQ0 and dP are the 'unconstrained' ones; 
		       the actual update steps may differ depending on the underlying model !
		"""
		dQ0p = [self.M.zero_position()] * self.nobs # dQ0p[n] : how the n-th observation would like to move the model
		dP   = [self.M.zero_momentum()] * self.nobs # dP[n]   : how the n-th observation would like to change the n-th shooting parameter
		Q    = [self.M.zero_position()] * self.nobs # Q[n]    : the n-th model, stored for visualization purposes
		C  =   zeros( self.nobs )                   # C[n]    : the cost to map the n-th observation
		qts = []
		transports = []
		for n in range(self.nobs) : # one could use "parfor" if needed, but on small examples, it's a waste of time...
			(q1, qt, pt, vt, prec_qt) = self.M.shoot(state.Q0, state.P[n]) # Shooting from the template to the n-th model using the momentum self.P[n]
			out = self.C(q1, Xt[n], progress ) # compute how the n-th observation Xt[n] would like the n-th model q1 to change
			cdata = out[0]
			dq1   = out[1] 
			dq1 = dq1.ravel() # make sure dq1 is a vector - numpy is sometimes tricky to work with
			Q[n] = q1         # Store the n-th model for visulalization purposes
			qts.append(qt)    # Store the n-th shooting trajectory for visulalization purposes
			
			if len(out) > 2 :
				transports.append((out[2], q1, Xt[n])) # Store the transport plan, if any
			# backward scheme to transform the desired update 'dq1' into
			# a desired update in the variables dq0 (start point, i.e. the template self.Q0)
			#                                   dp0 (shooting momentum, i.e. self.P[n])
			# As the data attachment term doesn't care about p(t=1), dp1 = 0 :
			(dq0, dp0) = self.M.backward_scheme(dq1, self.M.zero_momentum(), qt, pt, prec_qt)
			(creg, dqreg, dpreg) = self.d_regularization_term(state.Q0, state.P[n], prec_qt[0])
			# The eventual 'desires' (i.e. updates modulo constraints encoded in the self.update method)
			# are combinations of :
			#     - the data attachment term dq0/dp0
			#     - the regularization term dqreg/dpreg, which typically enforces small deformation norms
			dq0 = (+ self.gamma_V * dqreg + self.gamma_W * dq0) / self.nobs
			dp0 = (+ self.gamma_V * dpreg + self.gamma_W * dp0) / self.nobs
			
			# Store the n-th cost :
			C[n] = (+ self.gamma_V * creg + self.gamma_W * cdata) /self.nobs
			
			dQ0p[n] = dq0
			dP[n]   = dp0 
		
		if self.FrechetExponent == 1 : # not so sure...
			raise(NotImplementedError)
			dQ0p = [ dq0 / self.M.norm(state.Q0, dq0) for dq0 in dQ0p ]
		elif self.FrechetExponent != 2 :
			raise(NotImplementedError)
		

		"""	
		if 0 : # Hack to allow the momentums to actually converge in finite time...
			dP = 0.5*dP /  minimum(.5, self.M.norm_p(self.Q0, dP)).reshape(self.nobs, 1)
			dQ0p = .5*dQ0p / sqrt(mean(dQ0p**2))
		else :
			dP = dP
			#dQ0p = 5*dQ0p
		dQ0p = - step * self.s * dQ0p # this is useful for vizualisation            
		dQ0 =  mean(dQ0p, 0)          # but there's only one template in the end !
		dP  =  - step * self.s * dP
		"""
		
		# Store the variables to be displayed in a tuple attribute
		self.displayable_vars = (qts, Q, C, transports)
		
		# Set the descent speeds according to self.descent_parameters
		grad_Q0 = self.normalize_and_regularize_template_updates(dQ0p, iteration) 
		grad_P  = self.normalize_momentums(dP, prec_qt[0], iteration)
		
		C = self.template_cost(prec_qt[0]) + sum(C)
		
		return (C, grad_Q0, grad_P)
		
	def display_iteration(self, display_shoots = True, display_transports = True) :
		# Display
		self.show_HT()
		self.M.marker(self.Q0, marker = dict(size = 10, color='blue'), name='Frechet Mean', visible=False)
		if display_shoots :
			self.M.plot_traj(self.displayable_vars[0], line = dict(width= 2, color='rgb(0,128,0)'), name='Shootings', visible=False)
		else :
			self.M.marker(self.Q0, marker = dict(size = 0, color='blue'), name='Shoots', visible=False)
			
		self.show_data_attachment(self.displayable_vars[1], self.displayable_vars[2]) # 'Targets' + 'Distances'
		
		if (self.displayable_vars[3] != []) and display_transports :
			self.M.show_transport( self.displayable_vars[3], line = dict(width= 2, color='rgb(128,0,128)'), name = 'Transport', visible=False) # 'Transport'
		else :
			self.M.marker(self.Q0, marker = dict(size = 0, color='blue'), name='Transport', visible=False)
		
		# We use a simple hide/show scheme for the plot updates
		frames = [dict(visible = False), dict(visible = True)]
		return frames
		
		
	def cost_gradient(self, state, Xt, iteration, progress) :
		(C, grad_Q0, grad_P) = self.shooting_gradient(state, Xt, iteration, progress)
		return (C, AtlasVariables(grad_Q0, grad_P))
		
	def scal_L2(self, s, y) :
		"""A class which uses a non-orthodox AtlasVariables will have to overload this method."""
		scal_float = (self.M.L2_product_tangent(self.Q0, s.Q0, y.Q0 ) ) + \
		 sum( [ self.M.L2_product_cotangent(self.Q0, dps, dpy )  for (dps, dpy) in zip( s.P, y.P) ] )
		print('scal_float : ', scal_float)
		return ScalarAtlasVariables(scal_float, [ scal_float for p in s.P])
	def scals_L2(self, s, y) :
		"""A class which uses a non-orthodox AtlasVariables will have to overload this method."""
		return ScalarAtlasVariables(self.M.L2_product_tangent(self.Q0, s.Q0, y.Q0 ) ,
								  [ self.M.L2_product_cotangent(self.Q0, dps, dpy )  for (dps, dpy) in zip( s.P, y.P) ] )
	
	def density_normalize_direction(self, grad) :
		"Applies a diagonal matrix on the gradient."
		weights = self.density_normalization_weights()
		y = AtlasVariables(- grad.Q0, [p.scale(-weights) for p in grad.P])
		return y
			 
	def density_normalization_weights(self) :
		"As of today, it's a hack !!!"
		C = self.Q0 # assume it's a curve
		M = C.to_measure()
		#X = normal(0, 1, (10, 3))
		#X = M.points
		X = C.to_array()

		Ks = self.M.precompute_kernels(C)
		K = Ks[0]
		
		mutilde = zeros(C.array_shape()[0])
		for (i,s) in enumerate(C.connectivity) :
			mutilde[s[0]] += .5 * M.weights[i]
			mutilde[s[1]] += .5 * M.weights[i]
		
		return mutilde / ( (K.dot(mutilde))**2)
	
	
	def normalize_momentums(self, dP, prec_Q0, iteration) :
		# 'mode' values : 'Fixed Stepsize', '1/nit', 'Gradient', 'Gauss-Newton'
		if self.descent_parameters['models']['mode'] == 'Fixed Stepsize' :
			dP = [self.descent_parameters['models']['speed'] * dp / ( self.M.norm_p(self.Q0, dp, prec_Q0) ) for dp in dP]
		elif self.descent_parameters['models']['mode'] == '1/nit' :
			dP = [self.descent_parameters['models']['speed'] * dp / ( iteration * (self.M.norm_p(self.Q0, dp, prec_Q0) )) for dp in dP]
		elif self.descent_parameters['models']['mode'] == 'Gradient' :
			dP = [self.descent_parameters['models']['speed'] * dp  for dp in dP]
		else :
			raise(NotImplementedError)
		return dP
		
	def normalize_and_regularize_template_updates(self, dQ0p, iteration) :
		raise(NotImplementedError)
	def normalize_template_updates(self, dQ0p, norms, iteration) :
		# 'mode' values : 'Fixed Stepsize', '1/nit', 'Gradient', 'Gauss-Newton'
		if self.descent_parameters['template']['mode'] == 'Fixed Stepsize' :
			dQ0p = [self.descent_parameters['template']['speed'] * dq0 / norms for dq0 in dQ0p]
		elif self.descent_parameters['template']['mode'] == '1/nit' :
			dQ0p = [self.descent_parameters['template']['speed'] * dq0 / ( norms * iteration) for dq0 in dQ0p]
		elif self.descent_parameters['template']['mode'] == 'Gradient' :
			dQ0p = [self.descent_parameters['template']['speed'] * dq0 for dq0 in dQ0p]
		else :
			raise(NotImplementedError)
		return dQ0p
		
	def after_step(self, step) :
		"""
		Parameters update : simplest thing to do would be
		self.Q0 = self.Q0 + dQ0;
		self.P  = self.P  + dP;
		but dimensional constraints etc. might play a part.
		"update" is the method which actually separates models
		from one another
		"""
		new_state_Q0 = self.after_step_Q0(step.Q0)
		new_state_P  = self.after_step_P(step.P)
		return AtlasVariables(new_state_Q0, new_state_P)
	def set_state(self, new_state) :
		self.Q0 = new_state.Q0
		self.P  = new_state.P
		
	def get_frame(self, f) :
		# Five plotly traces per frame : 'Frechet Mean', 'Shootings', 'Targets', 'Models', 'Transports'
		list1 = str([ 1 + 5*self.current_frame, 2 + 5*self.current_frame, 3 + 5*self.current_frame, 4 + 5*self.current_frame, 5 + 5*self.current_frame])[1:-1]
		self.current_frame = f
		list2 = str([ 1 + 5*self.current_frame, 2 + 5*self.current_frame, 3 + 5*self.current_frame, 4 + 5*self.current_frame, 5 + 5*self.current_frame])[1:-1]
		return (self.frames[f] , [list1, list2])







