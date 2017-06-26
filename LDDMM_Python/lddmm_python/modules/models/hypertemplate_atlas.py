from pylab import *

from .atlas import Atlas

class HypertemplateAtlas(Atlas):
	"""
	Atlas with a shooted mean Q0 and no dimensional constraint.
	Important note :
	in general, the shooting method associated to the
	HT -> Template geodesic (Hilbert Space V0) can be different
	from the one used to get models X from the template.
	Here, we won't bother to define an other metric,
	and will simply use the one given by self.M.
	"""
	def __init__(self, *args, **kwargs):
		Atlas.__init__(self, *args, **kwargs)
		self.P0 = self.M.zero_momentum()
		self.QHT = self.Q0
	def template(self) :
		(Q0, _, _, _, _) = self.M.shoot(self.QHT, self.P0)
		return Q0
	def update(self, dP0, dP):
		# Parameters update :
		self.update_Q0(dP0)
		self.update_P(dP)
	def after_step_Q0(self, dP0) :
		# template update
		return self.P0 + dP0   # update of the template shooting parameter...
	def after_step_P(self, dP) :
		return [ p + dp for (p,dp) in zip(self.P, dP)]   # No dimensional constraint
	def set_state(self, new_state) :
		self.P0 = new_state.Q0
		self.Q0 = self.template() # to get a new template
		self.P  = new_state.P
	def template_cost(self, prec_QHT) :
		return .5* self.gamma_V0 * ( self.M.norm_p(self.QHT, self.P0, prec_QHT) **2) # Quadratic cost
		
	def normalize_and_regularize_template_updates(self, dQ0p, iteration) :
		dQ0 =  self.M.sum_position(self.Q0, dQ0p)
		
		(Q0, Qt, Pt, _, prec_Qt) = self.M.shoot(self.QHT, self.P0)
		(dQHT, dP0) = self.M.backward_scheme(dQ0, self.M.zero_momentum(), Qt, Pt, prec_Qt)
		(dQHT_reg, dP0_reg) = self.d_regularization_term_HT(self.QHT, self.P0, prec_Qt[0])
		# we will actually discard dQHT_reg, as we suppose it is fixed.
		dP0 = self.gamma_V0 * dP0_reg + dP0 
		
		norm_update = self.M.norm_p(self.QHT, dP0, prec_Qt[0]) # usual approx
		dP0 = self.normalize_template_updates([dP0], norm_update, iteration)[0]
		return dP0.ravel()
	def d_regularization_term_HT(self, q, p, prec_q) :
		"""
		Differential of the term associated to gamma_V0 :
		.5 * (p0, Kqht p0)
		"""
		dp =   self.M.K(q, p, prec_q)
		dq = - self.M.upP(q,p, prec_q)
		return (dq, dp)
	def show_HT(self) :
		self.M.marker(self.QHT, marker = dict(size = 10, color='rgb(255,0,255)'), name='Hypertemplate', visible=False)
		(Q0, Qt, Pt, _, _) = self.M.shoot(self.QHT, self.P0)
		self.M.plot_traj(Qt, line = dict(width= 2, color='rgb(255,128,0)'), name='Template Shooting', visible=False)
	def get_frame(self, f) :
		# Six plotly traces per frame : 'Hypertemplate', 'Template Shooting', 'Frechet Mean', 'Shootings', 'Targets', 'Models'
		list1 = str([ 1 + 6*self.current_frame, 2 + 6*self.current_frame, 3 + 6*self.current_frame, 4 + 6*self.current_frame, 5 + 6*self.current_frame, 6 + 6*self.current_frame])[1:-1]
		self.current_frame = f
		list2 = str([ 1 + 6*self.current_frame, 2 + 6*self.current_frame, 3 + 6*self.current_frame, 4 + 6*self.current_frame, 5 + 6*self.current_frame, 6 + 6*self.current_frame])[1:-1]
		return (self.frames[f] , [list1, list2])
		
		
