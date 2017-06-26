
from IPython.display import HTML, display
import plotly.offline
import json
from plotly import utils
from pylab import *
from ipywidgets import *
from pprint import pprint
import traitlets

from scipy.optimize import line_search
from numpy.random import random_sample

import plotly.graph_objs as go
from ..io.my_iplot import my_iplot

"""
def my_iplot(figure_or_data, show_link=False, link_text='Export to plot.ly',
		  validate=True, image=None, filename='plot_image', image_width=800,
		  image_height=600) :
	plot_html, plotdivid, width, height = plotly.offline._plot_html(
		figure_or_data, show_link, link_text, validate,
		'100%', 525, global_requirejs=True)
	display(HTML(plot_html))
	return plotdivid
"""

class Model :
	def __init__(self, M, C, nobs) :
		"""
		M : Manifold
		C : Cost function
		nobs : number of observations to fit
		s steps size
		"""
		self.M = M
		self.C = C
		self.nobs = nobs
		# Vizualisation
		self.its_per_frame = 10
		self.frames = []
		self.current_frame = 0
		self.cost_values = []
		self.M.show('3D')
		self.training_widget = None
		self.is_current_cost_computed = False
	def training_step(self, Mod, Xt) :
		None
		
	def train(self, Xt, descent_parameters, nits = 100, progressbar = None) :
		self.descent_parameters = descent_parameters
		self.init_iHessian()
		Cs = zeros(nits)
		self.Xt = Xt
		for it in range(nits) :
			print('Iteration : ', it, '  ==========================================')
			(_, c, dQ, frame, exit_code) = self.training_step(Xt, iteration = it + 1, progress = it / (nits-1) )
			c = sum(c)
			Cs[it] = c
			self.frames.append(frame)
			if progressbar is not None :
				progressbar.value = it + 1
			if exit_code == 1 :
				break
		self.cost_values = hstack((self.cost_values,Cs))
		return Cs
		
	def training_step(self, Xt, iteration = 1, progress = 0) :
		"""Makes one gradient step, using a naive or an L-BFGS scheme."""
		if self.is_current_cost_computed :
			# Depending on the line search method, we may have computed the
			# current cost & gradient already. We simply load it :
			(C, grad) = self.current_cost_grad
			self.is_current_point_computed = False
		else :
			# Otherwise, we simply compute it
			(C, grad) = self.cost_gradient(self.current_state(), Xt, iteration, progress)
			
		# Don't forget that we are *minimizing* the energy,
		# so there'll be a minus sign somewhere in this method !
		search_dir = self.descent_direction(grad)
		(step, exit_code) = self.line_search(search_dir, C, grad, Xt, iteration, progress)
		
		# Display the stored variables
		frames = self.display_iteration(display_shoots     = ((iteration % 10) == 1) or (progress >= .95) ,
										display_transports = ((iteration % 10) == 1) or (progress >= .95))
		# Update our (invert of) descent metric, the (inverse of the) Hessian.
		self.update_iHessian(step, grad)
		return ([],C, step, frames, exit_code)
	
	def line_search(self, search_dir, C, grad, Xt, iteration, progress) :
		if self.descent_parameters['scheme']['line search'] == 'naive' :
			step = search_dir
			new_state = self.after_step(step)
			self.set_state(new_state)
			return (step, 0) 
		elif self.descent_parameters['scheme']['line search'] == 'backtracking' :
			fun = lambda x : self.cost_gradient(x, Xt, iteration, progress)
			exp_decrease = self.scal_L2(grad, search_dir).Q0   # Here, we assume that we're using a L2 gradient !
			return self.backtracking_line_search(fun, search_dir, C, exp_decrease)
		elif self.descent_parameters['scheme']['line search'] == 'wolfe' :
			fun = lambda x : self.cost_gradient(x, Xt, iteration, progress)
			exp_decrease = self.scal_L2(grad, search_dir).Q0   # Here, we assume that we're using a L2 gradient !
			return self.wolfe_line_search(fun, search_dir, C, exp_decrease)
		
		
		
	def init_iHessian(self) :
		if self.descent_parameters['scheme']['direction'] == 'gradient' :
			self.iHessian = None
		elif self.descent_parameters['scheme']['direction'] == 'L-BFGS' :
			self.iHessian = []
			
	def descent_direction(self, grad) :
		ret = None
		if self.descent_parameters['scheme']['direction'] == 'gradient' :
			ret = self.naive_descent_direction(grad)
		elif self.descent_parameters['scheme']['direction'] == 'density normalized' :
			ret = self.density_normalize_direction(grad)
		elif self.descent_parameters['scheme']['direction'] == 'L-BFGS' :
			ret = self.LBFGS_descent_direction(grad)
			
		if self.reg_template_gradient is not None :
			ret.Q0 = self.regularize_template_gradient(ret.Q0)
			
		return ret
		
		
	def update_iHessian(self, step, grad) :
		if self.descent_parameters['scheme']['direction'] == 'gradient' :
			self.iHessian = None
		if self.descent_parameters['scheme']['direction'] == 'L-BFGS' :
			self.LBFGS_update_iHessian(step, grad) 
	
	def naive_descent_direction(self, grad) :
		return -grad
	def density_normalize_direction(self, grad) :
		raise(NotImplementedError)
	def regularize_template_gradient(self, grad) :
		raise(NotImplementedError)
		
	def LBFGS_descent_direction(self, grad) :
		"""
		Reference : wikipedia : https://en.wikipedia.org/wiki/Limited-memory_BFGS.
		This implements a two-loop recursion L-BFGS algorithm.
		
		Using this code, we try to be aware of the metric which is used in the gradient / quasi-Newton descent :
		as the most principled metric on the momentums p is K_q0,
		we cannot carelessly merge grad_Q0 and the concatenated grad_P[i] in a single large vector
		on which we would do an L2 minimization.
		As of today, we use an L2 metric (which tends to regularize the updates dP[i]),
		but this is subject to changes in the future !
		"""
		
		def scal( s, y ) :
			return self.scals_L2(s, y)
			
		#	return (self.M.L2_product_tangent(self.Q0, s.Q0, y.Q0 ) ) + \
		#	 sum( [ self.M.L2_product_cotangent(self.Q0, dps, dpy )  for (dps, dpy) in zip( s.P, y.P) ] )
			 
		q = grad                      # q = grad_k
		if len(self.iHessian) > 0 :
			self.iHessian[-1][1] += q # y_(k-1) = q - grad_(k-1)
			self.iHessian[-1][2] = scal( self.iHessian[-1][0], self.iHessian[-1][1]).inverse() # r_k = 1 / (s_k . y_k)
			print(self.iHessian[-1][2])
		prevs = self.iHessian
		
		a = []
		for (si, yi, ri) in reversed(prevs) : # for i = k-1, ..., k-m :
			a.append( ri * scal( si, q ) )    #      a_i = r_i * (s_i . q )
			q -= a[-1] * yi                   #      q   = q - a_i * y_i
		
		if len(prevs) > 0 :
			Hk0 = scal( prevs[-1][0], prevs[-1][1] ) / scal( prevs[-1][1], prevs[-1][1] ) # H_k^0 = (s_(k-1) . y_(k-1)) / (y_(k-1) . y_(k-1))
			z = Hk0 * q  # initial descent direction
		else :
			z = q
			
		for ((si, yi, ri), ai) in zip( prevs, reversed(a) ) :  # for i = k-m, ..., k-1 :
			b  = ri * scal( yi, z )                            #      b_i = r_i * (y_i . z )
			z += (ai - b) * si                                 #      z   = z + (a_i - b_i) * s_i
		z = -z  # We are minimizing the objective !
		
		return z
		
	def LBFGS_update_iHessian(self, step, grad) :
		self.iHessian.append( [step, -grad, 0] )
		m = int(self.descent_parameters['scheme']['direction memory'])
		self.iHessian = self.iHessian[-m:] # We only keep in memory the last m updates
	
	def backtracking_line_search(self, fun, search_dir, curr_value, exp_decrease, rho = .5, c = 0) :
		"""
		Simple backtracking line search : see Numerical Optimization,
		Nocedal and Wright, Algorithm 3.1, p. 37
		"""
		a = 1 # \bar{alpha} = 1
		step = a * search_dir
		for i in range(6) :
			new_state = self.after_step(step)
			(C,grad) = fun(new_state)
			if C <= curr_value + c * a * exp_decrease :
				self.set_state(new_state)
				self.current_cost_grad = (C,grad)
				self.is_current_cost_computed = True
				return (step , 0)
			else :
				a    = rho * a
				step = a * search_dir
		
		if C > curr_value :
			print('This is not a descent direction !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			print('Expected decrease per unit : ', exp_decrease)
			print('Current value :              ', curr_value)
			print('Minimum required value :     ', curr_value + c * a * exp_decrease / rho)
			print('New value :                  ', C)
		else :
			print('This is not the correct gradient ! (note that this might be due to the sampling of the geodesics)')
			print('Expected decrease per unit : ', exp_decrease)
			print('Current value :              ', curr_value)
			print('Minimum required value :     ', curr_value + c * a * exp_decrease / rho)
			print('New value :                  ', C)
		
		step = .1 * search_dir  
		new_state = self.after_step(step)
		(C,grad) = fun(new_state)
		self.set_state(new_state) # we'd like to be able to see what's wrong here
		self.current_cost_grad = (C,grad)
		self.is_current_cost_computed = True
		return  (step, 0) # Exit_code = 1 : break !
		
	def wolfe_line_search(self, fun, search_dir, curr_value, exp_decrease) :
		"""
		see Numerical Optimization,
		Nocedal and Wright, Algorithm 3.5, p. 60
		"""
		f =  lambda t : fun(self.after_step(t * search_dir))[0]
		fp = lambda t : self.scal_L2(fun(self.after_step(t * search_dir))[1], search_dir).Q0
		exit_code = 0 # Default : everything is all right
		
		
		# Code to uncomment to check that fp is the true derivative of f========
		h = 1e-8
		for i in range(5) :
			t = random_sample()
			update_th = fp(t)
			update_emp = (f(t+h) - f(t-h)) / (2*h)
			print('')
			print('search dir : ', search_dir.to_array())
			print('Checking the function passed to the Wolfe line search, t = ', t)
			print('Empirical   derivative : ', update_emp)
			print('Theoretical derivative : ', update_th)
		
		#=======================================================================
		
		
		
		print("Exp decrease : ", exp_decrease)
		(a, _, _, _, _, _) = line_search(f, fp, 0, 1, exp_decrease, curr_value, c2 = 0.95)
		if a == None :
			print('Error during the wolfe line search')
			a = 0
			exit_code = 1 # Exit_code = 1 : break !
		step = a * search_dir
		new_state = self.after_step(step)
		self.set_state(new_state)
		#self.current_cost_grad = (C,grad)
		#self.is_current_cost_computed = True
		self.is_current_point_computed = False
		return (step, exit_code)
		
	def show(self, vis_mode = '3D', title = '') :
		self.M.layout['showlegend'] = True
		(wid, div_id) = self.M.iplot(title)
		(costs_curve, div_costs_curve, cost_values) = self.costs_curve()
		pause(1)
		hb = self.show_training(div_id, div_costs_curve, cost_values)
		self.training_widget = VBox([hb, wid, costs_curve], layout = Layout(width='100%', justify_content='center'))
		display(self.training_widget)
		return div_id
	def costs_curve(self) :
		values = self.cost_values
		s = unique(values)
		values = values - s[0] + .5 * (s[1] - s[0])
		# Create a trace
		trace = go.Scatter(
			x = arange(len(values))+1,
			y = array(values),
			name = 'Cost excess'
		)
		mark = go.Scatter(
			x = array([1]),
			y = array([values[0]]),
			marker = dict(
				color = "rgb(0, 0, 128)",
				size = 15
			),
			name = 'Current value',
			mode = 'markers'
		)
		data = [trace, mark]
		layout = go.Layout(
			title='Cost excess across iterations',
			width=800,
			height=800,
			legend = dict( x = .8, y = 1),
			#xaxis = dict(range = [-3,3]),
			#yaxis = dict(range = [-3,3])
			yaxis=dict(
				type='log',
				autorange=True
			)
		)
		return my_iplot(go.Figure(data=data, layout=layout)) + (values,)
		
		
	def show_data_attachment(self, Q, C):
		C = (9 * (C  / max(C) )).round() + 1
		X = self.M.I(Q)
		assert(len(self.Xt) == len(X))
		self.M.marker(self.Xt, marker = dict( color = C, colorscale='Viridis', showscale=False ), name = 'Targets', visible=False)
		self.M.quiver(X, [xt - x for (xt,x) in zip(self.Xt, X)], line   = dict(width= 3, color='red'), name = 'Models', visible=False)
		
	def show_training(self, div_id, div_costs_curve, cost_values) :
		#div_id = self.show(*args, **kwargs)
		
		def change_frame(w) :
			#print(w)
			updates = self.get_frame(w-1)
			script = ''
			for i in range(len(updates[0])) :
				jupdate      = json.dumps(updates[0][i], cls=utils.PlotlyJSONEncoder)
				jupdate_cost = json.dumps(dict(x = [[w]], y = [[cost_values[w-1]]]), cls=utils.PlotlyJSONEncoder)
				script = script \
					+ 'Plotly.restyle("{id}", {update}, [{index}]);'.format(
					id=div_id,
					update=jupdate, index = updates[1][i]) \
					+ 'Plotly.restyle("{id}", {update}, [{index}]);'.format(
					id=div_costs_curve,
					update=jupdate_cost, index = 1)
			update_str = (
				'<script type="text/javascript">' +
				'window.PLOTLYENV=window.PLOTLYENV || {{}};' +
				'window.PLOTLYENV.BASE_URL="' + 'https://plot.ly' + '";' +
				'{script}' +
				'</script>').format(script=script)
			#print(script)
			display(HTML(update_str))
		#print(self.frames)
		maxframe = len(self.frames) - 1
		play = Play(value=maxframe)
		slider = IntSlider(min=1, max=maxframe, step=1, value=maxframe,continuous_update=False)
		slider.layout.width = '100%'
		#jslink((play, 'value'), (slider, 'value'))
		traitlets.link((play, 'value'), (slider, 'value'))
		hb = HBox([play, slider])
		slider.observe((lambda iteration : change_frame(iteration['new'])), names='value')
		change_frame(maxframe)
		#play.observe((lambda iteration : print(iteration['new'])), names='value')
		#display(hb)
		return hb
