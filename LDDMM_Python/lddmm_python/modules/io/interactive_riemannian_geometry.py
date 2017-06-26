from pylab import *
from pprint import pprint
import json
from IPython.display import HTML, display

import traitlets
from ipywidgets import *

from .show_code import show_code

from ..manifolds.torus import Torus
from ..manifolds.landmarks import Landmarks
from ..manifolds.curves import Curve, Curves

from ..models.frechet_mean import FrechetMean
from ..models.free_atlas import FreeAtlas
from ..models.free_1D_atlas import Free1DAtlas
from ..models.hypertemplate_atlas import HypertemplateAtlas
#from SharedVariables import t, M, model, Xt
global t, M, model, Xt
t = None
M = None
model = None
Xt = None

class AtlasInterface :
	def __init__(self, show = False) :
		self.create_manifold_interface()
		self.create_start_point_interface()
		self.create_data_interface()
		self.create_model_interface()
		self.create_training_interface()
		self.create_display_interface()
		
		self.link_widgets()
		self.create_layout()
		if show :
			display(self.widget)

	def link_widgets(self) :
		self.w_manifold_type.observe(self.on_manifold_type_change, names='value')
		self.w_create_manifold.on_click(self.create_manifold)
		self.w_choose_hypertemplate.on_click(self.choose_hypertemplate)
		self.w_choose_data.on_click(self.choose_data)
		self.w_create_model.on_click(self.create_model)
		self.w_train.on_click(self.train_model)
		self.w_show.on_click(self.show_model)
		traitlets.link((self.w_niterations, 'value'), (self.w_iterations, 'max'))
		
		# Energy update
		self.w_template_type.observe(self.update_energy_latex, names='value')
		self.w_reg_hypertemplate.observe(self.update_energy_latex, names='value')
		self.w_sigma_reg_hypertemplate.observe(self.update_energy_latex, names='value')
		self.w_gamma_V0.observe(self.update_energy_latex, names='value')
		self.w_sigma_V0.observe(self.update_energy_latex, names='value')
		self.w_shooting_dim_constraint.observe(self.update_energy_latex, names='value')
		self.w_gamma_V.observe(self.update_energy_latex, names='value')
		self.w_frechet_exponent.observe(self.update_energy_latex, names='value')
		self.w_data_attachment.observe(self.update_energy_latex, names='value')
		self.w_gamma_W.observe(self.update_energy_latex, names='value')
		
		
		# Algorithm update
		self.w_X0_gradient_distribution.observe(self.update_algorithm_latex, names='value')
		self.w_X0_descent_mode.observe(self.update_algorithm_latex, names='value')
		self.w_X0_descent_speed.observe(self.update_algorithm_latex, names='value')
		self.w_Xi_gradient_distribution.observe(self.update_algorithm_latex, names='value')
		self.w_Xi_descent_mode.observe(self.update_algorithm_latex, names='value')
		self.w_Xi_descent_speed.observe(self.update_algorithm_latex, names='value')
		self.w_Ei_gradient_distribution.observe(self.update_algorithm_latex, names='value')
		self.w_Ei_descent_mode.observe(self.update_algorithm_latex, names='value')
		self.w_Ei_descent_speed.observe(self.update_algorithm_latex, names='value')
		self.w_descent_stopping_criterion.observe(self.update_algorithm_latex, names='value')
		self.w_niterations.observe(self.update_algorithm_latex, names='value')
		self.w_descent_threshold.observe(self.update_algorithm_latex, names='value')
		
		
		self.update_energy_latex()
		self.update_algorithm_latex()
	def create_manifold_interface(self):
		self.w_manifold_type = Dropdown(
			options={'Landmarks 2D': 'Landmarks', 'Torus': 'Torus', 'Curves 2D' : 'Curves'},
			value='Torus',
			description='',
			disabled = False,
			button_style = 'primary',
			width = '200px'
		)
		# Landmarks param
		self.w_npoints = BoundedIntText(
			value = 25,
			min=1,
			max=400,
			step=1,
			description='Points :',
			disabled=True,
			width='148px'
		)
		self.w_kernel_type = Dropdown(
			options={'Gaussian Kernel': 'gaussian'},
			value='gaussian',
			description='',
			disabled = True,
			width = '148px'
		)
		self.w_kernel_size = BoundedFloatText(
			value=0.5,
			min=0.01,
			max=10.0,
			description='Size :',
			disabled=True,
			width='148px'
		)
		# Torus param
		self.w_donut_radius = BoundedFloatText(
			value=2,
			min=0,
			max=4,
			disabled=False,
			description='Radius :',
			width='148px'
		)
		self.w_donut_section = BoundedFloatText(
			value=1,
			min=0,
			max=4,
			disabled=False,
			description='Section :',
			width='148px'
		)
		self.w_create_manifold = Button(
			description="Create Manifold",
			button_style = 'success',
			width = '148px',
			height = '68px',
			disabled = False)
	def update_hypertemplate_field(self, i) :
		def curryfied(x) :
			self.w_hypertemplate.value = self.hypertemplate_buttons[self.w_manifold_type.value][i]['code']
		return curryfied
	def create_start_point_interface(self) :
		self.w_hypertemplate = Text(
			value='',
			placeholder='Type something or click on one of the above buttons',
			description='',
			disabled=False,
			width = '450px',
			font_family = 'monospace',
			font_size = 16
		)
		item_layout = Layout(height='68px', width='71px')
		self.hypertemplate_buttons = {
			'Torus' : [ 
				{ 'text' : 'zero', 'code' : '(0,0)' } ,
				{ 'text' : 'rand', 'code' : '(rand(),rand())' } ,
				{ 'text' : '', 'code' : '' } ,
				{ 'text' : '', 'code' : '' } ,
				{ 'text' : '', 'code' : '' } ,
				{ 'text' : '', 'code' : '' } ] ,
			'Landmarks' : [
				{ 'text' : 'circle', 'code' : '(cos(t), sin(t))'},
				{ 'text' : 'square', 'code' : '( minimum(maximum(  (4/pi)*abs(t - .75*pi) - 2  , -1) , 1)  , minimum(maximum(  (4/pi)*abs(t - 1.25*pi) - 2  , -1) , 1) )'}, 
				{ 'text' : 'segment', 'code' : '(0*t, (t/pi) - 1)'}, 
				{ 'text' : '', 'code' : ''}, 
				{ 'text' : '', 'code' : ''}, 
				{ 'text' : '', 'code' : ''} ] ,
			'Curves' : [
				{ 'text' : 'circle', 'code' : '(cos(t), sin(t))'},
				{ 'text' : 'square', 'code' : '( minimum(maximum(  (4/pi)*abs(t - .75*pi) - 2  , -1) , 1)  , minimum(maximum(  (4/pi)*abs(t - 1.25*pi) - 2  , -1) , 1) )'}, 
				{ 'text' : 'segment', 'code' : '(0*t, (t/pi) - 1)'}, 
				{ 'text' : 'skull', 'code' : "'data/skulls_2D/skull.vtk'"}, 
				{ 'text' : '', 'code' : ''}, 
				{ 'text' : '', 'code' : ''} ]
		}
		
		self.w_options_hypertemplate_buttons = [Button(description = self.hypertemplate_buttons[self.w_manifold_type.value][i]['text'], layout=item_layout, button_style='') for i in range(6)]
		for i in range(6) :
			self.w_options_hypertemplate_buttons[i].on_click(self.update_hypertemplate_field(i))
		carousel_layout = Layout(
							width='450px',
							height='',
							flex_direction='row',
							display='flex')
		self.w_options_hypertemplate = HBox(children=self.w_options_hypertemplate_buttons, layout=carousel_layout)
		self.w_choose_hypertemplate = Button(
			description="Choose HT",
			button_style = 'success',
			width = '148px',
			height = '104px',
			disabled = True)
	
	def update_data_field(self, i) :
		def curryfied(x) :
			self.w_data.value = self.data_buttons[self.w_manifold_type.value][i]['code']
		return curryfied
	def create_data_interface(self) :
		self.w_data = Text(
			value='',
			placeholder='Type something or click on one of the above buttons',
			description='',
			disabled=False,
			width = '450px',
			font_family = 'monospace',
			font_size = 16
		)
		item_layout = Layout(height='68px', width='71px')             
		self.data_buttons = {
			'Torus' : [ 
				{ 'text' : 'line', 'code' : '[ (R*cos(theta), R*sin(theta), 2 * ((theta)/pi - .5) + 0.2*randn() ) for '+
											' (R,theta) in zip( 3 + 0.2*randn(10), pi*rand(10) ) ]' } ,
				{ 'text' : 'randn', 'code' : '[ (.8*randn(), .8*randn(), .8*randn()) for i in range(20) ]' } ,
				{ 'text' : '', 'code' : '' } ,
				{ 'text' : '', 'code' : '' } ,
				{ 'text' : '', 'code' : '' } ,
				{ 'text' : '', 'code' : '' } ] ,
			'Landmarks' : [
				{ 'text' : 'chips', 'code' : '[(2*cos(t) + sin(t), sin(t) + randn() *sin(2*t) ) for i in range(2)]'},
				{ 'text' : 'tricky', 'code' : '[((.5+rand())*cos(t) + randn(), (.5+rand())*sin(t) + randn()) for i in range(2)]'}, 
				{ 'text' : 'easy', 'code' : '[((.5+rand())*cos(t) + (rand()-.5), (.5+rand())*sin(t) + (rand()-.5) ) for i in range(2)]'}, 
				{ 'text' : 'segments', 'code' : '[(.7*randn()*((t/pi)-1) + .2*randn(), .7*randn()*((t/pi)-1) + .2*randn() ) for i in range(2)]'}, 
				{ 'text' : '', 'code' : ''}, 
				{ 'text' : '', 'code' : ''} ] ,
			'Curves' : [
				{ 'text' : 'chips', 'code' : '[(2*cos(t) + sin(t), sin(t) + randn() *sin(2*t) ) for i in range(2)]'},
				{ 'text' : 'tricky', 'code' : '[((.5+rand())*cos(t) + randn(), (.5+rand())*sin(t) + randn()) for i in range(2)]'}, 
				{ 'text' : 'easy', 'code' : '[((.5+rand())*cos(t) + (rand()-.5), (.5+rand())*sin(t) + (rand()-.5) ) for i in range(2)]'}, 
				{ 'text' : 'segments', 'code' : '[(.7*randn()*((t/pi)-1) + .2*randn(), .7*randn()*((t/pi)-1) + .2*randn() ) for i in range(2)]'}, 
				{ 'text' : 'skulls', 'code' : "[ 'data/skulls_2D/skull_australopithecus.vtk', 'data/skulls_2D/skull_erectus.vtk', 'data/skulls_2D/skull_habilis.vtk', 'data/skulls_2D/skull_neandertalis.vtk', 'data/skulls_2D/skull_sapiens.vtk' ]"}, 
				{ 'text' : '', 'code' : ''} ]
		}
		
		self.w_options_data_buttons = [Button(description = self.data_buttons[self.w_manifold_type.value][i]['text'], layout=item_layout, button_style='') for i in range(6)]
		for i in range(6) :
			self.w_options_data_buttons[i].on_click(self.update_data_field(i))
				
		carousel_layout = Layout(
							width='450px',
							height='',
							flex_direction='row',
							display='flex')
		self.w_options_data = HBox(children=self.w_options_data_buttons, layout=carousel_layout)
		self.w_choose_data = Button(
			description="Choose Data",
			button_style = 'success',
			width = '148px',
			height = '104px',
			disabled = True)
			
	def create_model_interface(self) :
		self.w_template_type = Dropdown(
			options=['Free', 'Shooted'],
			value='Free',
			description='',
			button_style = 'info',
			width = '148px'
		)
		self.w_reg_hypertemplate = Dropdown(
			options = ['No reglztion', 'Gaussian reglztion'],
			value = 'No reglztion',
			description = '',
			info = 'Prevents a Free Template from becoming edgy',
			width = '148px'
		)
		self.w_sigma_reg_hypertemplate = BoundedFloatText(
			value = 0.5,
			min=0,
			max=2,
			description='$\\sigma_{\\text{reg}}$',
			width='148px'
		)
		self.w_gamma_V0 = FloatText(
			value=0.01,
			description='$\\gamma_{V_0}$',
			width='148px'
		)
		self.w_sigma_V0 = BoundedFloatText(
			value = 0.5,
			min=0.1,
			max=2,
			description='$\\sigma_{V_0}$',
			width='148px'
		)
		
		self.w_shooting_dim_constraint = Dropdown(
			options = ['No shooting', 'rank(p) = 1', 'rank(p) = 2', 'No dim constr'],
			value = 'No dim constr',
			description = '',
			button_style = 'info',
			width = '148px'
		)
		self.w_gamma_V = FloatText(
			value=0.05,
			description='$\\gamma_{V}$',
			disabled=False,
			width='148px'
		)
		self.w_frechet_exponent = BoundedIntText(
			value = 2,
			min=1,
			max=2,
			step=1,
			description='Frechet :',
			disabled=False,
			width='148px'
		)
		
		self.data_attachment_options = {
			'Torus' : {
				'options' : {
					'Squared L2': 'squared_distance', 
					'L2': 'distance'},
				'value' : 'squared_distance'
				},
			'Landmarks' : {
				'options' : {
					'Squared L2': 'squared_distance', 
					'L2': 'distance', 
					'Gaussian Kernel' : 'kernel_matching', 
					'Optimal Transport' : 'sinkhorn_matching'},
				'value' : 'squared_distance'
				},
			'Curves' : {
				'options' : {
					'Currents' : 'currents',
					'Varifolds' : 'varifolds',
					'Gaussian Kernel' : 'kernel_matching',
					'Normal Cycles' : 'normal_cycles', 
					'Optimal Transport' : 'sinkhorn_matching'},
				'value' : 'kernel_matching'
				},
		}
		self.w_data_attachment = Dropdown(
			options=self.data_attachment_options[self.w_manifold_type.value]['options'],
			value=self.data_attachment_options[self.w_manifold_type.value]['value'],
			description='',
			disabled = False,
			button_style = 'info',
			width = '148px'
		)
		self.w_gamma_W = FloatText(
			value=1,
			description='$\\gamma_{W}$',
			disabled=False,
			width='148px',
		)
		self.w_sigma_W_start = BoundedFloatText(
			value = 2,
			min=0.01,
			max=5,
			description='$\\sigma_{W}^{\\text{start}}$',
			width='148px'
		)
		self.w_sigma_W_end = BoundedFloatText(
			value = .1,
			min=0.01,
			max=5,
			description='$\\sigma_{W}^{\\text{end}}$',
			width='148px'
		)
		self.w_energy_latex = Label('$$E = 0$$', width = '450px')
		
		self.w_create_model = Button(
			description="Choose Model",
			button_style = 'success',
			width = '148px',
			height = '350px',
			disabled = True)
	def get_descent_parameters(self) :
		descent_parameters = {
			'template' : {
				'distribution' : self.w_X0_gradient_distribution.value,
				'mode' : self.w_X0_descent_mode.value,
				'speed' : self.w_X0_descent_speed.value
			},
			'models' : {
				'distribution' : self.w_Xi_gradient_distribution.value,
				'mode' : self.w_Xi_descent_mode.value,
				'speed' : self.w_Xi_descent_speed.value
			},
			'directions' : {
				'distribution' : self.w_Ei_gradient_distribution.value,
				'mode' : self.w_Ei_descent_mode.value,
				'speed' : self.w_Ei_descent_speed.value
			},
			'scheme' : {
				'direction'   : 'gradient',
				'line search' : 'naive',
				'direction memory' : 5
			}
		}
		return descent_parameters
	def create_training_interface(self) :
		
		# Descent parameters for X0
		self.w_X0_gradient_distribution = Dropdown(
			options = ['Mean'],
			value = 'Mean',
			width = '148px'
		)
		self.w_X0_descent_mode = Dropdown(
			#options = ['Fixed Stepsize', '1/nit', 'Gradient', 'Gauss-Newton'],
			options = ['Fixed Stepsize', '1/nit', 'Gradient'],
			value = 'Gradient',
			width = '148px'
		)
		self.w_X0_descent_speed = FloatText(
			value=1,
			description = 'Speed :',
			width='148px'
		)
		
		# Descent parameters for Xi / lambda_i
		self.w_Xi_gradient_distribution = Dropdown(
			options = ['Mean'],
			value = 'Mean',
			width = '148px'
		)
		self.w_Xi_descent_mode = Dropdown(
			#options = ['Fixed Stepsize', '1/nit', 'Gradient', 'Gauss-Newton'],
			options = ['Fixed Stepsize', '1/nit', 'Gradient'],
			value = 'Gradient',
			width = '148px'
		)
		self.w_Xi_descent_speed = FloatText(
			value=1,
			description = 'Speed :',
			width='148px'
		)
		
		# Descent parameters for Ei
		self.w_Ei_gradient_distribution = Dropdown(
			options = ['lambda', 'sign(lambda)', '1/lambda'],
			value = 'sign(lambda)',
			width = '148px'
		)
		self.w_Ei_descent_mode = Dropdown(
			#options = ['Fixed Stepsize', '1/nit', 'Gradient', 'Gauss-Newton'],
			options = ['Fixed Stepsize', '1/nit', 'Gradient'],
			value = 'Fixed Stepsize',
			width = '148px'
		)
		self.w_Ei_descent_speed = FloatText(
			value=0.01,
			description = 'Speed :',
			width='148px'
		)
		
		# Stopping criterion
		self.w_descent_stopping_criterion = Dropdown(
			#options = ['nits > ...', 'dE < ...'],
			options = ['nits > ...'],
			value = 'nits > ...',
			width = '148px'
		)
		self.w_niterations = BoundedIntText(
			value = 100,
			min=1,
			max=1000,
			step=1,
			description='',
			disabled=False,
			width='148px'
		)
		self.w_descent_threshold = FloatText(
			value = 0,
			width = '148px'
			)
			
		# Latex cell
		self.w_descent_algorithm_latex = Label(
			'',
			width = '450px',
			height = '100px')
		# Train button
		self.w_train = Button(
			description="Train",
			button_style = 'success',
			width='148px',
			height='250px',
			disabled = True)
	def create_display_interface(self) :
		self.w_iterations = IntProgress(
			value=0,
			min=0,
			max=self.w_niterations.value,
			step=1,
			description='',
			bar_style='success', # 'success', 'info', 'warning', 'danger' or ''
			orientation='horizontal',
			width = '450px'
		)
		self.w_show = Button(
			description="Show",
			button_style = 'success',
			width='148px',
			disabled = True)
			
	def on_manifold_type_change(self, change):
		if change['new'] == 'Torus' :
			self.w_npoints.disabled = True
			self.w_kernel_type.disabled = True
			self.w_kernel_size.disabled = True
			self.w_donut_radius.disabled = False
			self.w_donut_section.disabled = False
		elif change['new'] == 'Landmarks' :
			self.w_npoints.disabled = False
			self.w_kernel_type.disabled = False
			self.w_kernel_size.disabled = False
			self.w_donut_radius.disabled = True
			self.w_donut_section.disabled = True
		elif change['new'] == 'Curves' :
			self.w_npoints.disabled = False
			self.w_kernel_type.disabled = False
			self.w_kernel_size.disabled = False
			self.w_donut_radius.disabled = True
			self.w_donut_section.disabled = True
		for i in range(len(self.w_options_data_buttons)) :
			self.w_options_data_buttons[i].description = self.data_buttons[self.w_manifold_type.value][i]['text']
		for i in range(len(self.w_options_hypertemplate_buttons)) :
			self.w_options_hypertemplate_buttons[i].description = self.hypertemplate_buttons[self.w_manifold_type.value][i]['text']
		self.w_data_attachment.options = self.data_attachment_options[self.w_manifold_type.value]['options']
		self.w_data_attachment.value = self.data_attachment_options[self.w_manifold_type.value]['value']
		
	def update_energy_latex(self, nobodycares=None) :
		if self.w_frechet_exponent.value == 1 :
			frechet_str = '^{1/2}'
		elif self.w_frechet_exponent.value == 2 :
			frechet_str = ''
			
		if self.w_template_type.value == 'Free' :
			XHT_X0 = '0~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\\text{(flat template prior)}'
		elif self.w_template_type.value == 'Shooted' :
			XHT_X0 = '\\frac{\\gamma_{V_0}}{2}\\big(p_0, K_{\\sigma_{V_0},X_{\\text{HT}}} p_0 \\big)~~~~~~~~~~\\text{(template prior)}'
					 
		if self.w_data_attachment.value == 'squared_distance' :
			attachment = '\\big|X_i - \\widetilde{X_i}\\big|^2_{\\text{pointwise}}'
		elif self.w_data_attachment.value == 'distance' :
			attachment = '\\big|X_i - \\widetilde{X_i}\\big|_{\\text{pointwise}}'
		elif self.w_data_attachment.value == 'monge_kantorovitch' :
			attachment = '\\big|X_i - \\widetilde{X_i}\\big|^2_{\\text{transport}}'
		elif self.w_data_attachment.value == 'currents' :
			attachment = '\\big|\\omega(X_i) - \\omega(\\widetilde{X_i})\\big|^2_{\\sigma_W}'
		elif self.w_data_attachment.value == 'varifolds' :
			attachment = '\\big|X_i - \\widetilde{X_i}\\big|^2_{\\text{varifolds}}'
		elif self.w_data_attachment.value == 'kernel_matching' :
			attachment = '\\big|\\mu_{X_i} - \\mu_{\\widetilde{X_i}}\\big|^{\\star 2}_{\\sigma_W}'
		elif self.w_data_attachment.value == 'sinkhorn_matching' :
			attachment = '\\big|\\mu_{X_i} - \\mu_{\\widetilde{X_i}}\\big|^{\\star 2}_{\\text{Wasserstein}}'
		elif self.w_data_attachment.value == 'normal_cycles' :
			attachment = '\\big|N_{X_i} - N_{\\widetilde{X_i}}\\big|^{2}_{\\sigma_W}'
			
			
		X0_Xi = '\\frac{\\gamma_{V}}{2 \cdot n_{\\text{obs} }} \\sum_{i=1}^{n_{\\text{obs}}} \\big(p_i, K_{X_0} p_i \\big)' + frechet_str + '~~~~\\text{(shooting cost)}'
		Xi_Xt = '\\frac{\\gamma_{W}}{2 \cdot n_{\\text{obs} }} \\sum_{i=1}^{n_{\\text{obs}}}' + attachment  + '~~~~\\text{(data attachment)}'
		pad = '~~~~~~~~~~~~'
		code = pad + 'C = '    + XHT_X0 \
		     + '\\\\' +pad+ '\\,~~+' + X0_Xi \
		     + '\\\\' +pad+ '\\,~~+' + Xi_Xt
		
		self.w_energy_latex.value = '$$' + code + '$$'
	def create_manifold(self, b):
		global M
		if self.w_manifold_type.value == "Torus" :
			code = '''M = Torus( a = {rad}, 
           b = {sec} )'''.format(
				rad = self.w_donut_radius.value,
				sec = self.w_donut_section.value
			)
			M = Torus( a = self.w_donut_radius.value, b = self.w_donut_section.value )
		elif self.w_manifold_type.value == "Landmarks" :
			code = '''M = Landmarks( npoints = {npoints}, 
               kernel = {kernel} )'''.format(
				npoints = self.w_npoints.value,
				kernel = (self.w_kernel_type.value, self.w_kernel_size.value)
			) 
			M = Landmarks(npoints = self.w_npoints.value, 
				kernel = (self.w_kernel_type.value, self.w_kernel_size.value))
		elif self.w_manifold_type.value == "Curves" :
			code = '''M =    Curves( connectivity = array([ [i, i+1] for i in range(self.w_npoints.value - 1) ]),
               npoints = {npoints}, 
               kernel = {kernel} )'''.format(
				npoints = self.w_npoints.value,
				kernel = (self.w_kernel_type.value, self.w_kernel_size.value)
			) 
			M = Curves( connectivity = array([ [i, i+1] for i in range(self.w_npoints.value - 1) ]),
				npoints = self.w_npoints.value, 
				kernel = (self.w_kernel_type.value, self.w_kernel_size.value))
		show_code(code)
		self.w_create_manifold.disabled = True
		self.w_choose_hypertemplate.disabled = False
	def choose_hypertemplate(self, b):
		if self.w_hypertemplate.value != '' :
			global t
			t = linspace(0, 2*pi, self.w_npoints.value, endpoint=False)
			if self.w_manifold_type.value == "Torus" :
				code = "q0 = array({ht})".format(
					ht = self.w_hypertemplate.value
				)
			elif self.w_manifold_type.value == "Landmarks" :
				code = "q0 = (vstack({ht}).T).ravel()".format(
					ht = self.w_hypertemplate.value
				) 
			elif self.w_manifold_type.value == "Curves" :
				obj = eval(self.w_hypertemplate.value)
				if type(obj) is str :
					code = "q0 = Curve.from_file('"+obj+"', offset = [.3, -.7])\n"
				else :
					code = """q0 = Curve( (vstack({ht}).T).ravel(),
				array([ [i, i+1] for i in range(len(t) - 1) ]),
				2
			  )""".format(
						ht = self.w_hypertemplate.value
					) 
			show_code(code)
			exec('global q0; ' + code)
			self.w_choose_hypertemplate.disabled = True
			self.w_choose_data.disabled = False
	def choose_data(self, b):
		if self.w_data.value != '' :
			global t
			t = linspace(0, 2*pi, self.w_npoints.value, endpoint=False)
			if self.w_manifold_type.value == "Torus" :
				code = "Xt = vstack( {ht} )".format(
					ht = self.w_data.value
				)
			elif self.w_manifold_type.value == "Landmarks" :
				code = """Xt = vstack( tuple(  (vstack(x).T).ravel() 
				 for x in 
				 ({ht})  
				) )""".format(
					ht = self.w_data.value
				) 
			elif self.w_manifold_type.value == "Curves" :
				obj = eval(self.w_hypertemplate.value)
				if type(obj[0]) is str :
					code = """Xt = [  Curve.from_file(f) for f in 
			({ht})  
		 ] """.format(
						ht = self.w_data.value
					) 
				else :
					code = """Xt = [  Curve( (vstack(x).T).ravel() ,
				   array([ [i, i+1] for i in range(len(t) - 1) ]),
				   2
				 )
			for x in 
			({ht})  
		 ] """.format(
						ht = self.w_data.value
					) 
			show_code(code)
			exec('global Xt; ' + code)
			self.w_choose_data.disabled = True
			self.w_create_model.disabled = False
	def create_model(self, b):
		global model
		if hasattr(model, 'training_widget') : 
			model.training_widget.close()
			
		reg_template_gradient_str = ''
		
		if self.w_template_type.value == 'Free' :
			atlas_type = '         FreeAtlas'
			if self.w_reg_hypertemplate.value == 'Gaussian reglztion' :
				reg_template_gradient_str = "  reg_template_gradient = ('gaussian', {s})\n                          ".format( s = self.w_sigma_reg_hypertemplate.value )
		elif self.w_template_type.value == 'Shooted' :
			atlas_type = 'HypertemplateAtlas'
		
		if self.w_data_attachment.value == 'kernel_matching' :
			data_attachment = 'kernel_matchings(start_scale = {ss}, end_scale = {es} )'.format(ss = self.w_sigma_W_start.value, es = self.w_sigma_W_end.value)
		elif self.w_data_attachment.value == 'sinkhorn_matching' :
			data_attachment = 'sinkhorn_matchings( sinkhorn_options = None )'
		elif self.w_data_attachment.value == 'currents' :
			data_attachment = 'current_matchings(start_scale = {ss}, end_scale = {es} )'.format(ss = self.w_sigma_W_start.value, es = self.w_sigma_W_end.value)
		elif self.w_data_attachment.value == 'varifolds' :
			data_attachment = 'varifold_matchings(start_scale = {ss}, end_scale = {es} )'.format(ss = self.w_sigma_W_start.value, es = self.w_sigma_W_end.value)
		elif self.w_data_attachment.value == 'normal_cycles' :
			data_attachment = 'normalcycles_matchings(start_scale = {ss}, end_scale = {es} )'.format(ss = self.w_sigma_W_start.value, es = self.w_sigma_W_end.value)
		else :
			data_attachment = self.w_data_attachment.value
			
		code = ('model = ' + atlas_type + '''( Manifold        = M, 
                            DataAttachment  = M.{dist}, 
                            FrechetExponent = {frechexp}, 
                            weights         = ({g_V0}, {g_V}, {g_W}), 
                            nobs            = {nobs}, 
                            Q0              = q0 
                          {reg_template_gradient})'''
				   ).format(
			dist = data_attachment,
			frechexp = self.w_frechet_exponent.value,
			g_V0 = self.w_gamma_V0.value,
			g_V  = self.w_gamma_V.value,
			g_W  = self.w_gamma_W.value,
			nobs = len(Xt),
			reg_template_gradient = reg_template_gradient_str
		)
		
		show_code(code)
		exec('global model; ' + code)
		self.w_create_model.disabled = True
		self.w_train.disabled = False
	
	def update_algorithm_latex(self, nobodycares=None) :
		txt = lambda x : '\\text{' + x + '}'
		nline = '\\\\~~~~~~~~~~~~~~'
		
		grad_X0 = ''
		if self.w_X0_descent_mode.value == 'Fixed Stepsize' :
			grad_X0 = '\\nabla_{X_0} C ~\\big/~| \\nabla_{X_0} C |'
		elif self.w_X0_descent_mode.value == '1/nit' :
			grad_X0 = '\\nabla_{X_0} C ~\\big/~| n_{\\text{it}} \\cdot \\nabla_{X_0} C |'
		elif self.w_X0_descent_mode.value == 'Gradient' :
			grad_X0 = '\\nabla_{X_0} C'
		
		
		grad_Xi = ''
		if self.w_Xi_descent_mode.value == 'Fixed Stepsize' :
			grad_Xi = '\\nabla_{p_i} C ~\\big/~| \\nabla_{p_i} C |'
		elif self.w_Xi_descent_mode.value == '1/nit' :
			grad_Xi = '\\nabla_{p_i} C ~\\big/~| n_{\\text{it}} \\cdot \\nabla_{p_i} C |'
		elif self.w_Xi_descent_mode.value == 'Gradient' :
			grad_Xi = '\\nabla_{p_i} C'
		code =  txt('While $n_{\\text{it}} < ' + str(self.w_niterations.value) + '~$:') \
			+ nline + 'X_0 \\leftarrow X_0 + ' + '{0:.2f}'.format(self.w_X0_descent_speed.value) + '\\cdot' + grad_X0 \
			+ nline + 'p_i ~\\leftarrow p_i ~+ ' + '{0:.2f}'.format(self.w_Xi_descent_speed.value) + '\\cdot' + grad_Xi
		self.w_descent_algorithm_latex.value = '$$' + code + '$$'
	def train_model(self, b):
		descent_parameters = self.get_descent_parameters()
		global model
		code = '''model.train( Xt, 
             descent_parameters = {desc}, 
             nits = {nits}
           )'''.format(
			desc = '             '.join(json.dumps(descent_parameters, sort_keys=False, indent=4).splitlines(True)),
			nits = self.w_niterations.value)
		show_code(code)
		model.train(Xt, descent_parameters, nits = self.w_niterations.value, progressbar = self.w_iterations)
		self.w_train.disabled = True
		self.w_show.disabled = False
	def show_model(self, b):
		global model
		code = 'model.show()\n'
		show_code(code)
		model.show()
		self.w_show.disabled = True
		self.w_create_manifold.disabled = False
			
		
	def create_layout(self) :
		spaceright = '50px'
		b_manifold = HBox([self.w_manifold_type, VBox(
			[ HBox([self.w_donut_radius, self.w_donut_section , 
					 Label('', width=self.w_kernel_size.width)]),
			   HBox([self.w_npoints, self.w_kernel_type, self.w_kernel_size]) ]
					 ), Label('', width=spaceright), self.w_create_manifold] ,
			layout=Layout(width='100%', justify_content='center')
			)
		b_start_point = HBox([Label('Hypertemplate $X_{\\text{HT}}$ :', width='202px'),
							  VBox([self.w_options_hypertemplate, self.w_hypertemplate]),
							  Label('', width=spaceright),
							  self.w_choose_hypertemplate],
							  layout=Layout(width='100%', justify_content='center')
							)
		b_data = HBox([Label('Dataset $\\widetilde{X_i}$ :', width='202px'),
							  VBox([self.w_options_data, self.w_data]),
							  Label('', width=spaceright),
							  self.w_choose_data],
							  layout=Layout(width='100%', justify_content='center')
							)
		b_HT_X0 = HBox( [ Label('Template constr $X_{\\text{HT}} \\rightarrow X_0$ :', width='202px', padding='22px 0px 0px 0px'),
						  self.w_template_type,
						  VBox([HBox([self.w_reg_hypertemplate, 
						              self.w_sigma_reg_hypertemplate]),
						        HBox([self.w_gamma_V0, 
						              self.w_sigma_V0]) ])  ] )
		b_X0_X  = HBox( [ Label('Models shooting $X_{0} \\rightarrow X_i$ :', width='202px', padding='5px 0px 0px 0px'),
						  self.w_shooting_dim_constraint, 
						  self.w_gamma_V, 
						  self.w_frechet_exponent ] )
		b_X_Xt  = VBox([
				  HBox( [ Label('Data attachment $X_{i} \\rightarrow \\widetilde{X_i}$ :', width='202px', padding='5px 0px 0px 0px'),
						  self.w_data_attachment, 
						  self.w_gamma_W, 
						  Label('', width='148px', padding='5px 0px 0px 0px') ] ),
				  HBox( [ Label('', width='202px', padding='5px 0px 0px 0px'),
						  Label('Coarse to fine scheme :', width='148px', padding='5px 0px 0px 0px'), 
						  self.w_sigma_W_start, 
						  self.w_sigma_W_end ] )
					])
		b_energy_latex = HBox( [Label('', width='202px'), self.w_energy_latex ] )
		
		b_model = HBox( [VBox([b_HT_X0, b_X0_X, b_X_Xt, b_energy_latex]), 
						 Label('', width=spaceright),
						 self.w_create_model
						],
						layout = Layout(width='100%', justify_content='center'))
						
		# Descent parameters
		b_descent_X0 = HBox([ Label('Update of the template $X_0$', width='202px', padding='6px 0px 0px 0px'),
							  self.w_X0_gradient_distribution,
							  self.w_X0_descent_mode,
							  self.w_X0_descent_speed ])
		b_descent_Xi = HBox([ Label('Update of the models $X_i$', width='202px', padding='6px 0px 0px 0px'),
							  self.w_Xi_gradient_distribution,
							  self.w_Xi_descent_mode,
							  self.w_Xi_descent_speed ])
		b_descent_Ei = HBox([ Label('Update of the directions $E_j$', width='202px', padding='6px 0px 0px 0px'),
							  self.w_Ei_gradient_distribution,
							  self.w_Ei_descent_mode,
							  self.w_Ei_descent_speed ])
		b_descent_stopping = HBox([ Label('Stopping criterion', width='202px', padding='6px 0px 0px 0px'),
							  self.w_descent_stopping_criterion,
							  self.w_niterations,
							  self.w_descent_threshold ])
							  
		b_train = HBox([VBox([b_descent_X0, b_descent_Xi, b_descent_Ei, b_descent_stopping, 
						HBox([Label('', width='202px'), self.w_descent_algorithm_latex]) ]),
						Label('', width=spaceright),
						self.w_train],
						layout = Layout(width='100%', justify_content='center'))
		
		b_display = HBox([Label('Progress :', width='202px', padding='5px 0px 0px 0px'),
						self.w_iterations,
						Label('', width=spaceright),
						self.w_show],
						layout = Layout(width='100%', justify_content='center'))
		self.widget = VBox([b_manifold, b_start_point, b_data, b_model, b_train, b_display])
		
		
