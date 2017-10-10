# Main demo script. (see the __main__ section of the code)

# Import the relevant tools
import time                 # to measure performance
import numpy as np          # standard array library
import torch
from   torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

# No need for a ~/.theanorc file anymore !
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor

from input_output    import GridData, DisplayShoot
from shooting        import _Hqp, _HamiltonianShooting, _HamiltonianCarrying
from data_attachment import _data_attachment
from curve           import Curve

# Cost function and derivatives =================================================================

def _cost( q,p, xt_measure, connec, params ) :
	"""
	Returns a total cost, sum of a small regularization term and the data attachment.
	.. math ::
	
		C(q_0, p_0) = .01 * H(q0,p0) + 1 * A(q_1, x_t)
	
	Needless to say, the weights can be tuned according to the signal-to-noise ratio.
	"""
	s,r  = params                        # Deformation scale, Attachment scale
	q1 = _HamiltonianShooting(q,p,s)[0]  # Geodesic shooting from q0 to q1
	# To compute a data attachment cost, we need the set of vertices 'q1' into a measure.
	q1_measure  = Curve._vertices_to_measure( q1, connec ) 
	attach_info = _data_attachment( q1_measure,  xt_measure,  r )
	return [ .01* _Hqp(q, p, s) + 1* attach_info[0] , attach_info[1] ]

# The discrete backward scheme is automatically computed :
def _dcost_p( q,p, xt_measure, connec, params ) :
	"The gradients of C wrt. p_0 is automatically computed."
	return torch.autograd.grad( _cost(q,p, xt_measure, connec, params)[0] , p)

#================================================================================================

def VisualizationRoutine(Q0, params) :
	def ShootingVisualization(q,p,grid) :
		return _HamiltonianCarrying(q, p, grid, params[0])
	return ShootingVisualization

def perform_matching( Q0, Xt, params, scale_momentum = 1, scale_attach = 1) :
	"Performs a matching from the source Q0 to the target Xt, returns the optimal momentum P0."
	(Xt_x, Xt_mu) = Xt.to_measure()      # Transform the target into a measure once and for all
	connec = torch.from_numpy(Q0.connectivity).type(dtypeint) ; 
	# Declaration of variable types -------------------------------------------------------------
	# Cost is a function of 6 parameters :
	# The source 'q',                    the starting momentum 'p',
	# the target points 'xt_x',          the target weights 'xt_mu',
	# the deformation scale 'sigma_def', the attachment scale 'sigma_att'.
	q0    = Variable(torch.from_numpy(    Q0.points ).type(dtype), requires_grad=True)
	p0    = Variable(torch.from_numpy( 0.*Q0.points ).type(dtype), requires_grad=True )
	Xt_x  = Variable(torch.from_numpy( Xt_x         ).type(dtype), requires_grad=False)
	Xt_mu = Variable(torch.from_numpy( Xt_mu        ).type(dtype), requires_grad=False)
	
	# Compilation. Depending on settings specified in the ~/.theanorc file or explicitely given
	# at execution time, this will produce CPU or GPU code under the hood.
	def Cost(q,p, xt_x,xt_mu) : 
		return _cost( q,p, (xt_x,xt_mu), connec, params )
	
	# Display pre-computing ---------------------------------------------------------------------
	g0,cgrid = GridData() ; G0 = Curve(g0, cgrid )
	g0 = Variable( torch.from_numpy( g0 ).type(dtype), requires_grad = False )
	# Given q0, p0 and grid points grid0 , outputs (q1,p1,grid1) after the flow
	# of the geodesic equations from t=0 to t=1 :
	ShootingVisualization = VisualizationRoutine(q0, params) 
	
	# L-BFGS minimization -----------------------------------------------------------------------
	from scipy.optimize import minimize
	def matching_problem(p0) :
		"Energy minimized in the variable 'p0'."
		[c, info] = Cost(q0, p0, Xt_x, Xt_mu)
		
		matching_problem.Info = info
		if (matching_problem.it % 20 == 0):# and (c.data.cpu().numpy()[0] < matching_problem.bestc):
			matching_problem.bestc = c.data.cpu().numpy()[0]
			q1,p1,g1 = ShootingVisualization(q0, p0, g0)
			
			q1 = q1.data.cpu().numpy()
			p1 = p1.data.cpu().numpy()
			g1 = g1.data.cpu().numpy()
			
			Q1 = Curve(q1, connec) ; G1 = Curve(g1, cgrid )
			DisplayShoot( Q0, G0,       p0.data.cpu().numpy(), 
			              Q1, G1, Xt, info.data.cpu().numpy(),
			              matching_problem.it, scale_momentum, scale_attach)
		
		print('Iteration : ', matching_problem.it, ', cost : ', c.data.cpu().numpy(), 
		                                            ' info : ', info.data.cpu().numpy().shape)
		matching_problem.it += 1
		return c
	matching_problem.bestc = np.inf ; matching_problem.it = 0 ; matching_problem.Info = None
	
	optimizer = torch.optim.LBFGS(
					[p0],
					max_iter = 1000, 
					tolerance_change = .000001, 
					history_size = 10)
	#optimizer = torch.optim.Adam(
	#				[p0])
	time1 = time.time()
	def closure():
		optimizer.zero_grad()
		c = matching_problem(p0)
		c.backward()
		return c
	for it in range(100) :
		optimizer.step(closure)
	time2 = time.time()
	return p0, matching_problem.Info

def matching_demo(source_file, target_file, params, scale_mom = 1, scale_att = 1) :
	Q0 = Curve.from_file('data/' + source_file) # Load source...
	Xt = Curve.from_file('data/' + target_file) # and target.
	
	# Compute the optimal shooting momentum :
	p0, info = perform_matching( Q0, Xt, params, scale_mom, scale_att) 


if __name__ == '__main__' :
	plt.ion()
	plt.show()
	# N.B. : this minimalistic toolbox showcases the Hamiltonian shooting theory...
	#        To get good-looking matching results on a consistent basis, you should
	#        use a data attachment term which takes into account the orientation of curve
	#        elements, such as "currents" and "varifold" kernel-formulas, not implemented here.
	#matching_demo('australopithecus.vtk','sapiens.vtk', (.05,.2), scale_mom = .1,scale_att = .1)
	matching_demo('amoeba_1.png',        'amoeba_2.png', (.1,0), scale_mom = .1,scale_att = 0)
