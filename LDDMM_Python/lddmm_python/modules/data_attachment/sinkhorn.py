from pylab import *
from collections import namedtuple

import theano
import theano.tensor as T
from theano import config, printing
from theano import OpFromGraph # We'll use a dev version which allows grad overrides and inlining (for GPU integration), see pull request 5255


SinkhornOptions = namedtuple('SinkhornOptions', 'epsilon niter tau rho dual_cost discard_entropy discard_KL grad_hack display_error')


# Symbolic Theano method ==========================================================

def _sinkhorn_log(Mu, Nu, C, options) :
	"""
	Theano symbolic version.
	Note that the gradient wrt Mu, Nu and C is computed as if
	the transport plan "gamma(Mu,Nu,C)" was piecewise constant.
	"""
	# First, load the parameters.
	epsilon            = options.epsilon   # regularization parameter
	niter              = options.niter     # max niter in the sinkhorn loop
	tau                = options.tau       # use for acceleration
	rho                = options.rho       # parameter for unbalanced transport 
	use_dual_cost      = options.dual_cost # If False, use the primal cost
	discard_entropy    = options.discard_entropy # If True + primal cost, remove the -eps*H(gamma)
	discard_KL         = options.discard_KL      # If True + primal cost, remove the rho*KL(...)
	grad_override_hack = options.grad_hack
	display_error      = options.display_error
	
	# Update exponent :
	if rho == inf : # balanced transport : no mass creation is allowed.
		lam = 1.
	else :          # lam = 1 / (1 + epsilon/rho)
		lam = rho / (rho + epsilon)
	
	
	# First, define the transport plan theano "Op" ---------------------------------
	# it takes as input three Theano variables :
	if grad_override_hack :
		mu = T.vector('mu')
		nu = T.vector('nu')
		c  = T.matrix('c')
	else :
		mu = Mu
		nu = Nu
		c  = C
	
	# Elementary operations ..................................................
	def ave(u,u1,it) : 
		"""
		Barycenter subroutine, used by kinetic acceleration through extrapolation.
		tau = 0 -> returns u1.
		tau < 0 -> returns an extrapolation coming from u.
		
		Note that doing it on the "exponentiated" variables would not make any sense.
		"""
		t = tau #t = (1. - 1./((it+2.)**2)) * tau
		return t * u + (1-t) * u1 
	def M(u,v)  : 
		"""
		M_ij = (-c_ij + u_i + v_j) / epsilon
		"""
		u_col = u.dimshuffle(0,'x') # theano syntax to make a vector broadcastable in the 2nd dimension
		v_row = v.dimshuffle('x',0) # theano syntax to make a vector broadcastable in the 1st dimension
		return (-c + u_col + v_row) / epsilon
	lse = lambda A    : T.log(T.sum( T.exp(A), axis=1 ) + 1e-6) # slight modif to prevent NaN
	
	# Actual Sinkhorn loop ..................................................
	# Iteration step :
	def sinkhorn_step(nit, u, v, foo) :
		u1=u # useful to check the update
		u = ave(u, lam * ( epsilon * ( T.log(mu) - lse(M(u,v))   ) + u ) , nit[0])
		v = ave(v, lam * ( epsilon * ( T.log(nu) - lse(M(u,v).T) ) + v ) , nit[0])
		if rho == inf :
			err = T.sum(abs( T.sum(T.exp(M(u,v)), 1) - mu ) )
		else :
			err = T.sum(abs(u - u1))
			
		return (u,v,err), theano.scan_module.until(err < 1e-4) # "break" the scan loop if error < tol
		
	# Scan = "For loop" :
	iternumbers = np.arange(niter, dtype=config.floatX)
	iternumbers = stack( (iternumbers, iternumbers), 1 )
	"""
	result, updates = theano.scan_checkpoints(fn            = sinkhorn_step,              # Iterated routine
											  sequences     = [iternumbers],
											  outputs_info  = [(0. * mu), (0. * nu)],     # Starting estimates for [u,v]
											  save_every_N  = niter, padding=False )      # Efficient memory management, at an additional computational cost
	"""
	err0 = np.arange(1, dtype=config.floatX)[0]
	result, updates = theano.scan(            fn            = sinkhorn_step,               # Iterated routine
											  sequences     = [iternumbers],
											  outputs_info  = [(0. * mu), (0. * nu), err0] # Starting estimates for [u,v]
											  #n_steps       = niter                       # Number of iterations
											  )    
	
	
	u, v = result[0][-1], result[1][-1] # We only keep the final dual variables
	gamma = T.exp( M(u,v) )             # Eventual transport plan g = diag(a)*K*diag(b)
	
	# Gradient override .....................................................
	
	if grad_override_hack : # We give U,V,Gamma, albeit with a "hacked" explicit (i.e. autodiff-free) derivative
		# HERE, WE USE A DEV VERSION which allows :
		# - grad overrides
		# - inlining (for GPU integration)
		# See pull request 5255 on Theano's Github.
		if use_dual_cost :
			hack_derivative = lambda x,g : [0*x[0],0*x[1],0*x[2]]
			_transport_plan = OpFromGraph([mu, nu, c    ], 
										  [ u,  v, gamma],
										  inline = True,
										  grad_overrides = hack_derivative
										  )
			U,V,Gamma = _transport_plan(Mu, Nu, C)
		else :
			null_derivative = lambda x,g : [0*x[0],0*x[1],0*x[2]]
			_transport_plan = OpFromGraph([mu, nu, c    ], 
										  [gamma],
										  inline = True,
										  grad_overrides = null_derivative
										  )
			Gamma = _transport_plan(Mu, Nu, C)
	else :
		U,V,Gamma = u,v,gamma
		
	# Final cost computation .................................................
	if use_dual_cost :
		"""
		print_U  = printing.Print('U  : ', attrs = [ 'shape' ]) ; U  = print_U(U)
		print_Mu = printing.Print('Mu : ', attrs = [ 'shape' ]) ; Mu = print_Mu(Mu)
		print_V  = printing.Print('V  : ', attrs = [ 'shape' ]) ; V  = print_V(V)
		print_Nu = printing.Print('Nu : ', attrs = [ 'shape' ]) ; Nu = print_Nu(Nu)
		print_G  = printing.Print('G  : ', attrs = [ 'shape' ]) ; Gamma = print_G(Gamma)
		"""
		if grad_override_hack : # allow the first term to have a derivative wrt x
			
			plan        = T.matrix('plan')
			cost_matrix = T.matrix('cost_matrix')
			virtual_cost = T.sum( plan * cost_matrix )
			#hack_derivative = lambda x,g : [ 0 * x[0], T.grad(virtual_cost,
			_firstterm = OpFromGraph( [plan, cost_matrix ],
									  [- epsilon * T.sum(plan)],
									  inline = True,
									  grad_overrides = hack_derivative )
			cost = _firstterm(Gamma, C)
		else :
			cost = - epsilon * T.sum( Gamma )
			
		if rho == inf :
			cost += T.sum(Mu * U) + T.sum(Nu * V)
		else :
			cost += - rho * (T.sum( Mu * (T.exp( -U / rho ) - 1) ) \
						   + T.sum( Nu * (T.exp( -V / rho ) - 1) ) )
	else :
		xlogx  = lambda x   : x * T.log(x + 1e-6)
		xlogy0 = lambda x,y : x * T.log(y + 1e-6)
		H      = lambda g   : - T.sum( xlogx(g) - g )
		# Primal :
		if discard_entropy :
			cost  = T.sum(  Gamma * C )
		else :
			cost  = T.sum(  Gamma * C ) - epsilon * H(Gamma)
		
		KL = lambda h, p : T.sum( xlogy0(h, h/p) - h + p )
		if rho != inf and not discard_KL :
			# We add the KL divergences
			KL_1 = KL( T.sum(Gamma, 1), Mu)
			KL_2 = KL( T.sum(Gamma, 0), Nu)
			cost += rho * (KL_1 + KL_2)
			
			
	
	if display_error :
		print_err_shape  = printing.Print('error  : ', attrs=['shape']) ; errors = print_err_shape(result[2])
		print_err  = printing.Print('error  : ') ; err_fin  = print_err(errors[-1])
		cost += .00000001 * err_fin # shameful hack to prevent the pruning of the error-printing node...
	
	
	return [cost, Gamma]
	


# Legacy Python method ==========================================================

def sinkhorn_log(mu, nu, c, options, warm_restart = None) :
	"""
	"""
	epsilon = options.epsilon
	niter = options.niter
	tau   = options.tau
	rho   = options.rho
	
		
	if rho == inf :
		lam = 1.
	else :
		lam = rho / (rho + epsilon)
	
	H1 = ones(len(mu))
	H2 = ones(len(nu))
	ave = lambda tau, u, u1 : tau * u + (1-tau) * u1
	lse = lambda A : log(sum( exp(A), 1 ))
	M   = lambda u,v : (-c + atleast_2d(u).T @ atleast_2d(H2) + atleast_2d(H1).T @ atleast_2d(v)) / epsilon
	
	err = []
	if warm_restart is None :
		u = zeros(len(mu))
		v = zeros(len(nu))
	else :
		u = warm_restart[0]
		v = warm_restart[1]
		
	for i in range(niter) :
		u1 = u
		u = ave(tau, u, 
			    lam * ( epsilon * ( log(mu) - lse(M(u,v)  ) ) + u ) )
		v = ave(tau, v, 
			    lam * ( epsilon * ( log(nu) - lse(M(u,v).T) ) + v ) )
		if rho == inf :
			err.append( norm( sum(exp(M(u,v)), 1) - mu) )
		else :
			err.append( sum( abs(u - u1)) )
		if err[-1] < 1e-16 :
			break
	gamma = exp( M(u,v) )
	
	
	
	# The cost value is needed as soon as you do some kind of line search
	# Dual :
	cost = - epsilon * sum(exp( M(u,v) ))
	if rho == inf :
		cost += sum(sum(gamma, 1) * u) + sum(sum(gamma, 0) * v)
		#print('"Balanced" dual cost   : ', cost)
	else :
		cost += - rho * (sum( mu * (exp( -u / rho ) - 1) ) + sum( nu * (exp( -v / rho ) - 1) ) )
		#print('Unbalanced dual cost   : ', cost)
	# Primal :
	xlogx = lambda x : x * log(x + 1e-10)
	cost = sum(  gamma * c + epsilon * (xlogx(gamma) - gamma) ) 
	#print('"Balanced" primal cost : ', cost)
	
	KL = lambda h, p : sum( h * log( (h + 1e-10)/p) - h + p )
	if rho != inf :
		# We add the KL divergences
		KL_1 = KL( sum(gamma, 1), mu)
		KL_2 = KL( sum(gamma, 0), nu)
		cost += rho * (KL_1 + KL_2)
		#print('Unbalanced primal cost : ', cost)
	return (u, v, gamma, cost, err)










