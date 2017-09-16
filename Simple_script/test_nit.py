# Import the relevant tools
import time                 # to measure performance
import numpy as np          # standard array library
import theano               # Autodiff & symbolic calculus library :
import theano.tensor as T   #             - mathematical tools;
from   theano import config, printing #   - printing of the Sinkhorn error.


# Display routines :
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections  import LineCollection


# Input/Output routines =========================================================================
# from '.vtk' to Curves objects  ----------------------------------------------------------------
from pyvtk import VtkData
# from '.png' to level curves  ------------------------------------------------------------------
from skimage.measure import find_contours
from scipy import misc
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d

EPS = .01


def arclength_param(line) :
	"Arclength parametrisation of a piecewise affine curve."
	vel = line[1:, :] - line[:-1, :]
	vel = np.sqrt(np.sum( vel ** 2, 1 ))
	return np.hstack( ( [0], np.cumsum( vel, 0 ) ) )
def arclength(line) :
	"Total length of a piecewise affine curve."
	return arclength_param(line)[-1]
	
def resample(line, npoints) :
	"Resamples a curve by arclength through linear interpolation."
	s = arclength_param(line)
	f = interp1d(s, line, kind = 'linear', axis = 0, assume_sorted = True)
	
	p = f( np.linspace(0, s[-1], npoints) )
	connec = np.vstack( (np.arange(0, len(p) - 1), 
						 np.arange(1, len(p)    )) ).T
	if np.array_equal(p[0], p[-1]) : # i.e. p is a loop
		p = p[:-1]
		connec = np.vstack( (connec[:-1,:],  [len(p)-1, 0]) )
	return (p, connec)

def level_curves(fname, npoints = 200, smoothing = 10, level = 0.5) :
	"Loads regularly sampled curves from a .PNG image."
	# Find the contour lines
	img = misc.imread(fname, flatten = True) # Grayscale
	img = (img.T[:, ::-1])  / 255.
	img = gaussian_filter(img, smoothing, mode='nearest')
	lines = find_contours(img, level)
	
	# Compute the sampling ratio for every contour line
	lengths = np.array( [arclength(line) for line in lines] )
	points_per_line = np.ceil( npoints * lengths / np.sum(lengths) )
	
	# Interpolate accordingly
	points = [] ; connec = [] ; index_offset = 0
	for ppl, line in zip(points_per_line, lines) :
		(p, c) = resample(line, ppl)
		points.append(p)
		connec.append(c + index_offset)
		index_offset += len(p)
	
	size   = np.maximum(img.shape[0], img.shape[1])
	points = np.vstack(points) / size
	connec = np.vstack(connec)
	return Curve(points, connec)
# Pyplot Output =================================================================================

def GridData() :
	"Returns the coordinates and connectivity of the grid carried along by a deformation."
	nlines = 11 ; ranges = [ (0,1), (0,1) ] # one square = (.1,.1)
	np_per_lines = (nlines-1) * 4 + 1       # Supsample lines to get smooth figures
	x_l = [np.linspace(min_r, max_r, nlines      ) for (min_r,max_r) in ranges]
	x_d = [np.linspace(min_r, max_r, np_per_lines) for (min_r,max_r) in ranges]
	
	v = [] ; c = [] ; i = 0
	for x in x_l[0] :                    # One vertical line per x :
		v += [ [x, y] for y in x_d[1] ]  # Add points to the list of vertices.
		c += [ [i+j,i+j+1] for j in range(np_per_lines-1)] # + appropriate connectivity
		i += np_per_lines
	for y in x_l[1] :                    # One horizontal line per y :
		v += [ [x, y] for x in x_d[1] ]  # Add points to the list of vertices.
		c += [ [i+j,i+j+1] for j in range(np_per_lines-1)] # + appropriate connectivity
		i += np_per_lines
	
	return ( np.vstack(v), np.vstack(c) ) # (vertices, connectivity)
	
def ShowTransport( Q, Xt, Gamma, ax ) :
	"Displays a transport plan."
	points = [] ; connectivity = [] ; curr_id = 0
	Q_points,Q_weights = Q.to_measure()  ;  xtpoints = Xt.points # Extract the centers + areas
	for (a, mui, gi) in zip(Q_points, Q_weights, Gamma) :
		gi = gi / mui # gi[j] = fraction of the mass from "a" which goes to xtpoints[j]
		for (seg, gij) in zip(Xt.connectivity, gi) :
			mass_per_line = 0.01
			if gij >= mass_per_line :
				nlines = np.floor(gij / mass_per_line)
				ts     = np.linspace(.48, .52, nlines)
				for t in ts :
					b = (1-t) * xtpoints[seg[0]] + t * xtpoints[seg[1]]
					points += [a, b]; connectivity += [[curr_id, curr_id + 1]]; curr_id += 2
	if len(connectivity) > 0 :
		Plan = Curve(np.vstack(points), np.vstack(connectivity))
		Plan.plot(ax, color = (.8,.9,1.,.1), linewidth = 1)

def DisplayShoot(Q0, G0, p0, Q1, G1, Xt, info, it, scale_momentum, scale_attach, form='.svg') :
	"Displays a pyplot Figure and save it."
	# Figure at "t = 0" : -----------------------------------------------------------------------
	fig = plt.figure(1, figsize = (10,10), dpi=100); fig.clf(); ax = fig.add_subplot(1, 1, 1)
	ax.autoscale(tight=True)
	
	G0.plot(ax, color = (.8,.8,.8), linewidth = 1)
	Xt.plot(ax, color = (.85, .6, 1.))
	Q0.plot(ax)
	ax.quiver( Q0.points[:,0], Q0.points[:,1], p0[:,0], p0[:,1], 
	           scale = scale_momentum, color='blue')
	
	ax.axis([0, 1, 0, 1]) ; ax.set_aspect('equal') ; plt.draw() ; plt.pause(0.001)
	fig.savefig( 'output/momentum_' + str(it) + form )
	
	# Figure at "t = 1" : -----------------------------------------------------------------------
	fig = plt.figure(2, figsize = (20,20), dpi=100); fig.clf(); ax = fig.add_subplot(1, 1, 1)
	ax.autoscale(tight=True)
	
	if scale_attach == 0 : # Convenient way of saying that we're using a transport plan.
		ShowTransport( Q1, Xt, info, ax)
	else :                 # Otherwise, it's a kernel matching term.
		ax.imshow(info, interpolation='bilinear', origin='lower', 
				vmin = -scale_attach, vmax = scale_attach, cmap=cm.RdBu, 
				extent=(0,1, 0, 1)) 
	#G1.plot(ax, color = (.8,.8,.8), linewidth = 1)
	
	ruler = Curve([[.3-scale_momentum/2,.8],[.3+scale_momentum/2,.8]], [[0,1]])
	ruler.plot(ax, color = (1.,0.,0.), linewidth = 1)
	
	Xt.plot(ax, color = (.76, .29, 1.))
	Q1.plot(ax)
	
	ax.axis([-.5, 1.5, -.5, 1.5]) ; ax.set_aspect('equal') ; plt.draw() ; plt.pause(0.001)
	fig.savefig( 'output/model_' + str(it) + form )
# Curve representations =========================================================================

class Curve :
	"Encodes a 2D curve as an array of float coordinates + a connectivity list."
	def __init__(self, points, connectivity) :
		"points should be a n-by-2 float array, connectivity an nsegments-by-2 int array." 
		self.points       = points
		self.connectivity = connectivity
	
	def segments(self) :
		"Returns the list of segments the curve is made of."
		return np.array( [  [self.points[l[0]], self.points[l[1]]] for l in self.connectivity ] )
		
	def to_measure(self) :
		"""
		Outputs the sum-of-diracs measure associated to the curve.
		Each segment from the connectivity matrix self.c
		is represented as a weighted dirac located at its center,
		with weight equal to the segment length.
		"""
		segments = self.segments()
		centers = [         .5 * (  seg[0] + seg[1]      ) for seg in segments ]
		lengths = [np.sqrt(np.sum( (seg[1] - seg[0])**2 )) for seg in segments ]
		return ( np.array(centers), np.array(lengths) )
	
	@staticmethod
	def _vertices_to_measure( q, connec ) :
		"""
		Transforms a theano array 'q1' into a measure, assuming a connectivity matrix connec.
		It is the Theano equivalent of 'to_measure' : as theano only handles numeric arrays,
		it could not be implemented in a neat Object-Oriented fashion.
		"""
		a = q[connec[:,0]] ; b = q[connec[:,1]]
		# A curve is represented as a sum of diracs, one for each segment
		x  = .5 * (a + b)                 # Mean
		mu = T.sqrt( ((b-a)**2).sum(1) )  # Length
		return (x, mu)
		
	def plot(self, ax, color = 'rainbow', linewidth = 3) :
		"Simple display using a per-id color scheme."
		segs = self.segments()
		
		if color == 'rainbow' :   # rainbow color scheme to see pointwise displacements
			ncycles    = 5
			cNorm      = colors.Normalize(vmin=0, vmax=(len(segs)-1)/ncycles)
			scalarMap  = cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('hsv') )
			seg_colors = [ scalarMap.to_rgba( i % ((len(segs)-1)/ncycles) ) 
			               for i in range(len(segs)) ]
		else :                    # uniform color
			seg_colors = [ color for i in range(len(segs)) ] 
		
		line_segments = LineCollection(segs, linewidths=(linewidth,), 
		                               colors=seg_colors, linestyle='solid')
		ax.add_collection(line_segments)
		
	@staticmethod
	def from_file(fname) :
		if   fname[-4:] == '.png' :
			return level_curves(fname)
		elif fname[-4:] == '.vtk' :
			data = VtkData(fname)
			points = np.array(data.structure.points)[:,0:2] # Discard "Z"
			connec = np.array(data.structure.polygons)
			return Curve((points + 150)/300, connec)
			





# My .theanorc reads as follow :
# [nvcc]
# flags=-D_FORCE_INLINES
#
# [global]
# device=cuda
# floatX=float32
#
# The first section, copy-pasted from the "easy-install on Ubuntu", is supposed to fix a gcc bug.
# The second one allows me to use my GPU as the default computing device in float32 precision.
#
# On my Dell laptop, it is a GeForce GTX 960M with 640 Cuda cores and 2Gb of memory.


# Theano is a fantastic deep learning library : it transforms symbolic python code
# into highly optimized CPU/GPU binaries, which are called in the background seamlessly.
#
# We now show how to code a whole LDDMM pipeline into one (!!!) page of theano symbolic code.

# Part 1 : cometric on the space of landmarks, kinetic energy on the phase space (Hamiltonian)===


def _squared_distances(x, y) :
	"Returns the matrix of $\|x_i-y_j\|^2$."
	x_col = x.dimshuffle(0, 'x', 1)
	y_lin = y.dimshuffle('x', 0, 1)
	return T.sum( (x_col - y_lin)**2 , 2 )

def _k(x, y, s) :
	"Returns the matrix of k(x_i,y_j)."
	sq = _squared_distances(x, y) / (s**2)
	return T.pow( 1. / ( 1. + sq ), .25 )

def _cross_kernels(q, x, s) :
	"Returns the full k-correlation matrices between two point clouds q and x."
	K_qq = _k(q, q, s)
	K_qx = _k(q, x, s)
	K_xx = _k(x, x, s)
	return (K_qq, K_qx, K_xx)

def _Hqp(q, p, sigma) :
	"The hamiltonian, or kinetic energy of the shape q with momenta p."
	pKqp =  _k(q, q, sigma) * (p.dot(p.T))# Use a simple isotropic kernel
	return .5 * T.sum(pKqp)               # $H(q,p) = \frac{1}{2} * sum_{i,j} k(x_i,x_j) p_i.p_j$
    
    
# Part 2 : Geodesic shooting ====================================================================
# The partial derivatives of the Hamiltonian are automatically computed !
def _dq_Hqp(q,p,sigma) : 
	return T.grad(_Hqp(q,p,sigma), q)
def _dp_Hqp(q,p,sigma) :
	return T.grad(_Hqp(q,p,sigma), p)

def _hamiltonian_step(q,p, sigma) :
	"Simplistic euler scheme step with dt = .1."
	return [q + .1 * _dp_Hqp(q,p,sigma) ,  # See eq. 
			p - .1 * _dq_Hqp(q,p,sigma) ]

def _HamiltonianShooting(q, p, sigma) :
	"Shoots to time 1 a k-geodesic starting (at time 0) from q with momentum p."
	# Here, we use the "scan" theano routine, which  can be understood as a "for" loop
	result, updates = theano.scan(fn            = _hamiltonian_step,
								  outputs_info  = [q,p],
								  non_sequences = sigma,
								  n_steps       = 10            )  # We hardcode the "dt = .1"
	final_result = [result[0][-1], result[1][-1]]  # We do not store the intermediate results
	return final_result                            # and only return the final state + momentum



# Part 2bis : Geodesic shooting + deformation of the ambient space, for visualization ===========
def _HamiltonianCarrying(q0, p0, grid0, sigma) :
	"""
	Similar to _HamiltonianShooting, but also conveys information about the deformation of
	an arbitrary point cloud 'grid' in the ambient space.
	""" 
	def _carrying_step(q,p,g,s) :
		"Simplistic euler scheme step with dt = .1."
		return [q + .1 * _dp_Hqp(q,p, s),  p - .1 * _dq_Hqp(q,p, s), g + .1 * _k(g, q, s).dot(p)]
	# Here, we use the "scan" theano routine, which  can be understood as a "for" loop
	result, updates = theano.scan(fn            = _carrying_step,
								  outputs_info  = [q0,p0,grid0],
								  non_sequences = sigma,
								  n_steps       = 10           ) # We hardcode the "dt = .1"
	final_result = [result[0][-1], result[1][-1], result[2][-1]] # Don't store intermediate steps
	return final_result                            # return the final state + momentum + grid

# Part 3 : Data attachment ======================================================================

def _ot_matching(q1_x, q1_mu, xt_x, xt_mu, radius) :
	"""
	Given two measures q1 and xt represented by locations/weights arrays, 
	outputs an optimal transport fidelity term and the transport plan.
	"""
	# The Sinkhorn algorithm takes as input three Theano variables :
	c = _squared_distances(q1_x, xt_x) # Wasserstein cost function
	mu = q1_mu ; nu = xt_mu
	
	# Parameters of the Sinkhorn algorithm.
	epsilon            = (EPS)**2          # regularization parameter
	rho                = (.5) **2          # unbalanced transport (See PhD Th. of Lenaic Chizat)
	niter              = 50             # max niter in the sinkhorn loop
	tau                = -.8               # nesterov-like acceleration
	
	lam = rho / (rho + epsilon)            # Update exponent
	
	# Elementary operations .....................................................................
	def ave(u,u1) : 
		"Barycenter subroutine, used by kinetic acceleration through extrapolation."
		return tau * u + (1-tau) * u1 
	def M(u,v)  : 
		"$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
		return (-c + u.dimshuffle(0,'x') + v.dimshuffle('x',0)) / epsilon
	lse = lambda A    : T.log(T.sum( T.exp(A), axis=1 ) + 1e-6) # slight modif to prevent NaN
	
	# Actual Sinkhorn loop ......................................................................
	# Iteration step :
	def sinkhorn_step(u, v, foo) :
		u1=u # useful to check the update
		u = ave( u, lam * ( epsilon * ( T.log(mu) - lse(M(u,v))   ) + u ) )
		v = ave( v, lam * ( epsilon * ( T.log(nu) - lse(M(u,v).T) ) + v ) )
		err = T.sum(abs(u - u1))
		
		return (u,v,err), theano.scan_module.until(err < 1e-4) # "break" the loop if error < tol
		
	# Scan = "For loop" :
	err0 = np.arange(1, dtype=config.floatX)[0]
	result, updates = theano.scan( fn            = sinkhorn_step,            # Iterated routine
								   outputs_info  = [(0.*mu), (0.*nu), err0], # Starting estimates
								   n_steps       = niter                   # Number of iterations
								 )    
	U, V = result[0][-1], result[1][-1] # We only keep the final dual variables
	Gamma = T.exp( M(U,V) )             # Eventual transport plan g = diag(a)*K*diag(b)
	cost  = T.sum( Gamma * c )         # Simplistic cost, chosen for readability in this tutorial
	if True :
		print_err_shape = printing.Print('error  : ', attrs=['shape'])
		errors          = print_err_shape(result[2])
		print_err  = printing.Print('error  : ') ; err_fin  = print_err(errors[-1])
		cost += .00000001 * err_fin   # hack to prevent the pruning of the error-printing node...
	return [cost, Gamma]
def _kernel_matching(q1_x, q1_mu, xt_x, xt_mu, radius) :
	"""
	Given two measures q1 and xt represented by locations/weights arrays, 
	outputs a kernel-fidelity term and an empty 'info' array.
	"""
	K_qq, K_qx, K_xx = _cross_kernels(q1_x, xt_x, radius)
	q1_mu = q1_mu.dimshuffle(0,'x')  # column
	xt_mu = xt_mu.dimshuffle(0,'x')  # column
	cost = .5 * (   T.sum(K_qq * q1_mu.dot(q1_mu.T)) \
				 +  T.sum(K_xx * xt_mu.dot(xt_mu.T)) \
				 -2*T.sum(K_qx * q1_mu.dot(xt_mu.T))  )
				 
	# Info = the 2D graph of the blurred distance function
	res    = 10 ; ticks = np.linspace( 0, 1, res + 1)[:-1] + 1/(2*res) 
	X,Y    = np.meshgrid( ticks, ticks )
	points = T.TensorConstant( T.TensorType( config.floatX, [False,False] ) ,
							   np.vstack( (X.ravel(), Y.ravel()) ).T.astype(config.floatX) )
							   
	info   = _k( points, q1_x , radius ).dot(q1_mu) \
	       - _k( points, xt_x , radius ).dot(xt_mu)
	return [cost , info.reshape( (res,res) ) ]

def _data_attachment(q1_measure, xt_measure, radius) :
	"Given two measures and a radius, returns a cost - as a Theano symbolic variable."
	if radius == 0 : # Convenient way to allow the choice of a method
		return _ot_matching(q1_measure[0], q1_measure[1], 
								xt_measure[0], xt_measure[1], 
								radius)
	else :
		return _kernel_matching(q1_measure[0], q1_measure[1], 
								xt_measure[0], xt_measure[1], 
								radius)

# Part 4 : Cost function and derivatives ========================================================

def _cost( q,p, xt_measure, connec, params ) :
	"""
	Returns a total cost, sum of a small regularization term and the data attachment.
	.. math ::
	
		C(q_0, p_0) = .1 * H(q0,p0) + 1 * A(q_1, x_t)
	
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
	return T.grad( _cost(q,p, xt_measure, connec, params)[0] , p)

#================================================================================================

def VisualizationRoutine(Q0, params) :
	print('Compiling the ShootingVisualization routine.')
	time1 = time.time()
	q, p, grid = T.matrices('q', 'p', 'g') #  assign types to the teano variables
	ShootingVisualization = theano.function([q,p, grid],                                # input
										  _HamiltonianCarrying(q, p, grid, params[0]),  # output
										  allow_input_downcast=True)  # GPU = float32 only, 
                                  # whereas numpy uses float64 : we allow silent conversion
	time2 = time.time()   
	print('Compiled in : ', '{0:.2f}'.format(time2 - time1), 's')
	return ShootingVisualization

def perform_matching( Q0, Xt, params, scale_momentum = 1, scale_attach = 1) :
	"Performs a matching from the source Q0 to the target Xt, returns the optimal momentum P0."
	(Xt_x, Xt_mu) = Xt.to_measure()      # Transform the target into a measure once and for all
	(Q0_x, Q0_mu) = Q0.to_measure()
	q0 = Q0.points ; p0 = np.zeros(q0.shape)    # Null initialization for the shooting momentum
	
	# Compilation -------------------------------------------------------------------------------
	print('Compiling the energy functional.')
	time1 = time.time()
	# Cost is a function of 6 parameters :
	# The source 'q',                    the starting momentum 'p',
	# the target points 'xt_x',          the target weights 'xt_mu',
	# the deformation scale 'sigma_def', the attachment scale 'sigma_att'.
	q1_x, xt_x = T.matrices('q1_x', 'xt_x') ; q1_mu, xt_mu = T.vectors('q1_mu', 'xt_mu') #  assign types
	
	# Compilation. Depending on settings specified in the ~/.theanorc file or explicitely given
	# at execution time, this will produce CPU or GPU code under the hood.
	Cost  = theano.function([q1_x, q1_mu, xt_x,xt_mu ],
							   _ot_matching(q1_x, q1_mu, xt_x, xt_mu, 0),  
							allow_input_downcast=True)
	time2 = time.time()   
	print('Compiled in : ', '{0:.2f}'.format(time2 - time1), 's')
	
	# Display pre-computing ---------------------------------------------------------------------
	connec = Q0.connectivity ; q0 = Q0.points ; g0,cgrid = GridData() ; G0 = Curve(g0, cgrid )
	# Given q0, p0 and grid points grid0 , outputs (q1,p1,grid1) after the flow
	# of the geodesic equations from t=0 to t=1 :
	#ShootingVisualization = VisualizationRoutine(q0, params) 
	Gamma = Cost( Q0_x, Q0_mu, Xt_x, Xt_mu)[1]
	DisplayShoot( Q0, G0, p0, Q0, G0, Xt, Gamma, 
			              1, scale_momentum, scale_attach)

	
	return 0
def matching_demo(source_file, target_file, params, scale_mom = 1, scale_att = 1) :
	Q0 = Curve.from_file(source_file) # Load source...
	Xt = Curve.from_file(target_file) # and target.
	
	# Compute the optimal shooting momentum :
	t = perform_matching( Q0, Xt, params, scale_mom, scale_att) 

if __name__ == '__main__' :
	plt.ion()
	plt.show()
	#matching_demo('australopithecus.vtk','sapiens.vtk', (.05,.01), scale_mom = .3,scale_att = .1)
	matching_demo('amoeba_1.png',        'amoeba_2.png',(.05,  0), scale_mom = EPS, scale_att = 0)
	#matching_demo('australopithecus.vtk','sapiens.vtk', (.25,.01), scale_mom = 1.5,scale_att = .1)
	#matching_demo('australopithecus.vtk','sapiens.vtk', (.25,0), scale_mom = 1.5,scale_att = 0)


