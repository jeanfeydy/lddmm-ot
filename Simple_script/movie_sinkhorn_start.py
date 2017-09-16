# Import the relevant tools
import time                 # to measure performance
import numpy as np          # standard array library

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

EPS = .03


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
		Plan.plot(ax, color = (.8,.9,1.,.05), linewidth = 1.5)

def DisplayShoot(Q0, G0, p0, Q1, G1, Xt, scale_momentum, scale_attach, form='.png', info = None) :
	"Displays a pyplot Figure and save it."
	# Figure at "t = 1" : -----------------------------------------------------------------------
	fig = plt.figure(2, figsize = (20,20), dpi=50); fig.clf(); ax = fig.add_subplot(1, 1, 1)
	ax.autoscale(tight=True)
	
	"""if scale_attach == 0 : # Convenient way of saying that we're using a transport plan.
		ShowTransport( Q1, Xt, info, ax)
	else :                 # Otherwise, it's a kernel matching term.
		ax.imshow(info, interpolation='bilinear', origin='lower', 
				vmin = -scale_attach, vmax = scale_attach, cmap=cm.RdBu, 
				extent=(0,1, 0, 1)) """
	scale_attach = np.amax( np.abs(info) )
	if info is not None :
		ax.imshow(-info, interpolation='bilinear', origin='lower', 
				vmin = -scale_attach, vmax = scale_attach, cmap=cm.RdBu, 
				extent=(-.25, 1.25, -.25, 1.25))
	#G1.plot(ax, color = (.8,.8,.8), linewidth = 1)
	
	#ruler = Curve([[.3-scale_momentum/2,.8],[.3+scale_momentum/2,.8]], [[0,1]])
	#ruler.plot(ax, color = (1.,0.,0.), linewidth = 1)
	
	#Xt.plot(ax, color = (0., 0., .8), linewidth = 5)
	#Q1.plot(ax, color = (.8, 0., 0.), linewidth = 5)
	title = ""#'Sinkhorn Iteration ' + str(it).zfill(3)
	ax.text(0.5, 0., title, horizontalalignment='center', verticalalignment='center',\
	        transform = ax.transAxes, size=48)
	ax.axis([-.25, 1.25, -.25, 1.25]) ; ax.set_aspect('equal') ; plt.draw() ; 
	plt.axis('off')
	plt.pause(0.001)
	
	#fig.savefig( 'output/sinkhorn_start/sinkhorn_kernel' + form, bbox_inches='tight')
	fig.savefig( 'output/kernel_fidelity/large_kernel' + form, bbox_inches='tight')
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
			



# Part 3 : Data attachment ======================================================================
"""
def _ot_matching(q1_x, q1_mu, xt_x, xt_mu, radius) :
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
"""

def ot_matching(q1_x, q1_mu, xt_x, xt_mu, tau = -.8) :
	"Full Python implementation !"
	x_col = q1_x.reshape((q1_x.shape[0], 1, q1_x.shape[1]))
	y_lin = xt_x.reshape((1, xt_x.shape[0], xt_x.shape[1]))
	c = np.sum( (x_col - y_lin)**2 , 2 )
	mu = q1_mu ; nu = xt_mu
	
	# Parameters of the Sinkhorn algorithm.
	epsilon            = (EPS)**2          # regularization parameter
	rho                = (.5) **2          # unbalanced transport (See PhD Th. of Lenaic Chizat)
	niter              = 1000                # max niter in the sinkhorn loop
	#tau                = -.8               # nesterov-like acceleration
	
	lam = rho / (rho + epsilon)            # Update exponent
	
	# Elementary operations .....................................................................
	def ave(u,u1) : 
		"Barycenter subroutine, used by kinetic acceleration through extrapolation."
		return tau * u + (1-tau) * u1 
	def M(u,v)  : 
		"$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
		return (-c + u[:,np.newaxis] + v[np.newaxis]) / epsilon
	lse = lambda A    : np.log(np.sum( np.exp(A), axis=1 ) + 1e-6) # slight modif to prevent NaN
	
	# Actual Sinkhorn loop ......................................................................
	# Iteration step :
	def sinkhorn_step(u, v, foo) :
		u1=u # useful to check the update
		u = ave( u, lam * ( epsilon * ( np.log(mu) - lse(M(u,v))   ) + u ) )
		v = ave( v, lam * ( epsilon * ( np.log(nu) - lse(M(u,v).T) ) + v ) )
		err = np.sum(np.abs(u - u1))
		
		return u,v,err # "break" the loop if error < tol
		
	# Scan = "For loop" :
	u = 0.*mu; v = 0.*nu; err_it = 1;
	errs = []; gammas = []; costs = [];
	for it in range(niter) :
		gamma_it = np.exp( M(u,v) )
		cost     = np.sum( gamma_it * c ) + epsilon * np.sum( gamma_it * np.log(gamma_it + 1e-9) )
		print(cost)
		
		errs.append( err_it )
		gammas.append( gamma_it )
		costs.append( cost )
		
		(u,v,err_it) = sinkhorn_step( u, v, 0)
		if err_it < 1e-7 :
			break
		
		
	costs = np.array(costs)
	errs  = np.array(errs)
	return [gammas, costs, errs]
	
	

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

def k(x, y, r) :
	x_col = x.reshape((x.shape[0], 1, x.shape[1]))
	y_lin = y.reshape((1, y.shape[0], y.shape[1]))
	c = np.sum( (x_col - y_lin)**2 , 2 )
	
	#return 1/(c+r)**(.5)
	return np.exp(-c / r)
	

def perform_matching( Q0, Xt, params, scale_momentum = 1, scale_attach = 1) :
	"Performs a matching from the source Q0 to the target Xt, returns the optimal momentum P0."
	(Xt_x, Xt_mu) = Xt.to_measure()      # Transform the target into a measure once and for all
	(Q0_x, Q0_mu) = Q0.to_measure()
	q0 = Q0.points ; p0 = np.zeros(q0.shape)    # Null initialization for the shooting momentum
	
	#[gammas, costs, errs] = ot_matching(Q0_x, Q0_mu, Xt_x, Xt_mu, tau = 0.)
	
	# Display pre-computing ---------------------------------------------------------------------
	connec = Q0.connectivity ; q0 = Q0.points ; g0,cgrid = GridData() ; G0 = Curve(g0, cgrid )
	# Given q0, p0 and grid points grid0 , outputs (q1,p1,grid1) after the flow
	# of the geodesic equations from t=0 to t=1 :
	#ShootingVisualization = VisualizationRoutine(q0, params) 
	
	#DisplayShoot( Q0, G0, p0, Q0, G0, Xt, 0, scale_momentum, scale_attach)
	
	res    = 400 ; ticks = np.linspace( -.25, 1.25, res + 1)[:-1] + 1/(2*res) 
	X,Y    = np.meshgrid( ticks, ticks )
	points = np.vstack( (X.ravel(), Y.ravel()) ).T
							   
	#radius = (EPS**2)/2
	radius = .02
	#radius = .00008
	info   = k( points, Q0_x , radius ).dot(Q0_mu) \
	       - k( points, Xt_x , radius ).dot(Xt_mu)
	info = info.reshape( (res,res) )
	DisplayShoot( Q0, G0, p0, Q0, G0, Xt, scale_momentum, scale_attach, info = info)

	
	
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


