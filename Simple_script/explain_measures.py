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

EPS = 1


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
	Gamma = 5*Gamma
	points = [] ; connectivity = [] ; curr_id = 0
	Q_points,Q_weights = Q.to_measure()  ;  xtpoints = Xt.points # Extract the centers + areas
	for (a, mui, gi) in zip(Q_points, Q_weights, Gamma) :
		#gi = gi / mui # gi[j] = fraction of the mass from "a" which goes to xtpoints[j]
		for (seg, gij) in zip(Xt.connectivity, gi) :
			mass_per_line = 0.01
			if gij >= mass_per_line :
				nlines = np.floor(gij / mass_per_line)
				ts     = np.linspace(.5-.002*nlines, .5+.002*nlines, nlines)
				for t in ts :
					b = (1-t) * xtpoints[seg[0]] + t * xtpoints[seg[1]]
					points += [a, b]; connectivity += [[curr_id, curr_id + 1]]; curr_id += 2
	if len(connectivity) > 0 :
		Plan = Curve(np.vstack(points), np.vstack(connectivity))
		Plan.plot(ax, color = (.8,.9,1.,.5), linewidth = 1.5)

def DisplayMatrix(M, foldername) :
	Rmax = .45
	M=M.T
	scale = Rmax / (np.amax(np.sqrt(M)))
	
	fig = plt.figure(1, figsize = (M.shape[1],M.shape[0]), dpi=50); fig.clf(); ax = fig.add_subplot(1, 1, 1)
	ax.autoscale(tight=True)
	
	W = M.shape[1]
	H = M.shape[0]
	for i in range(W) :
		ax.plot( [0,W], [i+.5,i+.5] , color = (.9,.9,.9), linewidth = 2.)
	for j in range(H) :
		ax.plot( [j+.5,j+.5], [0,H] , color = (.9,.9,.9), linewidth = 2.)
	
	for (i,M_i) in enumerate(M) :
		for (j, M_ij) in enumerate(M_i) :
			circle = plt.Circle((.5+j, .5+i), scale * np.sqrt(M_ij), color=(.8,0.,0.),zorder=3)
			                     #color=scalarMap.to_rgba(np.sqrt(K_ij))  )
			ax.add_artist(circle)
	
	ax.axis([-1, M.shape[1], -1, M.shape[0]]) ; ax.set_aspect('equal') ; 
	ax.invert_yaxis(); ax.axis('off'); plt.draw() ; plt.pause(0.001)
	fig.savefig( foldername+'image_grid.png', dpi=fig.dpi, bbox_inches='tight' )
	
def DisplayFactorization(A, B, M, res, filename) :
	Rmax = .45
	
	scale_a = Rmax / (np.amax(np.sqrt(A)))
	scale_b = Rmax / (np.amax(np.sqrt(B)))
	scale_k  = Rmax / (np.amax(np.sqrt(M)))
	fig = plt.figure(3, figsize = (M.shape[1]+2,M.shape[0]+2), dpi=res); fig.clf(); ax = fig.add_subplot(1, 1, 1)
	ax.autoscale(tight=True)
	
	W = M.shape[1]
	H = M.shape[0]
	print(W,H)
	print(A)
	print(B)
	for (i,mu_i) in enumerate(A) :
		circle = plt.Circle((-.5, .5+i), scale_a*np.sqrt(mu_i), color=(.8,0.,0.) )
		ax.add_artist(circle)
		if i > 0 :
			print(W)
			ax.plot( [-1,W], [i,i] , color = (.9,.9,.9), linewidth = 1.)
	for (j,nu_j) in enumerate(B) :
		circle = plt.Circle((.5+j, -.5), scale_b*np.sqrt(nu_j), color=(0.,0.,.8) )
		ax.add_artist(circle)
		if j > 0 :
			ax.plot( [j,j], [-1,H] , color = (.9,.9,.9), linewidth = 1.)
	
	ax.plot( [-1,W], [0.001,0.001], color = (.9,.9,.9), linewidth = 2.)
	ax.plot( [0.001,0.001], [-1,H], color = (.9,.9,.9), linewidth = 2.)
	ax.plot( [-1,-1], [0.02,H],  color = (.9,.9,.9), linewidth = 3.)
	ax.plot( [0.02,W], [-1,-1],  color = (.9,.9,.9), linewidth = 3.)
	ax.plot( [-1,W,W], [H,H,-1],  color = (.9,.9,.9), linewidth = 3.)
	
	
	#cNorm      = colors.Normalize(vmin=-1., vmax=1.)
	#scalarMap  = cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('RdYlBu') )
	
	for (i,M_i) in enumerate(M) :
		for (j, M_ij) in enumerate(M_i) :
			circle = plt.Circle((.5+j, .5+i), scale_k * np.sqrt(M_ij), color=(.8,.9,1.))
			                     #color=scalarMap.to_rgba(np.sqrt(K_ij))  )
			ax.add_artist(circle)
	
	ax.axis([-1, M.shape[1], -1, M.shape[0]]) ; ax.set_aspect('equal') ; 
	ax.invert_yaxis(); ax.axis('off'); plt.draw() ; plt.pause(0.001)
	fig.savefig( filename, dpi=fig.dpi, bbox_inches='tight' )
	
	
	
def DisplayWireframe(Xc, X, mu, scale, foldername) :
	# Wireframe plot : -----------------------------------------------------------------------
	fig = plt.figure(1, figsize = (10,10), dpi=50); fig.clf(); ax = fig.add_subplot(1, 1, 1)
	ax.autoscale(tight=True)
	
	Xc.plot(ax, color=(.8,0.,0.), linewidth=1)
	ax.axis([0, 1, 0, 1]) ; ax.set_aspect('equal') ; plt.draw() ; 
	plt.axis('off')
	plt.pause(0.001)
	
	fig.savefig( foldername + 'curve_naked.png', bbox_inches='tight')
	# Wireframe plot : -----------------------------------------------------------------------
	fig = plt.figure(2, figsize = (10,10), dpi=50); fig.clf(); ax = fig.add_subplot(1, 1, 1)
	ax.autoscale(tight=True)
	
	scale = scale / np.amax(np.sqrt(mu))
	
	Xc.plot(ax, color=(.8,0.,0.), linewidth=1)
	for (i,(x_i,mu_i)) in enumerate(zip(X,mu)) :
		circle = plt.Circle(x_i, scale*np.sqrt(mu_i), color=(.8,0.,0.), zorder=3 )
		ax.add_artist(circle)
	
	ax.axis([0, 1, 0, 1]) ; ax.set_aspect('equal') ; plt.draw() ; 
	plt.axis('off')
	plt.pause(0.001)
	
	fig.savefig( foldername + 'curve_measure.png', bbox_inches='tight')
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

def Display( Q0, Xt ) :
	"Performs a matching from the source Q0 to the target Xt, returns the optimal momentum P0."
	(Xt_x, Xt_mu) = Xt.to_measure()      # Transform the target into a measure once and for all
	(Q0_x, Q0_mu) = Q0.to_measure()
	q0 = Q0.points ; p0 = np.zeros(q0.shape)    # Null initialization for the shooting momentum
	
	
	
	x_col = Q0_x.reshape((Q0_x.shape[0], 1, Q0_x.shape[1]))
	y_lin = Xt_x.reshape((1, Xt_x.shape[0], Xt_x.shape[1]))
	C = np.sum( (x_col - y_lin)**2 , 2 )
	
	C_2    = C[[3,2,1,0]]
	C_3    = C + C_2
	C_4    = np.ones(C.shape)
	C_5    = np.exp(C_2)
	C_6    = C[[2,1,0,3]]
	C_7    = C_3 + 2*C_6
	
	Costs = [C, C_2, C_3, C_4, C_5, C_6, C_7]
	for (i,Cost) in enumerate(Costs) :
		[A,B,K,Plan] = ot_matching(Cost, Q0_x, Q0_mu, Xt_x, Xt_mu)
		print(A)
		print(B)
		print(K)
		#DisplayMatrix( Q0_mu, Xt_mu, Plan, 100,
		#               'output/explain_ot/matrix_reg2_'+str(i+1)+'.png')
		
		               
	
	return 0
	

if __name__ == '__main__' :
	plt.ion()
	plt.show()
	
	A = Curve( np.array([
		[0.,0.],
		[-1.,-1.],
		[-1.,-3.],
		[-1.7,-3.5],
		[-1.5,-5]
		]), 
		np.array([[0,1],[1,2],[2,3],[3,4]]) )
		
	A.points = A.points / (arclength(A.points))
	A.points = A.points + [.55,.9]
	(A_x, A_mu) = A.to_measure()
	DisplayWireframe(A, A_x, A_mu, .02, 
		               'output/explain_measures/')
		               
	img = misc.imread('simple_image.png', flatten = True) # Grayscale
	img = (img.T[:, :])  / 255.
	DisplayMatrix(img, 'output/explain_measures/')
	

