from pylab import *


import plotly.graph_objs as go
from scipy.interpolate import interp1d
from plotly.offline import iplot, _plot_html
from IPython.display import HTML, display
from plotly.tools import FigureFactory as FF

from .riemannian_manifold import RManifold


class SurfaceOfRevolution(RManifold) :
	"Encodes a surface of revolution in R^3, typically a torus or a sphere."
	def __init__(self, R, Rp, Rpp, Z, Zp, D, vis_mode='3D') :
		"""
		Creates a Surface (d=2) of Revolution from function handles.
		
		Arguments :
		R 	-- @(r,t) -> R(r,t), the distance to z-axis
		Rp	-- its first derivative
		Rpp	-- its second derivative
		Z 	-- elevation function
		Zp 	-- derivative of the elevation function
		D 	-- periodic domain bounds, [[min_r, max_r], [min_t, max_t]]
		"""
		g = lambda q : array([[1, 0], [0, R(q[0])**2]])
		RManifold.__init__(self, 2, g)
		self.D = D
		self.R = R 
		self.Rp = Rp
		self.Rpp = Rpp
		self.Z = Z
		self.Zp = Zp
		
		self.vis_mode = vis_mode
		self.upsample_trajs = False
		
	def K(self,q,p, *args) :
		"""Overrides the generic kernel function for better efficiency.
		 K(r, theta) = [ 1 , 0       ]
		               [ 0, 1/R(r)^2 ]
		"""
		assert q.shape == (2, ), 'Wrong dimension of the starting point.'
		f = self.R(q[0])**2
		p = atleast_2d(p)
		#
		if len(p) == 1 :
			return array( (p[0,0], p[0,1] / f) )
		else :
			return atleast_2d(vstack((p[:,0], p[:,1] / f))).T
		
	def L2_repr_p(self,q,p, *args) :
		"""Mapping from the cotangent plane endowed with Kernel metric
		to R^2 endowed with the standard dot product.
		 K(r, theta)^.5 = [ 1 , 0     ]
		                  [ 0, 1/R(r) ]
		"""
		assert q.shape == (2, ), 'Wrong dimension of the starting point.'
		f = self.R(q[0])
		p = atleast_2d(p)
		return atleast_2d(vstack((p[:,0], p[:,1] / f))).T
		
	def upP(self,q,p, *args) :
		"""Returns an update step of the momentum p in the geodesic equations.
		- .5*d_(r,theta) (p, K_(r,theta) p) = [ p_theta^2 * R'(r) / R(r)^3 ]
		                                      [           0                ]
		"""
		return array( [ p[1]**2 * self.Rp(q[0]) / (self.R(q[0])**3) ,  0] )
	def gradq_pKqz(self, p, q, z, *args) :
		"""Useful for the adjoint integration scheme.
		d_(r,theta) (p, K_(r,theta) z) = [ -2*p_t * z_t * R'(r) / R(r)^3 ]
		                                 [              0                ]
		"""
		return array([ -2 * p[1] * z[1] * self.Rp(q[0]) /  (self.R(q[0])**3) , 0] )
	def dq_gradq_pKqp_a(self, q, p, a, *args) :
		"""Useful for the adjoint integration scheme."""
		r = q[0];
		return array([ -2 * a[0] * p[1]**2 * ( self.Rpp(r) * self.R(r) - 3 * self.Rp(r)**2 ) / self.R(r)**4 , 0] )
	def dq_Kqp_a(self,q,p,a, *args) :
		"""Useful for the adjoint integration scheme.
		d_(r,theta) (K_(r,theta) p) . a  = [                0                ]
		                                   [ -2*a_r p_theta * R'(r) / R(r)^3 ]
		"""
		return array( [0 , -2* a[0] * p[1] * self.Rp(q[0]) / (self.R(q[0])**3)] )
	
	def I(self, q=None, R=None, Theta=None) :
		"""Isometrically embeds a collection of points in the euclidean space (typically, R^2 -> R^3).
		Input points are identified 'modulo D'.
		
		Two usages :
			I(q=...),       with a 2-by-npoints array
			I(R=..., Theta=...), with two arrays of same shape
		"""
		if q is not None : # plotting a line
			q = atleast_2d(q)
			assert (q.shape[1] == self.d) , 'q does not have the right size - dim x npoints.'
			
			R     = q[:,0]
			Theta = q[:,1]
			
			return vstack( ( 	(self.R(R)) * cos(Theta) ,
								(self.R(R)) * sin(Theta) ,
								 self.Z(R)
								 ) ).T
		elif (R is not None) and (Theta is not None) :  # [X,Y,Z] = self.I(R, Theta)
			assert (R.shape == Theta.shape), 'I should be applied on two matrices of the same size'
			return ( (self.R(R)) * cos(Theta) , # X
					 (self.R(R)) * sin(Theta) , # Y
					  self.Z(R)					# Z
				)
		else :
			raise Exception ('Incorrect Usage.')
			
	def tangent_basis(self, q) :
		"""Returns the standard basis (e_r,e_t) in (R^3)x(R^3) at positions given by q."""
		q = atleast_2d(q)
		assert (q.shape[1] == self.d) , 'q does not have the right size - dim x npoints.'
		r = q[:,0]
		Theta = q[:,1]
		padding = zeros(Theta.shape)
		e_r = self.Rp(r) * vstack( ( cos(Theta), sin(Theta), padding )) \
			+ vstack( (padding, padding, self.Zp(r)) )
		e_t = self.R(r)  * vstack( (-sin(Theta), cos(Theta), padding ))
		return (e_r.T, e_t.T)
	def unit_tangent_basis(self, q) :
		"""Same as tangent_basis, but normalized wrt the dot product in R^3."""
		(e_cr, e_t) = self.tangent_basis(q)
		e_ct = e_t.T / self.R(q[:,0])
		return (e_cr, e_ct.T)
	def dI(self, q, v) :
		"""Differential of I at the points q, applied to v."""
		(e_r, e_t) = self.tangent_basis(q)
		return (atleast_2d(v[:,0]).T * e_r.T + atleast_2d(v[:,1]).T * e_t.T).T
	
	""" Distances """
	def squared_distance(self, Q, Xt, *args) :
		"""Returns 1/2 * |I(Q) - Xt|^2 and its Q-gradient."""
		X = self.I(q = Q)
		d2 = .5 * sum( (Xt - X)**2, 1)
		dX = .5 * 2 * (X - Xt)
		(e_cr, e_ct) = self.tangent_basis(Q)
		
		# NONONO ! We're not inverting the differential,
		# but *transposing* it : no need for renormalization !
		# n2_r = sum(e_cr**2, 1)
		# n2_t = sum(e_ct**2, 1)
		#dQ = vstack( (sum( e_cr * dX , 1) / n2_r,
		#			  sum( e_ct * dX , 1) / n2_t ) )
		dQ = vstack( (sum( e_cr * dX , 1),
					  sum( e_ct * dX , 1) ) )
		return (d2, dQ)
	def distance(self, Q, Xt, *args) :
		"""Returns |I(Q) - Xt| and its Q-gradient."""
		X = self.I(q = Q)
		Xt = Xt.reshape(X.shape) # In case of line/column confusion
		d = sqrt(sum( (Xt - X)**2, 1))
		dX = (X - Xt) / (d+0.00000001)
		(e_cr, e_ct) = self.tangent_basis(Q)
		n2_r = sum(e_cr**2, 1)
		n2_t = sum(e_ct**2, 1)
		dQ = vstack(( sum( e_cr * dX , 1) / n2_r,
					  sum( e_ct * dX , 1) / n2_t ) ) 
		return (d, dQ)
		
	"""Periodization & Co."""
	def periodize(self, q) :
		"""q is a n-by-d array of coordinates
		nq gives their representations in the fundamental domain
		as required by self.D """
		nq = q.astype(float) # We're using mod, so we have to be careful !
		assert(q.shape[1] == self.d)
		for d in range(self.d) :
			nq[:,d] = mod(nq[:,d]- self.D[d,0], self.D[d,1] - self.D[d,0]) + self.D[d,0]
		return nq

	def periodize_traj(self, qt) :
		"""qt is a 2xn trajectory
		   trajs is a list of trajectories on the rectangle domain"""
		pqt = self.periodize(qt)
		tile_dims =  self.D[:,1] - self.D[:,0]
		tiles = ( (qt - pqt) / tile_dims).round()
		cuts = tiles[1:-1,:] != tiles[0:-2,:]
		cuts = any(cuts, 1)
		cutlocs = concatenate( (find(cuts), [qt.shape[0]-1]) )
		ncuts = len(cutlocs)
		trajs = []
		ind = 0
		for i in range(ncuts) :
			to_concat = []
			if ind > 0 :
				to_concat.append( pqt[ind - 1] + tile_dims * (tiles[ind - 1] - tiles[ind ]) )
			to_concat.append( pqt[range(ind,cutlocs[i]+1)] )
			if cutlocs[i] < qt.shape[0]-1 :
				to_concat.append( (pqt[cutlocs[i] + 1] + tile_dims * (tiles[cutlocs[i] + 1] - tiles[cutlocs[i]])) )
				
			trajs += [vstack( to_concat )]
			ind = cutlocs[i] + 1
		return trajs
		
	def upsample(self, qt) : # !!! to be tested !!!
		"""upsample a trajectory by linear interpolation
		useful for 3D-plotting a not-so-well sampled trajectory"""
		if self.dt > 0.1 :
			#return numpy.interp(linspace(0, qt.shape[1]), range(qt.shape[1]), qt)
			f = interp1d( range(qt.shape[0]), qt , axis = 0)
			return f(linspace(0, qt.shape[0]-1, qt.shape[0]*round(self.dt / 0.001)))
		else :
			return qt
			
			
	""" Manifold display """
	def show(self, mode, ax=None) :
		self.vis_mode = mode
		if ax == None :
			ax = []
		self.layout = go.Layout(
			title='',
			width=800,
			height=800,
			legend = dict( x = .8, y = 1)
		)
		self.current_axis = ax
		if self.vis_mode == '2D' :
			self.layout['legend']['x'] = 1
			self.show_2D()
		elif self.vis_mode == '3D':
			self.show_3D()
	def show_2D(self) :
		# (r,theta) -> (y,x)
		self.layout['xaxis'] = dict( range = [-pi,pi])
									 #tickvals = [-pi,0,pi]
									 #ticktext = ['$-\\pi$', '$0$', '$\\pi$'] )
		self.layout['yaxis'] = dict( range = [-pi*self.b,pi*self.b])
									 #tickvals = [-pi*self.b,0,pi*self.b],
									 #ticktext = ['$-\\pi b$', '$0$', '$\\pi b$'] )
	def show_3D(self) :
		r  = linspace(self.D[0,0],self.D[0,1], 45)
		th = linspace(self.D[1,0],self.D[1,1], 45)
		
		(R, TH) = meshgrid(r, th)
		b_foo = self.b
		self.b = 0.99*self.b
		(X,Y,Z) = self.I(R = R, Theta = TH)
		self.b = b_foo
		
		surface = go.Surface(x=X, y=Y, z=Z,
                    opacity = 0.99,
                    colorscale = [[0, 'rgb(255,100,0)'], [1, 'rgb(255,255,0)']],
                    autocolorscale = False,
                    showscale = False,
                    hoverinfo = "none",
                    contours = {'x' : {'highlight' : False, 'highlightwidth' : 1},
                                'y' : {'highlight' : False, 'highlightwidth' : 1}, 
                                'z' : {'highlight' : False, 'highlightwidth' : 1}}
                    )
		self.layout['scene']['aspectmode'] = 'cube'
		m = 1.2 * (self.a + self.b)
		self.layout['scene']['xaxis'] = dict( range = [-m, m] )
		self.layout['scene']['yaxis'] = dict( range = [-m, m] )
		self.layout['scene']['zaxis'] = dict( range = [-m, m] )
		self.current_axis.append(surface)
	def plot_traj(self, qt, **kwargs) :
		if self.vis_mode == '2D' :
			trajs = self.periodize_traj(qt)
			for traj in trajs :
				# (r,theta) -> (y,x)
				curve = go.Scatter(x = traj[:,1], y = traj[:,0], mode = 'lines', hoverinfo='none', **kwargs)
				self.current_axis.append(curve)
		elif self.vis_mode == '3D' :
			if type(qt[0]) is not list :
				qt = [qt]
			if self.upsample_trajs :
				qt = list( self.upsample(q) for q in qt )
			traj = list( self.I(q = q) for q in qt )
			separator = array([None]* 3).reshape((1,3))
			traj = vstack( vstack((i, separator)) for i in traj )
			curve = go.Scatter3d(x = traj[:,0], y = traj[:,1], z = traj[:,2], mode = 'lines', hoverinfo='none', **kwargs)
			self.current_axis.append(curve)

	# Vector field display
	def quiver(self, qt, vt, **kwargs) :
		if self.vis_mode == '2D' :
			self.quiver_2D(qt, vt, **kwargs)
		elif self.vis_mode == '3D':
			self.quiver_3D(qt, vt, **kwargs)
	def quiver_2D(self, qt, vt, **kwargs) :
		# (r,theta) -> (y,x)
		qt = self.periodize(qt)
		arrows = FF.create_quiver(qt[:,1], qt[:,0], vt[:,1], vt[:,0], **kwargs)
		self.current_axis.append(arrows)
		
	def quiver_3D(self, qt, vt, **kwargs) :
		if qt.shape[1] == 2 :
			Qt = self.I(qt)
			Vt = self.dI(qt, vt)
		elif qt.shape[1] == 3 :
			Qt = qt
			Vt = vt
		
		# quiver3 is not implemented by plotly.js :
		# we have to settle for a poor derivative...
		H = Qt
		T = H + Vt
		arrows = go.Scatter3d(
			x = (hstack(tuple( (H[i,0], T[i,0], None) for i in range(T.shape[0]) ))),
			y = (hstack(tuple( (H[i,1], T[i,1], None) for i in range(T.shape[0]) ))),
			z = (hstack(tuple( (H[i,2], T[i,2], None) for i in range(T.shape[0]) ))),
			mode = 'lines',
			**kwargs
		)
		self.current_axis.append(arrows)
		
		
	"""Marker field display"""
	def marker(self, q, **kwargs) :
		q = atleast_2d(q)
		if self.vis_mode == '2D' :
			self.marker_2D(q, **kwargs)
		elif self.vis_mode == '3D' :
			self.marker_3D(q, **kwargs)
	def marker_2D(self, q, **kwargs) :
		# (r,theta) -> (y,x)
		Q = self.periodize(q)
		points = go.Scatter(x = array([Q[:,1]]), y = array([Q[:,0]]), mode = 'markers', hoverinfo='name', **kwargs)
		self.current_axis.append(points)
	def marker_3D(self, q, **kwargs) :
		if q.shape[1] == 2 :
			Q = self.I(q = q)
		elif q.shape[1] == 3 :
			Q = q
		points = go.Scatter3d(x = Q[:,0], y = Q[:,1], z = Q[:,2], mode = 'markers', hoverinfo='name', **kwargs)
		self.current_axis.append(points)
            
