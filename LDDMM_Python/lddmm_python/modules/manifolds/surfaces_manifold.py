# Import of the relevant tools
import time
import numpy as np
import theano
import theano.tensor as T
from   theano import pp, config


from plotly.tools import FigureFactory as FF
import plotly.graph_objs as go

from .surfaces import Surface

class SurfacesManifold :
	"""
	Encapsulates a few useful io + type conversion methods.
	"""
	
	def to_measure(self, q) :
		# Compute edges's vertices
		a = q[self.connectivity[:,0]]
		b = q[self.connectivity[:,1]]
		c = q[self.connectivity[:,2]]
		# A surface is represented as a sum of dirac, one for each triangle
		x  = .33333333 * (a + b + c)                 # Mean
		
		# Cross product
		ab = (b-a).dimshuffle(0, 1, 'x')
		ac = (c-a).dimshuffle(0, 'x', 1)
		
		t = (ab * ac).reshape((self.connectivity.shape[0], 9))
		cp = t.dot( np.array( [
					[0., 0.,  0.,  0., 0., 1., 0., -1., 0.],
					[0., 0., -1.,  0., 0., 0., 1.,  0., 0.],
					[0., 1.,  0., -1., 0., 0., 0.,  0., 0.]
					]
				).T)
		
		mu = .5 * T.sqrt( (cp**2).sum(1) )  # Length
		mu = T.cast(mu,  dtype=config.floatX)
		return (x, mu)
		
	def to_varifold(self, q) :
		# Compute edges's vertices
		a = q[self.connectivity[:,0]]
		b = q[self.connectivity[:,1]]
		c = q[self.connectivity[:,2]]
		# A surface is represented as a sum of dirac, one for each triangle
		x  = .33333333 * (a + b + c)                 # Mean
		
		# Cross product
		ab = (b-a).dimshuffle(0, 1, 'x')
		ac = (c-a).dimshuffle(0, 'x', 1)
		
		t = (ab * ac).reshape((self.connectivity.shape[0], 9))
		cp = t.dot( np.array( [
					[0., 0.,  0.,  0., 0., 1., 0., -1., 0.],
					[0., 0., -1.,  0., 0., 0., 1.,  0., 0.],
					[0., 1.,  0., -1., 0., 0., 0.,  0., 0.]
					]
				).T)
		
		mu = T.sqrt( (cp**2).sum(1) )  # Length
		u  = ( cp / mu.dimshuffle(0,'x'))    # Normal direction
		mu = T.cast(.5*mu,  dtype=config.floatX)
		u  = T.cast(u,   dtype=config.floatX)
		return (x, mu, u)
		
	def to_file(self, q, fname) :
		Q = Surface(q.ravel(), self.connectivity, q.shape[1])
		Q.to_file(fname)
		
		
	# Plot.ly visualization routines are implemented below ===================================================================
	def to_plot(self, q) :
		"""
		Uses self.connectivity to produce an appropriate list of coordinates.
		Returns an array of points which can be plotted by Plot.ly.
		Size of the return array ~= ((2+1)*nsegments)-by-dimension
		"""
		separator = np.array([None]* self.dimension)
		points = q
		ret = []
		for segment in self.connectivity :
			ret += [ points[segment[0]], points[segment[1]], separator ]
		return np.vstack(ret)
	
	def interactive_show(self, mode='', ax = None) :
		"Manifold display."
		self.layout = go.Layout(
			title='',
			width=800,
			height=800,
			legend = dict( x = .8, y = 1),
			xaxis = dict(range = [-2,2]),
			yaxis = dict(range = [-2,2])
		)
		
		
		
	def interactive_plot_traj(self, qts, **kwargs) :
		"""Trajectory display. qt can be an array of coordinates, or a list of such arrays."""
		if type(qts[0]) is not list :
			qts = [qts]
		points = []
		separator = np.array([None]* self.dimension).reshape((1,self.dimension))
		for qt in qts :
			for curve in qt :
				points.append( self.to_plot(curve) )
				points.append( separator )
		points = np.vstack(points)
		points = go.Scatter(x = np.array(points[:,0]), y = np.array(points[:,1]), mode = 'lines', hoverinfo='name', **kwargs)
		self.current_axis.append(points)
	def interactive_quiver(self, qt, vt, **kwargs) :
		"""Vector field display"""
		self.interactive_marker(qt, **kwargs)
		
	def interactive_marker(self, q, **kwargs) :
		"""Marker field display"""
		if type(q) is not list :
			q = [q]
		points = []
		separator = np.array([None]* self.dimension)
		for curve in q :
			points.append( self.to_plot(curve) )
			points.append( separator )
		points = np.vstack(points)
		points = go.Scatter(x = np.array(points[:,0]), y = np.array(points[:,1]), mode = 'lines', hoverinfo='name', **kwargs)
		self.current_axis.append(points)
	
	def interactive_marker_target(self, q, **kwargs) :
		"""Marker field display"""
		if type(q) is not list :
			q = [q]
		points = []
		separator = np.array([None]* self.dimension)
		for curve in q :
			points.append( curve.to_plot() )
			points.append( separator )
		points = np.vstack(points)
		points = go.Scatter(x = np.array(points[:,0]), y = np.array(points[:,1]), mode = 'lines', hoverinfo='name', **kwargs)
		self.current_axis.append(points)
		
	def interactive_show_transport(self, transports, **kwargs) :
		"""Display of a wasserstein transport plan."""
		raise(NotImplementedError)
		
		points = []
		separator = np.array([None]* self.dimension)
		for trans in transports :
			(gamma, q, xt) = trans
			Q  = q.to_measure()
			Xt = xt.to_measure()
			xtpoints = xt.to_array()
			for (a, mui, gi) in zip(Q.points, Q.weights, gamma) :
				# gi = sum(Q.weights) * gi / mui
				gi = gi / mui
				for (seg, gij) in zip(xt.connectivity, gi) :
					mass_per_line = 0.25
					if gij >= mass_per_line :
						#points += [a, b]
						#points.append( separator )
						nlines = floor(gij / mass_per_line)
						ts = linspace(.35, .65, nlines)
						for t in ts :
							b = (1-t) * xtpoints[seg[0]] + t * xtpoints[seg[1]]
							points += [a, b]
							points.append( separator )
		if points != [] :
			points = vstack(points)
			points = go.Scatter(x = array(points[:,0]), y = array(points[:,1]), mode = 'lines', hoverinfo='name', **kwargs)
		else :
			points = go.Scatter(x = array([0]), y = array([0]), mode = 'lines', hoverinfo='name', **kwargs)
		self.current_axis.append(points)
		
		
		
		
		
		
