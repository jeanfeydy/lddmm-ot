from pylab import *
from .surface_of_revolution import SurfaceOfRevolution

class Torus(SurfaceOfRevolution) : 
	def __init__(self, a=2, b=1, vis_mode='3D') :
		"""A Torus, seen as a Riemannian Manifold.
		
		Arguments :
		a 				-- radius of the donut
		b 				-- radius of its section
		vis_mode		-- '2D', '3D' : mode for displaying results
		"""
		R = lambda r : a + b * cos(r / b)
		Rp = lambda r : -sin(r / b)
		Rpp = lambda r : - cos(r / b) / b
		Z  = lambda r : b * sin(r / b)
		Zp = lambda r : cos(r / b)
		
		D = array([[-b*pi,b*pi],
				   [-pi, pi ]])
		
		SurfaceOfRevolution.__init__(self, R, Rp, Rpp, Z, Zp, D, vis_mode)
		
		self.a = a
		self.b = b
		
	
	""" Projection & Distances """
	def projection_onto(self, X) :
		"""Orthogonal projection from R^3 to the torus."""
		x = X[:,0]
		y = X[:,1]
		z = X[:,2]
		Theta = angle(x + 1j * y)
		Pl = sqrt(x**2 + y**2) - self.a
		Xhi = angle( Pl + 1j * z)
		r = Xhi / self.b
		return vstack((r , Theta))
	def squared_distance_from(self, Xt) :
		""".5*(d(Xt, Torus)^2) from R^3 to the torus."""
		Q = self.projection_onto(Xt)
		X = self.I(q = Q)
		return .5*sum( (Xt - X)**2, 1)
	
