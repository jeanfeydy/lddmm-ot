from pylab import *
from pprint import pprint
import plotly.graph_objs as go

from .model import Model

class FrechetMean(Model):
	"""Simple Frechet Mean - L2 or L1 depending on the cost function
	N.B.: this is a Frechet mean according to the "embedding" metric.
	To get an intrinsic Frechet mean, with geodesic distance,
	one needs to solve an atlas estimation problem."""
	def __init__(self, M, C, nobs, s, Q0 = array([0,0])) :
		Model.__init__(self, M, C, nobs, s)
		self.Q0 = Q0
		
	def training_step(self, Xt) :
		(C, dQ) = self.C(vstack((self.Q0,)*Xt.shape[1]).T, Xt) # Compute gradients
		dQ =  - self.s * mean(dQ, 1) # Rescale and Aggregate
		self.Q0 = self.Q0 + dQ       # Update
		
		# Display
		self.M.marker(self.Q0, marker = dict(size = 20, color='red'), name='Frechet Mean', visible=False)
		self.show_data_attachment(C) # 'Targets' + 'Distances'
		
		# We use a simple hide/show scheme for the plot updates
		frame = [dict(visible = False), dict(visible = True)]
		return (self.Q0,C,dQ, frame)
	def situation(self, Xt) :
		Q = vstack((self.Q0,)*Xt.shape[1]).T
		(C, dQ) = self.C(Q, Xt)
		return (Q,C,dQ)
	def get_frame(self, f) :
		# Three plotly traces per frame : 'Frechet Mean', 'Targets', 'Distances'
		list1 = str([ 1 + 3*self.current_frame, 2 + 3*self.current_frame, 3 + 3*self.current_frame])[1:-1]
		self.current_frame = f
		list2 = str([ 1 + 3*self.current_frame, 2 + 3*self.current_frame, 3 + 3*self.current_frame])[1:-1]
		return (self.frames[f] , [list1, list2])
