# Import of the relevant tools
import time
import numpy as np
import theano
import theano.tensor as T
from   theano import pp, config


from plotly.tools import FigureFactory as FF
import plotly.graph_objs as go

from ..io.read_vtk import ReadVTK


from .theano_shapescarrier import TheanoShapesCarrier
from .surfaces_manifold import SurfacesManifold

class TheanoSurfacesCarrier(SurfacesManifold, TheanoShapesCarrier) :
	"""
	Surface + HamiltonianDynamics with control points.
	"""
	def __init__(self, *args, **kwargs) :
		"""
		Creates a TheanoSurfaces manifold.
		Compilation takes place here.
		"""
		TheanoShapesCarrier.__init__(self, *args, **kwargs)
	
	

