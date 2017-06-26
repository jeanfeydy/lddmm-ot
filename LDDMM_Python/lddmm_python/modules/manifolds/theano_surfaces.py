# Import of the relevant tools
import time
import numpy as np
import theano
import theano.tensor as T
from   theano import pp, config


from plotly.tools import FigureFactory as FF
import plotly.graph_objs as go

from ..io.read_vtk import ReadVTK

# a TheanoCurves manifold will be created from a regular Curve, 
# with  information about connectivity and number of points.
# Basically, a TheanoCurves is an efficient implementation of
# a shape orbit.
from .theano_shapes import TheanoShapes
from .surfaces_manifold import SurfacesManifold

class TheanoSurfaces(SurfacesManifold, TheanoShapes) :
	"""
	Surface + HamiltonianDynamics with dense momentum field.
	"""
	def __init__(self, *args, **kwargs) :
		"""
		Creates a TheanoSurfaces manifold.
		Compilation takes place here.
		"""
		TheanoShapes.__init__(self, *args, **kwargs)
	
	
