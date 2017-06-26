from pylab import *
from plotly.tools import FigureFactory as FF
import plotly.graph_objs as go
from scipy.spatial.distance import pdist, squareform, cdist

from pyvtk import *
from ..io.read_vtk import ReadVTK

from .landmarks import Landmarks
from ..data_attachment.measures import Measures, Measure
from ..data_attachment.currents import Currents, Current
from ..data_attachment.varifolds import Varifolds, Varifold
from ..data_attachment.normal_cycles import NormalCycles, NormalCycle, Cylinders, Spherical

class Surface :
	"""
	Encodes a 2D/3D surface.
	Pythonic class which is especially useful thanks to its io methods.
	"""
	def __init__(self, points, connectivity, dimension) :
		assert (dimension == 3), ""
		assert isvector(points), "points should be a npoints*dimension vector."
		self.points = points
		self.connectivity = connectivity
		self.dimension = dimension
		
	@staticmethod
	def from_file(fname) :
		a = ReadVTK(fname)
		points = (array(a[0])[:,0:3])
		connec = array(a[1])
		return Surface(points.ravel(), connec, 3)
	def to_file(self, fname) :
		points = self.to_array()
		vtk = VtkData( PolyData(points = points, polygons = self.connectivity))
		vtk.tofile(fname, 'ascii')
	
	def mean_std(self) :
		"Returns the standard deviation of the mass repartition, which is useful in scripts."
		M = self.to_measure()
		w = (M.weights / sum(M.weights))[:, newaxis]
		points = M.points
		moy    = sum(multiply(points, w), 0)
		return (moy, sqrt( sum( ( (points - moy)**2) * w) ) )
	def translate_rescale(self, m, s) :
		points = self.to_array()
		points = (points - m) / s
		self.points = points.ravel()
		
	def normalize(self) :
		m, s = self.mean_std()
		self.translate_rescale(m,s)
	
	"Operations used to update current models."
	def __add__(self, dv) :
		return Surface(self.points + dv.points, self.connectivity, self.dimension)
	def __sub__(self, dv) :
		if self.points.shape == dv.points.shape :
			return Surface(self.points - dv.points, self.connectivity, self.dimension)
		else :
			return None
	def __rmul__(self, dt) :
		return Surface(dt * self.points, self.connectivity, self.dimension)
	def scale(self, weights) :
		"Row wise multiplication, useful for pointwise density normalization."
		return Surface((multiply(self.to_array(), weights[:, np.newaxis])).ravel(), self.connectivity, self.dimension)
	def __matmul__(self, curve2) :
		"Used in the norm computation..."
		return sum(self.points * curve2.points)
	def __truediv__(self, n) :
		return Surface(self.points / n , self.connectivity, self.dimension)
	def __neg__(self) :
		return Surface(-self.points, self.connectivity, self.dimension)
	def __pos__(self) :
		return Surface(self.points, self.connectivity, self.dimension)
	def array_shape(self) :
		return ( int(self.points.size / self.dimension), self.dimension)
	def to_array(self) :
		"""
		Reshapes self.points from vector to npoint-by-dimension array.
		"""
		return self.points.reshape(self.array_shape()) # This is a view, not a copy !!
	def ravel(self) :
		return self
	def to_measure(self) :
		"""
		Outputs the sum-of-diracs measure associated to the surface.
		Each triangle from the connectivity matrix self.c
		is represented as a weighted dirac located in its center,
		with weight equal to the triangle length.
		"""
		points = self.to_array()
		centers = zeros((len(self.connectivity), self.dimension))
		lengths = zeros(len(self.connectivity))
		for (i, triangle) in enumerate(self.connectivity) :
			a = points[triangle[0]]
			b = points[triangle[1]]
			c = points[triangle[2]]
			centers[i] =           (a+b+c ) / 3
			ab = b-a
			ac = c-a
			
			cross_prod = array([ ab[1]*ac[2] - ab[2]*ac[1],
			                     ab[2]*ac[0] - ab[0]*ac[2],
			                     ab[0]*ac[1] - ab[1]*ac[0] ])
			lengths[i] = .5*sqrt(sum( cross_prod**2 ) )
		return Measure( centers, lengths )
		
	def to_varifold(self) :
		"""
		Outputs the varifold measure associated to the curve.
		Each segment [a, b] is represented as a weighted dirac at
		the location ( (a+b)/2, b-a ) \in R^n x G_2(R^n),
		with weight equal to the segment length.
		"""
		points = self.to_array()
		centers = zeros((len(self.connectivity), self.dimension))
		normals = zeros((len(self.connectivity), self.dimension))
		lengths = zeros(len(self.connectivity))
		for (i, triangle) in enumerate(self.connectivity) :
			a = points[triangle[0]]
			b = points[triangle[1]]
			c = points[triangle[2]]
			centers[i] =           (a+b+c ) / 3
			ab = b-a
			ac = c-a
			
			cross_prod = array([ ab[1]*ac[2] - ab[2]*ac[1],
			                     ab[2]*ac[0] - ab[0]*ac[2],
			                     ab[0]*ac[1] - ab[1]*ac[0] ])
			lengths[i] = .5*sqrt(sum( cross_prod**2 ) )
			normals[i] = cross_prod / (2*lengths[i])
		
		return Varifold( centers, normals, lengths )
		
