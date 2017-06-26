from numpy import *
from skimage.measure import find_contours
from scipy import misc
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d

from ..manifolds.curves import Curve


def arclength_param(line) :
	vel = line[1:, :] - line[:-1, :]
	vel = sqrt(sum( vel ** 2, 1 ))
	return hstack( ( [0], cumsum( vel, 0 ) ) )
def arclength(line) :
	return arclength_param(line)[-1]
	
def resample(line, npoints) :
	s = arclength_param(line)
	f = interp1d(s, line, kind = 'linear', axis = 0, assume_sorted = True)
	
	t = linspace(0, s[-1], npoints)
	p = f(t)
	
	connec = vstack( (arange(0, len(p) - 1), arange(1, len(p)) ) ).T
	return (p, connec)

def level_curves(fname, npoints, smoothing = 10, level = 0.5) :
	# Find the contour lines
	img = misc.imread(fname, flatten = True) # Grayscale
	img = img.T[:, ::-1]
	img = img / 255.
	img = gaussian_filter(img, smoothing, mode='nearest')
	lines = find_contours(img, level)
	
	# Compute the sampling ratio
	lengths = []
	for line in lines :
		lengths.append( arclength(line) )
	lengths = array(lengths)
	points_per_line = ceil( npoints * lengths / sum(lengths) )
	
	# Interpolate accordingly
	points = []
	connec = []
	index_offset = 0
	for ppl, line in zip(points_per_line, lines) :
		(p, c) = resample(line, ppl)
		points.append(p)
		connec.append(c + index_offset)
		index_offset += len(p)
	
	points = vstack(points)
	connec = vstack(connec)
	return Curve(points.ravel(), connec, 2) # Dimension 2 !
	
