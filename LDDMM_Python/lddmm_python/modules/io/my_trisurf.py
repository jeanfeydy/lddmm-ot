from __future__ import absolute_import
from collections import OrderedDict

import warnings

import six
import math
import decimal

from plotly import utils
from plotly import exceptions
from plotly import graph_reference
from plotly import session
from plotly.files import (CONFIG_FILE, CREDENTIALS_FILE, FILE_CONTENT,
                          GRAPH_REFERENCE_FILE, check_file_permissions)

DEFAULT_PLOTLY_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                         'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                         'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                         'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                         'rgb(188, 189, 34)', 'rgb(23, 190, 207)']


REQUIRED_GANTT_KEYS = ['Task', 'Start', 'Finish']
PLOTLY_SCALES = {'Greys': ['rgb(0,0,0)', 'rgb(255,255,255)'],
                 'YlGnBu': ['rgb(8,29,88)', 'rgb(255,255,217)'],
                 'Greens': ['rgb(0,68,27)', 'rgb(247,252,245)'],
                 'YlOrRd': ['rgb(128,0,38)', 'rgb(255,255,204)'],
                 'Bluered': ['rgb(0,0,255)', 'rgb(255,0,0)'],
                 'RdBu': ['rgb(5,10,172)', 'rgb(178,10,28)'],
                 'Reds': ['rgb(220,220,220)', 'rgb(178,10,28)'],
                 'Blues': ['rgb(5,10,172)', 'rgb(220,220,220)'],
                 'Picnic': ['rgb(0,0,255)', 'rgb(255,0,0)'],
                 'Rainbow': ['rgb(150,0,90)', 'rgb(255,0,0)'],
                 'Portland': ['rgb(12,51,131)', 'rgb(217,30,30)'],
                 'Jet': ['rgb(0,0,131)', 'rgb(128,0,0)'],
                 'Hot': ['rgb(0,0,0)', 'rgb(255,255,255)'],
                 'Blackbody': ['rgb(0,0,0)', 'rgb(160,200,255)'],
                 'Earth': ['rgb(0,0,130)', 'rgb(255,255,255)'],
                 'Electric': ['rgb(0,0,0)', 'rgb(255,250,220)'],
                 'Viridis': ['rgb(68,1,84)', 'rgb(253,231,37)']}

# color constants for violin plot
DEFAULT_FILLCOLOR = '#1f77b4'
DEFAULT_HISTNORM = 'probability density'
ALTERNATIVE_HISTNORM = 'probability'


# Warning format
def warning_on_one_line(message, category, filename, lineno,
                        file=None, line=None):
    return '%s:%s: %s:\n\n%s\n\n' % (filename, lineno, category.__name__,
                                     message)
warnings.formatwarning = warning_on_one_line

try:
    import IPython
    import IPython.core.display
    _ipython_imported = True
except ImportError:
    _ipython_imported = False

try:
    import numpy as np
    _numpy_imported = True
except ImportError:
    _numpy_imported = False

try:
    import pandas as pd
    _pandas_imported = True
except ImportError:
    _pandas_imported = False

try:
    import scipy as scp
    _scipy_imported = True
except ImportError:
    _scipy_imported = False

try:
    import scipy.spatial as scs
    _scipy__spatial_imported = True
except ImportError:
    _scipy__spatial_imported = False

try:
    import scipy.cluster.hierarchy as sch
    _scipy__cluster__hierarchy_imported = True
except ImportError:
    _scipy__cluster__hierarchy_imported = False

try:
    import scipy
    import scipy.stats
    _scipy_imported = True
except ImportError:
    _scipy_imported = False


from plotly.tools import FigureFactory



def my_map_face2color(face, colormap, vmin, vmax):
	"""
	Normalize facecolor values by vmin/vmax and return rgb-color strings

	This function takes a tuple color along with a colormap and a minimum
	(vmin) and maximum (vmax) range of possible mean distances for the
	given parametrized surface. It returns an rgb color based on the mean
	distance between vmin and vmax

	"""
	if vmin >= vmax:
		vmax = vmin + 1

	if len(colormap) == 1:
		# color each triangle face with the same color in colormap
		face_color = colormap[0]
		face_color = FigureFactory._convert_to_RGB_255(face_color)
		face_color = FigureFactory._label_rgb(face_color)
	else:
		if face >= vmax:
			# pick last color in colormap
			face_color = colormap[-1]
			face_color = FigureFactory._convert_to_RGB_255(face_color)
			face_color = FigureFactory._label_rgb(face_color)
		else:
			# find the normalized distance t of a triangle face between
			# vmin and vmax where the distance is between 0 and 1
			t = (face - vmin) / float((vmax - vmin))
			low_color_index = int(t / (1./(len(colormap) - 1)))

			face_color = FigureFactory._find_intermediate_color(
				colormap[low_color_index],
				colormap[low_color_index + 1],
				t * (len(colormap) - 1) - low_color_index
			)

			face_color = FigureFactory._convert_to_RGB_255(face_color)
			face_color = FigureFactory._label_rgb(face_color)
	return face_color

def my_trisurf(x, y, z, simplices, show_colorbar, edges_color,
				 colormap=None, color_func=None, plot_edges=False,
				 x_edge=None, y_edge=None, z_edge=None, facecolor=None, data_list=False,
				 minmax_values = None):
		"""
		Refer to FigureFactory.create_trisurf() for docstring
		"""
		# numpy import check
		if _numpy_imported is False:
			raise ImportError("FigureFactory._trisurf() requires "
							  "numpy imported.")
		import numpy as np
		from plotly.graph_objs import graph_objs
		points3D = np.vstack((x, y, z)).T
		simplices = np.atleast_2d(simplices)

		# vertices of the surface triangles
		tri_vertices = points3D[simplices]

		# Define colors for the triangle faces
		if color_func is None:
			# mean values of z-coordinates of triangle vertices
			mean_dists = tri_vertices[:, :, 2].mean(-1)
		elif isinstance(color_func, (list, np.ndarray)):
			# Pre-computed list / array of values to map onto color
			if len(color_func) != len(simplices):
				raise ValueError("If color_func is a list/array, it must "
								 "be the same length as simplices.")

			# convert all colors in color_func to rgb
			for index in range(len(color_func)):
				if isinstance(color_func[index], str):
					if '#' in color_func[index]:
						foo = FigureFactory._hex_to_rgb(color_func[index])
						color_func[index] = FigureFactory._label_rgb(foo)

				if isinstance(color_func[index], tuple):
					foo = FigureFactory._convert_to_RGB_255(color_func[index])
					color_func[index] = FigureFactory._label_rgb(foo)

			mean_dists = np.asarray(color_func)
		else:
			# apply user inputted function to calculate
			# custom coloring for triangle vertices
			mean_dists = []
			for triangle in tri_vertices:
				dists = []
				for vertex in triangle:
					dist = color_func(vertex[0], vertex[1], vertex[2])
					dists.append(dist)
				mean_dists.append(np.mean(dists))
			mean_dists = np.asarray(mean_dists)

		# Check if facecolors are already strings and can be skipped
		if isinstance(mean_dists[0], str):
			facecolor = mean_dists
		else:
			if minmax_values == None :
				min_mean_dists = np.min(mean_dists)
				max_mean_dists = np.max(mean_dists)
			else :
				min_mean_dists = minmax_values[0]
				max_mean_dists = minmax_values[1]
			if facecolor is None:
				facecolor = []
			for index in range(len(mean_dists)):
				color = my_map_face2color(mean_dists[index],
													  colormap,
													  min_mean_dists,
													  max_mean_dists)
				facecolor.append(color)

		# Make sure facecolor is a list so output is consistent across Pythons
		facecolor = list(facecolor)
		ii, jj, kk = simplices.T

		triangles = graph_objs.Mesh3d(x=x, y=y, z=z, facecolor=facecolor,
									  i=ii, j=jj, k=kk, name='', hoverinfo='skip')

		mean_dists_are_numbers = not isinstance(mean_dists[0], str)

		if mean_dists_are_numbers and show_colorbar is True:
			# make a colorscale from the colors
			colorscale = FigureFactory._make_colorscale(colormap)
			colorscale = FigureFactory._convert_colorscale_to_rgb(colorscale)

			colorbar = graph_objs.Scatter3d(
				x=[x[0]], # !!! solve a bug in the orginal file !
				y=[y[0]],
				z=[z[0]],
				mode='markers',
				marker=dict(
					size=0.1,
					color=[min_mean_dists, max_mean_dists],
					colorscale=colorscale,
					showscale=True,
					colorbar = dict(
						len = 0.5
					),
				),
				hoverinfo='None',
				showlegend=False
			)

		# the triangle sides are not plotted
		if plot_edges is False:
			if mean_dists_are_numbers and show_colorbar is True:
				return graph_objs.Data([triangles, colorbar])
			else:
				return graph_objs.Data([triangles])

		# define the lists x_edge, y_edge and z_edge, of x, y, resp z
		# coordinates of edge end points for each triangle
		# None separates data corresponding to two consecutive triangles
		is_none = [ii is None for ii in [x_edge, y_edge, z_edge]]
		if any(is_none):
			if not all(is_none):
				raise ValueError("If any (x_edge, y_edge, z_edge) is None, "
								 "all must be None")
			else:
				x_edge = []
				y_edge = []
				z_edge = []

		# Pull indices we care about, then add a None column to separate tris
		ixs_triangles = [0, 1, 2, 0]
		pull_edges = tri_vertices[:, ixs_triangles, :]
		x_edge_pull = np.hstack([pull_edges[:, :, 0],
								 np.tile(None, [pull_edges.shape[0], 1])])
		y_edge_pull = np.hstack([pull_edges[:, :, 1],
								 np.tile(None, [pull_edges.shape[0], 1])])
		z_edge_pull = np.hstack([pull_edges[:, :, 2],
								 np.tile(None, [pull_edges.shape[0], 1])])

		# Now unravel the edges into a 1-d vector for plotting
		x_edge = np.hstack([x_edge, x_edge_pull.reshape([1, -1])[0]])
		y_edge = np.hstack([y_edge, y_edge_pull.reshape([1, -1])[0]])
		z_edge = np.hstack([z_edge, z_edge_pull.reshape([1, -1])[0]])

		if not (len(x_edge) == len(y_edge) == len(z_edge)):
			raise exceptions.PlotlyError("The lengths of x_edge, y_edge and "
										 "z_edge are not the same.")

		# define the lines for plotting
		lines = graph_objs.Scatter3d(
			x=x_edge, y=y_edge, z=z_edge, mode='lines',
			line=graph_objs.Line(
				color=edges_color,
				width=1.5
			),
			showlegend=False
		)
		if data_list :
			if mean_dists_are_numbers and show_colorbar is True:
				return [triangles, lines, colorbar]
			else:
				return [triangles, lines]
		else :
			if mean_dists_are_numbers and show_colorbar is True:
				return graph_objs.Data([triangles, lines, colorbar])
			else:
				return graph_objs.Data([triangles, lines])
