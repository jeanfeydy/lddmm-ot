from scipy import *
from scipy import ndimage
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from IPython.display import HTML, display
from ..io.my_iplot import my_iplot
from ..io.my_trisurf import my_trisurf

from skimage.measure import find_contours

class KendallTriangles :
	"Implementation of the triangles manifold from the classic article of David Kendall."
	def __init__(self, mode = 'whole sphere', resolution = (45, 90) ):
		"mode = 'whole sphere' | 'spherical blackboard'"
		self.mode = mode
		if self.mode == 'whole sphere' :
			thetas = linspace(0,   pi, resolution[0] + 1)
			phis   = linspace(-pi, pi, resolution[1] + 1)
		elif self.mode == 'spherical blackboard' :
			thetas = linspace(0,.5*pi, resolution[0] + 1)
			phis   = linspace(-pi/3, 0, resolution[1] + 1)
			
		(self.theta_grid, self.phi_grid) = meshgrid(thetas, phis)
		self.theta_centers = (self.theta_grid[0:-1,0:-1] + self.theta_grid[0:-1,1:] + self.theta_grid[1:,0:-1] + self.theta_grid[1:,1:]) / 4.
		self.phi_centers   = (self.phi_grid[0:-1,0:-1]   + self.phi_grid[0:-1,1:]   + self.phi_grid[1:,0:-1]   + self.phi_grid[1:,1:] ) / 4.
		
		areas_thetas = (phis[1]-phis[0]) * ( -cos(thetas[1:]) + cos(thetas[0:-1]))
		self.cell_areas = vstack( (areas_thetas,)*(resolution[1]) ).T
		
		# Setup routines for the interactive "Plot.ly" display
		self.setup_layout()
		self.current_axis = []
		
	def set_data(self, triangles):
		triangles = atleast_2d(triangles)
		assert (triangles.shape[1] == 3), 'Triangles should be given as triplets of complex numbers.'
		self.raw_data = triangles
		A = triangles[:,0]
		B = triangles[:,1]
		C = triangles[:,2]
		
		# See section 5 : The manifold carrying the shapes of triangles
		unit_length = absolute(B-A)/2
		self.M = absolute(C - (A+B)/2) / unit_length
		self.psi = angle( (C - .5*(A+B)) / (B-A) )
		# eq. (34) : 
		self.R = self.M / sqrt(3)
		self.phi = self.psi
		self.theta = 2 * arctan(self.R)
		# eq. (35)
		self.X = cos(self.theta)
		self.Y = sin(self.theta) * cos(self.phi)
		self.Z = sin(self.theta) * sin(self.phi)
		# Alternative spherical coordinates around the Z axis :
		self.theta_Z = arccos(self.Z)
		self.phi_Z   = angle(self.X + 1j * self.Y)
		
		if self.mode == 'spherical blackboard' :
			# Quotient by the symmetry + relabelling group
			self.theta_Z = arccos(abs(self.Z))
			self.phi_Z   = - absolute(  mod(self.phi_Z + (pi/3), 2*pi/3)  - (pi/3)  )
			# Recompute X, Y, Z
			self.X = sin(self.theta_Z) * cos(self.phi_Z)
			self.Y = sin(self.theta_Z) * sin(self.phi_Z)
			self.Z = cos(self.theta_Z)
			# Recompute Phi, Theta
			self.theta = arccos(self.X)
			self.phi   = angle(self.Y + 1j*self.Z)
			self.R     = tan(self.theta / 2)
		# eq. (36)
		A = - sqrt(3) - self.R * exp(1j * self.phi)
		B = + sqrt(3) - self.R * exp(1j * self.phi)
		C = 2 *         self.R * exp(1j * self.phi)
		self.triangles = vstack( (A,B,C) )
		
		# normalize self.triangles (for display purposes)
		centroids = sum(self.triangles, 0)
		self.triangles -= centroids
		norms = sqrt( sum( absolute(self.triangles)**2, 0) ) # inefficient, but whatever
		self.triangles /= norms
		self.triangles = self.triangles.T
		
		# Create an histogram
		self.compute_density()
		
	def compute_density(self) :
		"Computes an empirical histogram on the (theta x phi) grid."
		(self.density, xedges, yedges) = histogram2d(self.theta_Z, self.phi_Z, bins = (self.theta_grid[0], self.phi_grid[:,0]) )
		# some smoothing to prevent over-sensitivity next to the poles
		if self.mode == 'whole sphere' :
			sigma = .5
			total_area = 4*pi
			mode = 'wrap'
		elif self.mode == 'spherical blackboard' :
			sigma = 1.5
			total_area = 4*pi / 12
			mode = 'nearest'
		for (i, th) in enumerate( self.theta_centers[0] ) :
			self.density[i] = ndimage.filters.gaussian_filter1d(self.density[i], sigma / sin(th), mode = mode)
		self.density = self.density / sum(self.density)
		self.density = total_area*self.density / self.cell_areas
		
	def mesh(self, f) :
		"Returns a mesh (points, triangles, signals_per_triangle) with signals given by f."
		ntheta = self.theta_grid.shape[1]
		nphi   = self.theta_grid.shape[0]
		# X = cos(self.theta_grid)
		# Y = sin(self.theta_grid) * cos(self.phi_grid)
		# Z = sin(self.theta_grid) * sin(self.phi_grid)
		Z = cos(self.theta_grid)
		X = sin(self.theta_grid) * cos(self.phi_grid)
		Y = sin(self.theta_grid) * sin(self.phi_grid)
		
		
		# store the points in a npoints-by-3 array,
		# where npoints = ntheta * nphi
		points = vstack( ( X.ravel() ,
						   Y.ravel() ,
						   Z.ravel() ) ).T
		# generate the connectivity matrix, a ntriangles-by-3 int matrix,
		# each line giving the indices of 3 points -> one triangle.
		Ilen = nphi 
		Jlen = ntheta 
		#triangles = vstack([ [i+j*(ylen+1),  (i+1)+j*(ylen+1),  i+(j+1)*(ylen+1) ] for i in range(ylen) for j in range(xlen)])
		triangles = vstack( [  [ ( j +     Jlen*i    , (j+1) + Jlen*i     , j +     Jlen*(i+1) ) for j in range(Jlen - 1) ]
							 + [ ( j + 1 + Jlen*(i+1), j     + Jlen*(i+1) , j + 1 + Jlen*i     ) for j in range(Jlen - 1) ]
							   for i in range(Ilen - 1) ] )
		if f == 'density' :
			signals = self.density
		else :
			fv = vectorize(f)
			signals = fv(self.theta_centers.T, self.phi_centers.T)
		# given the connectivity matrix 'triangles', 
		# and the value of f on rectangles,
		# signals_per_triangle is easy to compute
		signals = kron( signals.T, [[1], [1]] )
		signals_per_triangle = signals.ravel()
		
		return (points, triangles, signals_per_triangle)
		
	def clear_axis(self):
		self.current_axis = []
	def show(self) : 
		"Interactive Display of the Kendall manifold."
		self.show_function(lambda theta,phi : 0, faces = True, contours = False)
		
	def show_glyphs(self, scale = 0.03, axis = 'Z') :
		"Displays triangles on the spherical manifold."
		if self.mode == 'whole sphere' or self.mode == 'spherical blackboard' :
			# We will embed self.triangles in the euclidean space R^3,
			# in the neighborhood of the sphere S(1/2).
			
			if axis == 'X' :
				theta = self.theta
				phi   = self.phi
				e_theta = vstack( ( -sin(theta)        ,  cos(theta) * cos(phi), cos(theta) * sin(phi) ) ).T
				e_phi   = vstack( ( zeros(theta.shape) , -sin(theta) * sin(phi), sin(theta) * cos(phi) ) ).T
			elif axis == 'Z' :
				theta = self.theta_Z
				phi   = self.phi_Z
				e_theta = - vstack( (  cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)         ) ).T
				e_phi   = + vstack( ( - sin(phi),  cos(phi), zeros(theta.shape)  ) ).T
				
			# We don't want glyphs to overlap
			e_theta = scale * e_theta
			e_phi   = scale * e_phi
			
			glyphs = []
			separator = [None, None, None]
			for i in range(self.triangles.shape[0]) :
				point = array([self.X[i], self.Y[i], self.Z[i]])
				glyphs.append(array([
					point + real(self.triangles[i,0]) * e_phi[i] + imag(self.triangles[i,0]) * e_theta[i] , # A
					point + real(self.triangles[i,1]) * e_phi[i] + imag(self.triangles[i,1]) * e_theta[i] , # B
					point + real(self.triangles[i,2]) * e_phi[i] + imag(self.triangles[i,2]) * e_theta[i] , # C
					point + real(self.triangles[i,0]) * e_phi[i] + imag(self.triangles[i,0]) * e_theta[i] , # A
					separator
				]))
			glyphs = vstack(glyphs)
			curves = go.Scatter3d(x = glyphs[:,0], y = glyphs[:,1], z = glyphs[:,2], mode = 'lines', hoverinfo='none', name = 'Triangles')
			self.current_axis.append(curves)
			
	def show_markers(self, **kwargs) :
		"Marker plot of the data on the spherical manifold."
		None
	
	def show_density(self, contours = True) :
		self.show_function('density', faces = True, contours = contours)
		
	def show_function(self, f, faces = True, contours = True, name = 'f') :
		"Displays a function f(phi, theta)."
		
		# Trisurf plot
		if faces :
			(points, triangles, signals_per_triangle) = self.mesh(f)
			points = array(points)
			triangles = array(triangles)
			#signals = array(signals)
			signals_per_triangle = array(signals_per_triangle)
			#signals_per_triangle = list( (signals[triangles[i,0]] + signals[triangles[i,1]] + signals[triangles[i,2]]) / 3
			#							for i in range(triangles.shape[0]) )
			signals_per_triangle[0] += 0.00001
			# Validate colormap
			my_colormap = FF._validate_colors("LinLhot", 'tuple')
			newdata = my_trisurf(x=points[:,0], y=points[:,1], z=points[:,2],
								 colormap=my_colormap,
								 simplices=triangles,
								 color_func = signals_per_triangle,
								 plot_edges=False,
								 edges_color = 'rgb(50, 50, 50)',
								 show_colorbar = True,
								 data_list = True,
								 minmax_values = (0, max(signals_per_triangle) ))
			self.current_axis += newdata
		
		# 3D contour plot
		if contours :
			R = 1.005
			if f == 'density' :
				name = 'Density'
				if self.mode == 'whole sphere' :
					den = pad( pad( self.density, (0,1), 'wrap') , (1,0), 'edge')
				else :
					den = pad( self.density, 1, 'edge' )
				values = (den[1:,1:] + den[0:-1,1:] + den[1:,0:-1] + den[0:-1,0:-1]) / 4.
			else :
				fv = vectorize(f)
				values = fv(self.theta_grid.T, self.phi_grid.T)
			levels = linspace(0, amax(values), 10)
			
			for level in levels :
				contours = find_contours(values, level)
				ntheta = self.theta_grid.shape[1] - 1
				nphi   = self.theta_grid.shape[0] - 1
				if self.mode == 'whole sphere' :
					theta   = lambda x : pi* x[0] / ntheta
					phi = lambda x : pi* (2*x[1]/nphi-1)
				elif self.mode == 'spherical blackboard' :
					theta   = lambda x : (pi/2) * x[0]/ntheta
					phi     = lambda x : (pi/3) * (x[1] / nphi - 1)
				#points3D = [ ( [ ( R* cos(pi* thph[0]/ntheta), 
				#				   R* sin(pi* thph[0]/ntheta) * cos(pi*(2*thph[1]/nphi-1)), 
				#				   R* sin(pi* thph[0]/ntheta) * sin(pi*(2*thph[1]/nphi-1)) ) 
				#				 for thph in contour ] 
				#			 + [(None, None, None)] ) 
				#			for contour in contours]
				points3D = [ ( [ (  
								   R* sin(theta(thph)) * cos(phi(thph)), 
								   R* sin(theta(thph)) * sin(phi(thph)) ,
								   R* cos(theta(thph)) ) 
								 for thph in contour ] 
							 + [(None, None, None)] ) 
							for contour in contours]
				if points3D != [] :
					contours3D = vstack(points3D)
					curves = go.Scatter3d(x = contours3D[:,0], y = contours3D[:,1], z = contours3D[:,2], mode = 'lines', hoverinfo='none',
										  line   = dict(width= 3, color='red'), name = (name + ' = ' + "{:.2f}".format(level)))
					self.current_axis.append(curves)
				
	def setup_layout(self) :
		"Setup the axis properties."
		axis = dict(
				showbackground=True,
				backgroundcolor='rgb(230, 230, 230)',
				gridcolor='rgb(255, 255, 255)',
				zerolinecolor='rgb(255, 255, 255)'
			)
		xaxis = axis.copy()
		xaxis['range'] = [-1.1,1.1]
		yaxis = axis.copy()
		yaxis['range'] = [-1.1,1.1]
		zaxis = axis.copy()
		zaxis['range'] = [-1.1,1.1]
		aspectratio=dict(x=1, y=1, z=1)
		self.layout = go.Layout(
			title='Kendall Triangles',
			width='100%',
			height= 800,
			scene=go.Scene(
				xaxis=go.XAxis(xaxis),
				yaxis=go.YAxis(yaxis),
				zaxis=go.ZAxis(zaxis),
				aspectratio=dict(
					x=aspectratio['x'],
					y=aspectratio['y'],
					z=aspectratio['z']),
				)
			)
			
	def iplot(self, title = '') :
		"Interactive manifold display "
		self.layout['title'] = title
		(wid, uid) =  my_iplot(go.Figure(data=self.current_axis, layout=self.layout))
		display(wid)

