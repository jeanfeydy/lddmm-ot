# We use a slightly hacked version of the plot.ly js/python library
lddmm_python = __import__(__name__.split('.')[0])
print(lddmm_python)
import lddmm_python.lib.plotly as plotly

import re
from pylab import *
from IPython.html.widgets import interact
from IPython.display import HTML, display
from pprint import pprint
import json

from plotly.tools import FigureFactory as FF
from plotly import utils, graph_objs

from .my_iplot import my_iplot
from .read_vtk import ReadVTK


class Anim3d :
	def __init__(self, filenames):
		self.frames = list(ReadVTK(f) for f in filenames)
		self.current_frame = list(range(len(self.frames)))
		print(self.current_frame)
	def get_frame(self, w) :
		points = array(self.frames[w][0])
		"""update = dict(
			x = [points[:,2]],
			y = [points[:,0]],
			z = [points[:,1]]
			)"""
		update1 = dict(
			visible = False
			)
		update2 = dict(
			visible = True
			)
		list1 = str(self.current_frame)
		self.current_frame = [w]
		list2 = str(self.current_frame)
		return ([update1, update2], [list1, list2])
	def show(self, title):
		figs = []
		for ind_f in range(len(self.frames)) :
			(points, triangles, signals) = self.frames[ind_f]
			points = array(points)
			triangles = array(triangles)
			signals = array(signals)
			signals_per_triangle = list( (signals[triangles[i,0]] + signals[triangles[i,1]] + signals[triangles[i,2]]) / 3
										for i in range(triangles.shape[0]) )
			signals_per_triangle[0] += 0.001
			# Validate colormap
			my_colormap = FF._validate_colors("Portland", 'tuple')
			newdata = FF._trisurf(x=points[:,2], y=points[:,0], z=points[:,1],
								 colormap=my_colormap,
								 simplices=triangles,
								 color_func = signals_per_triangle,
								 plot_edges=False,
								 edges_color = 'rgb(50, 50, 50)',
								 show_colorbar = False,
								 data_list = True)
			figs +=  newdata
		axis = dict(
			showbackground=True,
			backgroundcolor='rgb(230, 230, 230)',
			gridcolor='rgb(255, 255, 255)',
			zerolinecolor='rgb(255, 255, 255)'
		)
		xaxis = axis.copy()
		xaxis['range'] = [-0.08,0.09]
		yaxis = axis.copy()
		yaxis['range'] = [-0.11,0.05]
		zaxis = axis.copy()
		zaxis['range'] = [0.02,0.18]
		aspectratio=dict(x=1, y=1, z=1)
		layout = graph_objs.Layout(
			title=title,
			width='100%',
			height= 800,
			scene=graph_objs.Scene(
				xaxis=graph_objs.XAxis(xaxis),
				yaxis=graph_objs.YAxis(yaxis),
				zaxis=graph_objs.ZAxis(zaxis),
				aspectratio=dict(
					x=aspectratio['x'],
					y=aspectratio['y'],
					z=aspectratio['z']),
				)
		)
		
		return my_iplot(graph_objs.Figure( data = figs, layout=layout))
	def slider(self, div_id) :
		#div_id = self.show(*args, **kwargs)
		
		def change_frame(w) :
			(updates, indices) = self.get_frame(w-1)
			script = ''
			for i in range(len(updates)) :
				jupdate = json.dumps(updates[i], cls=utils.PlotlyJSONEncoder)
				#pprint(jupdate)
				script = script \
					+ 'Plotly.restyle("{id}", {update}, [{index}]);'.format(
					id=div_id,
					update=jupdate, index = indices[i][1:-1])
			#print(script)
			update_str = (
				''
				'<script type="text/javascript">' +
				'window.PLOTLYENV=window.PLOTLYENV || {{}};'
				'window.PLOTLYENV.BASE_URL="' + 'https://plot.ly' + '";'
				'{script}' +
				'</script>'
				'').format(script=script)
			display(HTML(update_str))
		interact((lambda frame : change_frame(frame)), frame=(1,len(self.frames)))	
		
		
