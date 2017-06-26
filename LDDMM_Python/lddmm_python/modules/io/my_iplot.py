from plotly.offline import iplot, _plot_html
from IPython.display import HTML, display
import ipywidgets as widgets

def my_iplot(figure_or_data, show_link=False, link_text='Export to plot.ly',
		  validate=True, image=None, filename='plot_image', image_width=800,
		  image_height=600) :
	plot_html, plotdivid, width, height = _plot_html(
		figure_or_data, show_link, link_text, validate,
		'100%', 525, global_requirejs=True)
	#display(HTML(plot_html))
	wid = widgets.HTML(
		value=plot_html,
		placeholder='Some HTML',
		description='Some HTML',
		disabled=False
	)
	
	return (wid, plotdivid)
