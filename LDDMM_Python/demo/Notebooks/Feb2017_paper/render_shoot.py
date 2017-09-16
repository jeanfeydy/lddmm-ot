from lddmm_kernel_ot import *
from pyvtk import VtkData
from scipy.interpolate import interp1d

def ShowTransport( Q, Xt, Gamma, ax ) :
	"Displays a transport plan."
	print(Gamma.shape)
	points = [] ; connectivity = [] ; curr_id = 0
	Q_points,Q_weights = Q.to_measure()  ;  xtpoints = Xt.points # Extract the centers + areas
	for (a, mui, gi) in zip(Q_points, Q_weights, Gamma) :
		gi = gi / mui # gi[j] = fraction of the mass from "a" which goes to xtpoints[j]
		for (seg, gij) in zip(Xt.connectivity, gi) :
			mass_per_line = 0.01
			if gij >= mass_per_line :
				nlines = np.floor(gij / mass_per_line)
				ts     = np.linspace(.2, .8, nlines)
				for t in ts :
					b = (1-t) * xtpoints[seg[0]] + t * xtpoints[seg[1]]
					points += [a, b]; connectivity += [[curr_id, curr_id + 1]]; curr_id += 2
	if len(connectivity) > 0 :
		Plan = Curve(np.vstack(points), np.vstack(connectivity))
		Plan.plot(ax, color = (.8,.9,1.,.1), linewidth = 1)


def DisplayModel(Q1, G1, Xt, info, it, scale_attach, form='.png') :
	"Displays a pyplot Figure and save it."
	# Figure at "t = 1" : -----------------------------------------------------------------------
	fig = plt.figure(2, figsize = (20,20), dpi=50); fig.clf(); ax = fig.add_subplot(1, 1, 1)
	ax.autoscale(tight=True)
	"""
	if scale_attach == 0 : # Convenient way of saying that we're using a transport plan.
		ShowTransport( Q1, Xt, info, ax)
	else :                 # Otherwise, it's a kernel matching term.
		ax.imshow(info, interpolation='bilinear', origin='lower', 
				vmin = -scale_attach, vmax = scale_attach, cmap=cm.RdBu, 
				extent=(0,1, 0, 1)) 
	"""
	G1.plot(ax, color = (.8,.8,.8),   linewidth = 2)
	Xt.plot(ax, color = (0., 0., .8), linewidth = 5)
	Q1.plot(ax, color = (.8, 0., 0.), linewidth = 5)
	
	ax.axis([-.6, .6, -.6, .6]) ; ax.set_aspect('equal') ; plt.draw() ; 
	plt.axis('off')
	plt.pause(0.001)
	fig.savefig( 'output/shoot/time_' + str(it).zfill(4) + form, bbox_inches='tight' )

if __name__ == '__main__' :
	plt.ion()
	plt.show()
	
	fname = 'results/vtk_files/sinkhorn_eps-s_rho-l/'
	
	
	Qt = [ Curve.from_file(fname+'Shoot/Shoot_'+str(it)+'.vtk', offset=False) for it in range(11)]
	Gt = [ Curve.from_file(fname+'Grid/grid_'  +str(it)+'.vtk', offset=False) for it in range(11)]
	
	Q_conn = Qt[0].connectivity
	G_conn = Gt[0].connectivity
	qt = [ Q.points for Q in Qt ]
	gt = [ G.points for G in Gt ]
	
	qt = np.array([qt[0]] + qt + [qt[-1]])
	gt = np.array([gt[0]] + gt + [gt[-1]])
	
	movie_times = np.arange(13)
	print(movie_times)
	print(qt.shape)
	q_t = interp1d( movie_times, qt, axis=0 )
	g_t = interp1d( movie_times, gt, axis=0 )
	
	Q_t = lambda t : Curve( q_t(t), Q_conn )
	G_t = lambda t : Curve( g_t(t), G_conn )
	
	Target = Curve.from_file(fname+'Target.vtk', offset=False)
	
	for it in range(121) :
		DisplayModel(Q_t(it/10.), G_t(it/10.), Target, 0, it, 0)
	
	
	



























