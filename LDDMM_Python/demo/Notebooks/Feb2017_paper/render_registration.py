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


def DisplayModel(Q1, G1, Xt, info, it, scale_attach, form='.png', title = '') :
	"Displays a pyplot Figure and save it."
	# Figure at "t = 1" : -----------------------------------------------------------------------
	fig = plt.figure(2, figsize = (20,20), dpi=50); fig.clf(); ax = fig.add_subplot(1, 1, 1)
	ax.autoscale(tight=True)
	
	G1.plot(ax, color = (.8,.8,.8), linewidth = 2)
	if scale_attach == 0 : # Convenient way of saying that we're using a transport plan.
		ShowTransport( Q1, Xt, info, ax)
	else :                 # Otherwise, it's a kernel matching term.
		ax.imshow(info, interpolation='bilinear', origin='lower', 
				vmin = -scale_attach, vmax = scale_attach, cmap=cm.RdBu, 
				extent=(0,1, 0, 1)) 
	
	Xt.plot(ax, color = (0., 0., .8), linewidth = 5)
	Q1.plot(ax, color = (.8, 0., 0.), linewidth = 5)
	ax.text(0.5, 0., title, horizontalalignment='center', verticalalignment='center',\
	        transform = ax.transAxes, size=48)
	ax.axis([-.6, .6, -.6, .6]) ; ax.set_aspect('equal') ; plt.draw() ; 
	plt.axis('off')
	plt.pause(0.001)
	fig.savefig( 'output/registration/frame_' + str(it).zfill(5) + form, bbox_inches='tight' )

if __name__ == '__main__' :
	plt.ion()
	plt.show()
	
	fname = 'results/vtk_files/sinkhorn_eps-s_rho-l/'
	
	script = [ (np.array([1,2,4,5,6,7,8,9,10,11]),
	            np.array([1,2,3,5,6,7,8,9,10,11]),
	            31, 1),
	            (np.arange(12,22), np.arange(12,22), 11, 1),
	            (np.arange(23,69), np.arange(23,69), 3,  1)
	]
	script = [  (np.array([11, 12]), np.array([11, 12]), 31, 1, 10),
	            (np.arange(12,24), np.arange(12,24), 11, 1,     11),
	            (np.arange(23,69), np.arange(23,69), 3,  1,     23)
	]
	script = [  (np.array([4, 5]), np.array([4, 5]), 31, 1, 3)
	]
	glob_it = 62
	for (iterations, plans, fr, pause, BFGSnumber) in script :
		Qt = [ Curve.from_file(fname+'Descent/Models/Model_'+str(it)+'.vtk', offset=False) for it in iterations ]
		Gt = [ Curve.from_file(fname+'Descent/Grids/Grid_'  +str(it)+'.vtk', offset=False) for it in iterations ]
		pt = [      np.loadtxt(fname+'Descent/Plans/plan_'  +str(it)+'.csv')               for it in plans ]
		
		pt[0] = (.3*pt[0]+.7*pt[1])
		
		Q_conn = Qt[0].connectivity
		G_conn = Gt[0].connectivity
		qt = np.array([ Q.points for Q in Qt ])
		gt = np.array([ G.points for G in Gt ])
		
		
		movie_times = np.arange( len(iterations) )
		print(movie_times)
		print(qt.shape)
		q_t = interp1d( movie_times, qt, axis=0 )
		g_t = interp1d( movie_times, gt, axis=0 )
		
		Q_t = lambda t : Curve( q_t(t), Q_conn )
		G_t = lambda t : Curve( g_t(t), G_conn )
		P_t = lambda t : pt[ int(np.floor(t)) ]
		
		Target = Curve.from_file(fname+'Target.vtk', offset=False)
		
		for it in range((len(movie_times)-1)*fr + 1) :
			glob_it = glob_it + 1
			step = it // fr
			time = (it%fr)/(fr-pause)
			if time > 1 :
				DisplayModel(Q_t(step+1), G_t(step+1), Target, P_t(step), glob_it, 0, title='L-BFGS Iteration ' + str(step+BFGSnumber).zfill(2))
			else :
				DisplayModel(Q_t(step + time), G_t(step + time), Target, P_t(step), glob_it, 0, title='L-BFGS Iteration ' + str(step+BFGSnumber).zfill(2))
		
	
	



























