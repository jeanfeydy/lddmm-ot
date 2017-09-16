from lddmm_kernel_ot import *
from pyvtk import VtkData

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


def DisplayModel(Q1, G1, Xt, info, it, scale_attach, form='.svg') :
	"Displays a pyplot Figure and save it."
	# Figure at "t = 1" : -----------------------------------------------------------------------
	fig = plt.figure(2, figsize = (20,20), dpi=100); fig.clf(); ax = fig.add_subplot(1, 1, 1)
	ax.autoscale(tight=True)
	
	if scale_attach == 0 : # Convenient way of saying that we're using a transport plan.
		ShowTransport( Q1, Xt, info, ax)
	else :                 # Otherwise, it's a kernel matching term.
		ax.imshow(info, interpolation='bilinear', origin='lower', 
				vmin = -scale_attach, vmax = scale_attach, cmap=cm.RdBu, 
				extent=(0,1, 0, 1)) 
	G1.plot(ax, color = (.8,.8,.8), linewidth = 2)
	Xt.plot(ax, color = (.76, .29, 1.))
	Q1.plot(ax)
	
	ax.axis([-.7, .7, -.7, .7]) ; ax.set_aspect('equal') ; plt.draw() ; plt.pause(0.001)
	fig.savefig( 'results/images/model_' + str(it) + form )

def FrameIt(it, target, fname) :
	Q1 = Curve.from_file(fname+'Descent/Models/Model_'+str(it)+'.vtk', offset=False)
	G1 = Curve.from_file(fname+'Descent/Grids/Grid_'  +str(it)+'.vtk', offset=False)
	P1 = np.loadtxt(     fname+'Descent/Plans/plan_'  +str(it)+'.csv')
	
	DisplayModel(Q1, G1, target, P1, it, 0)

if __name__ == '__main__' :
	plt.ion()
	plt.show()
	
	foldername = 'results/vtk_files/sinkhorn_eps-s_rho-l/'
	
	a  = open(foldername+'Target.vtk')
	print(a)
	a.close()
	Target = Curve.from_file(foldername+'Target.vtk', offset=False)
	
	for it in [1, 10, 20, 40, 68] :
		FrameIt(it, Target, foldername)
	
	
	



























