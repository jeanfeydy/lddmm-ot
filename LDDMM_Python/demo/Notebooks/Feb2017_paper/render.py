#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def display_reader(reader, renderView) :
	# show data in view
	Display = Show(reader, renderView)
	# trace defaults for the display properties.
	Display.AmbientColor = [0.0, 0.0, 0.0]
	Display.ColorArrayName = [None, '']
	Display.DiffuseColor = [0.6666666666666666, 0.6666666666666666, 1.0]
	Display.BackfaceDiffuseColor = [0.6666666666666666, 0.6666666666666666, 1.0]
	Display.OSPRayScaleFunction = 'PiecewiseFunction'
	Display.SelectOrientationVectors = 'None'
	Display.ScaleFactor = 0.29113320297760004
	Display.SelectScaleArray = 'None'
	Display.GlyphType = 'Arrow'
	Display.GaussianRadius = 0.14556660148880002
	Display.SetScaleArray = [None, '']
	Display.ScaleTransferFunction = 'PiecewiseFunction'
	Display.OpacityArray = [None, '']
	Display.OpacityTransferFunction = 'PiecewiseFunction'
	return Display
	
def display_plan(fname, renderView) :
	plan_1vtk = LegacyVTKReader(FileNames=[fname])
	plan_1vtkDisplay = display_reader(plan_1vtk, renderView)
	plan_1vtkDisplay.SetRepresentationType('Wireframe')
	plan_1vtkDisplay.AmbientColor = [0.7, 0.87, 1.0]

def display_target(fname, renderView) :
	targetvtk = LegacyVTKReader(FileNames=[fname])
	targetvtkDisplay = display_reader(targetvtk, renderView)
	targetvtkDisplay.SetRepresentationType('Wireframe')
	targetvtkDisplay.LineWidth = 8.0
	targetvtkDisplay.AmbientColor = [0.76, 0.29, 1.0]
	
def display_model(fname, renderView) :
	model_1vtk = LegacyVTKReader(FileNames=[fname])
	generateIds1 = GenerateIds(Input=model_1vtk)
	idsLUT = GetColorTransferFunction('Ids')
	
	# show data in view
	generateIds1Display = Show(generateIds1, renderView)
	# trace defaults for the display properties.
	generateIds1Display.AmbientColor = [0.0, 0.0, 0.0]
	generateIds1Display.ColorArrayName = ['POINTS', 'Ids']
	generateIds1Display.DiffuseColor = [0.6666666666666666, 0.6666666666666666, 1.0]
	generateIds1Display.LookupTable = idsLUT
	generateIds1Display.BackfaceDiffuseColor = [0.6666666666666666, 0.6666666666666666, 1.0]
	generateIds1Display.OSPRayScaleArray = 'Ids'
	generateIds1Display.OSPRayScaleFunction = 'PiecewiseFunction'
	generateIds1Display.SelectOrientationVectors = 'Ids'
	generateIds1Display.ScaleFactor = 0.29750560522100006
	generateIds1Display.SelectScaleArray = 'Ids'
	generateIds1Display.GlyphType = 'Arrow'
	generateIds1Display.GaussianRadius = 0.14875280261050003
	generateIds1Display.SetScaleArray = ['POINTS', 'Ids']
	generateIds1Display.ScaleTransferFunction = 'PiecewiseFunction'
	generateIds1Display.OpacityArray = ['POINTS', 'Ids']
	generateIds1Display.OpacityTransferFunction = 'PiecewiseFunction'
	
	generateIds1Display.SetScalarBarVisibility(renderView, False)
	idsPWF = GetOpacityTransferFunction('Ids')
	generateIds1Display.SetRepresentationType('Wireframe')
	generateIds1Display.LineWidth = 8.0
	idsLUT.ApplyPreset('Rainbow Desaturated', True)


def get_view() :
	renderView1 = CreateView('RenderView')
	renderView1.ViewSize = [2100, 1700]
	renderView1.OrientationAxesVisibility = 0
	return renderView1

def screenshot(renderView1, fname) :
	# reset view to fit data
	#renderView1.ResetCamera()
	#renderView1.InteractionMode = '2D'
	renderView1.CameraViewAngle = 45
	renderView1.CameraPosition = [0.0, 0.0, 2.]
	renderView1.CameraFocalPoint = [0.0, 0.0, 0.0]
	#renderView1.CameraParallelScale = 1.7482104584020597

	# save screenshot
	#SaveScreenshot(fname, magnification=1, quality=100, view=renderView1)
	ExportView(fname, view=renderView1)

def display_final_matching(folder, ruler = None) :
	targetfname = 'results/vtk_files/' + folder + '/Target.vtk'
	modelfname  = 'results/vtk_files/' + folder + '/Model.vtk'
	outfname    = 'results/images/matching_' + folder + '.png'
	renderView1 = get_view()
	#display_plan(planfname, renderView1)
	display_target(targetfname, renderView1)
	display_model(modelfname, renderView1)
	
	if ruler is not None :
		display_ruler(ruler[0], ruler[1], renderView1)
	
	screenshot(renderView1, outfname)
	
def display_first_plan(folder, ruler = None) :
	targetfname = 'results/vtk_files/' + folder + '/Target.vtk'
	modelfname  = 'results/vtk_files/' + folder + '/Descent/Models/Model_1.vtk'
	planfname  = 'results/vtk_files/' + folder + '/Descent/Plans/Plan_1.vtk'
	outfname    = 'results/images/firstplan_' + folder + '.png'
	renderView1 = get_view()
	display_target(targetfname, renderView1)
	display_plan(planfname, renderView1)
	display_model(modelfname, renderView1)
	if ruler is not None :
		display_ruler(ruler[0], ruler[1], renderView1)
	screenshot(renderView1, outfname)

def display_descent(folder, iterations) :
	targetfname = 'results/vtk_files/' + folder + '/Target.vtk'
	for it in iterations :
		modelfname  = 'results/vtk_files/' + folder + '/Descent/Models/Model_' + str(it) + '.vtk'
		planfname  = 'results/vtk_files/' + folder + '/Descent/Plans/Plan_' + str(it) + '.vtk'
		gridfname  = 'results/vtk_files/' + folder + '/Descent/Grids/Grid_' + str(it) + '.vtk'
		outfname    = 'results/images/descent_' + folder + '_it-' + str(it) + '.svg'
		renderView1 = get_view()
		display_grid(gridfname, renderView1)
		display_plan(planfname, renderView1)
		display_target(targetfname, renderView1)
		display_model(modelfname, renderView1)
		screenshot(renderView1, outfname)

def display_grid(gridfname, renderView, linewidth = 1.0, color = [.7,.7,.7]) :
	"""Displays a VTK file with a light gray scheme - typically a grid."""
	grid        = LegacyVTKReader(FileNames=[gridfname])
	gridDisplay = display_reader(grid, renderView)
	gridDisplay.SetRepresentationType('Wireframe')
	gridDisplay.LineWidth    = linewidth
	gridDisplay.AmbientColor = color

def display_ruler(name, length, view) :
	# create a new 'Ruler'
	ruler1 = Ruler()
	# Properties modified on ruler1
	ruler1.Point1 = [-0.2-length/2, 0.27, 0.0]
	ruler1.Point2 = [-0.2+length/2, 0.27, 0.0]
	# show data in view
	ruler1Display = Show(ruler1, view)
	# trace defaults for the display properties.
	ruler1Display.Color = [0.0, 0.0, 0.0]
	# Properties modified on ruler1Display
	ruler1Display.LabelFormat = name + ' = %6.3g'
	# Properties modified on ruler1Display
	ruler1Display.AxisLineWidth = 8.0
	# Properties modified on ruler1Display
	ruler1Display.AxisColor = [0.2, 0.6, 1.0]
	# Properties modified on ruler1Display
	ruler1Display.FontSize = 11
	# Properties modified on ruler1Display
	ruler1Display.FontFamily = 'Courier'
	

def display_dataset() :
	folder = 'sinkhorn_eps-s_rho-l'
	targetfname = 'results/vtk_files/' + folder + '/Target.vtk'
	modelfname  = 'results/vtk_files/' + folder + '/Template.vtk'
	gridfname  = 'results/vtk_files/' + folder + '/Grid/grid_0.vtk'
	outfname    = 'results/images/dataset.png'
	
	renderView = get_view()
	
	grid = LegacyVTKReader(FileNames=[gridfname])
	gridDisplay = display_reader(grid, renderView)
	gridDisplay.SetRepresentationType('Wireframe')
	gridDisplay.LineWidth = 1.0
	gridDisplay.AmbientColor = [0.75, 0.75, .75]
	
	display_target(targetfname, renderView)
	display_model(modelfname, renderView)
	screenshot(renderView, outfname)

#display_dataset()
#display_final_matching('kernel_big', ruler = ('$\\sigma$', .2))
#display_final_matching('kernel_small', ruler = ('$\\sigma$', .05))

#display_final_matching('sinkhorn_eps-l_rho-l', ruler = ('$\\sqrt{\\epsilon}$', .1))
#display_final_matching('sinkhorn_eps-m_rho-l', ruler = ('$\\sqrt{\\epsilon}$', .03))
#display_final_matching('sinkhorn_eps-s_rho-l', ruler = ('$\\sqrt{\\epsilon}$', .015))

#display_first_plan('sinkhorn_eps-m_rho-s', ruler = ('$\\sqrt{\\rho}$', .1))
#display_first_plan('sinkhorn_eps-m_rho-m', ruler = ('$\\sqrt{\\rho}$', .15))
#display_first_plan('sinkhorn_eps-m_rho-l', ruler = ('$\\sqrt{\\rho}$', .5))

#display_descent('sinkhorn_eps-s_rho-l', [1, 5, 10, 20, 40])
display_descent('sinkhorn_eps-s_rho-l', [1, 5, 10, 20, 40, 68])
