import re
from pylab import *

def ReadVTK(filename, points = True, polygons = True, signals = True, thicknesses = True):
	""" Specification of VTK-files:
		http://www.vtk.org/VTK/img/file-formats.pdf - page 4 """
	f = open(filename)
	lines = f.readlines()
	f.close()

	verticeList = []
	polygonsList = []
	signalsList = []
	
	lineNr = 0
	#pattern_points = re.compile('([-+]?\d*\.\d+|\d+) ([-+]?\d*\.\d+|\d+) ([-+]?\d*\.\d+|\d+)')
	pattern_points_3perline = re.compile('([-+Ee\d\.]+) ([-+Ee\d\.]+) ([-+Ee\d\.]+) ([-+Ee\d\.]+) ([-+Ee\d\.]+) ([-+Ee\d\.]+) ([-+Ee\d\.]+) ([-+Ee\d\.]+) ([-+Ee\d\.]+)')
	pattern_points_2perline = re.compile('([-+Ee\d\.]+) ([-+Ee\d\.]+) ([-+Ee\d\.]+) ([-+Ee\d\.]+) ([-+Ee\d\.]+) ([-+Ee\d\.]+)')
	pattern_points          = re.compile('([-+Ee\d\.]+) ([-+Ee\d\.]+) ([-+Ee\d\.]+)')
	pattern_polygons_3      = re.compile('([\d]+) ([\d]+) ([\d]+) ([\d]+)')
	pattern_polygons_2      = re.compile('([\d]+) ([\d]+) ([\d]+)')
	pattern_signal          = re.compile('([-+E\d\.]+)')
	
	def handle_points(l) :
		m = pattern_points_3perline.match(l)
		if m != None:
			x1 = float(m.group(1))
			y1 = float(m.group(2))
			z1 = float(m.group(3))
			x2 = float(m.group(4))
			y2 = float(m.group(5))
			z2 = float(m.group(6))
			x3 = float(m.group(7))
			y3 = float(m.group(8))
			z3 = float(m.group(9))
			verticeList.append([x1,y1,z1])
			verticeList.append([x2,y2,z2])
			verticeList.append([x3,y3,z3])
		else :
			m = pattern_points_2perline.match(l)
			if m != None:
				x1 = float(m.group(1))
				y1 = float(m.group(2))
				z1 = float(m.group(3))
				x2 = float(m.group(4))
				y2 = float(m.group(5))
				z2 = float(m.group(6))
				verticeList.append([x1,y1,z1])
				verticeList.append([x2,y2,z2])
			else :
				m = pattern_points.match(l)
				if m != None:
					x = float(m.group(1))
					y = float(m.group(2))
					z = float(m.group(3))
					verticeList.append([x,y,z])
	
	def handle_polygons(l) :
		m = pattern_polygons_3.match(l)
		if m != None :
			nrOfPoints = m.group(1)
			vertice1 = int(m.group(2))
			vertice2 = int(m.group(3))
			vertice3 = int(m.group(4))
			polygonsList.append([vertice1, vertice2, vertice3])
		else :
			m = pattern_polygons_2.match(l)
			if m != None :
				nrOfPoints = m.group(1)
				vertice1 = int(m.group(2))
				vertice2 = int(m.group(3))
				polygonsList.append([vertice1, vertice2])
	
	def handle_signal(l) :
		m = pattern_signal.match(l)
		if m != None :
			signal = float(m.group(1))
			signalsList.append(signal)
	
	def handle_nothing(l) :
		None
	
	current_treatment = handle_nothing 
	for line in lines :
		if "POINTS" in line :
			current_treatment = handle_points
		elif "POLYGONS" in line :
			current_treatment = handle_polygons
		elif "LINES" in line :
			current_treatment = handle_polygons
		elif "LOOKUP_TABLE" in line :
			current_treatment = handle_signal
		else :
			current_treatment(line)
			
	"""
	while "POINTS" not in lines[lineNr]:
		lineNr += 1
	while "POLYGONS" not in lines[lineNr]:
		lineNr += 1
		m = pattern_points.match(lines[lineNr])
		if m != None:
			x = float(m.group(1))
			y = float(m.group(2))
			z = float(m.group(3))
			verticeList.append([x,y,z])
	while "LOOKUP_TABLE" not in lines[lineNr]:
		lineNr += 1
		m = pattern_polygons.match(lines[lineNr])
		if m != None :
			nrOfPoints = m.group(1)
			vertice1 = int(m.group(2))
			vertice2 = int(m.group(3))
			vertice3 = int(m.group(4))
			gewicht = 1.0
			polygonsList.append([vertice1, vertice2, vertice3])
	while lineNr < len(lines)-1:
		lineNr += 1
		m = pattern_signal.match(lines[lineNr])
		if m != None :
			signal = float(m.group(1))
			signalsList.append(signal)
	"""
	return (verticeList, polygonsList, signalsList)

		
		
		
		
		
		
		
		
