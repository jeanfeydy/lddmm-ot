from pylab import *
from collections import namedtuple
from .varifolds import Varifolds, Varifold
from .currents import Currents, Current


Cylinders = namedtuple('Cylinders', 'centers normals weights')
Spherical = namedtuple('Spherical', 'points directions weights')

class NormalCycle :
	def __init__(self, cylinders, spherical) :
		"""
		Normal cycles are encoded as superpositions of two collections :
		- cylinders, which are given by :
			- centers    - n-by-d array
			- normals    - n-by-d array, with unit line
			- weights    - n vector
		- spherical, which are given by :
			- points     - N-by-d array
			- directions - N-list of n_j-by-d array with unit lines
			- weights    - N-list of n_j vectors
		
		When handling curves, 
		"cylinders" encode the normal cycle of the interior of segments ]x_i, x_j[, 
		while "spherical" encode the spherical (or circular) sections of the normal cycle,
		localized at junction/end points.
		"""
		self.cylinders = cylinders
		self.spherical = spherical
		
class NormalCycles :
	@staticmethod
	def kernel_matching(Q, Xt, s) :
		"""
		Outputs a cost C + a NormalCycle dQ of directional derivatives with respect
		to the variables of Q. Note that the directional derivatives with respect
		to unit directions are given as vectors orthogonal to those directions.
		
		Following the notations in 
		"Kernel Metrics on Normal Cycles and Application to Curve Matching",
		Pierre Roussillon and Joan Alexis Glaunes, 2015,
		we will take :
		- as "points kernel" k_p, a gaussian kernel of std = s
		- as "normal kernel" k_n, a simple "linear" kernel :
		    k_n(u, v) = (u.v) = "cos(theta)" - this is coherent with our choice
		    to restrict ourselves to the "quadratic" kernel "cos^2(theta)"
		    in the Varifolds.kernel_matching method.
		
		This choice of k_n means that NormalCycles.kernel_matching would basically
		be equivalent to Varifolds.kernel_matching, if it was not for the "spherical" part
		(which encodes the order 2 curvature information as angle variations/non-alignment).
		
		The most important 'lemma' to have in mind here is that :
		
		\int_{U-\pi/2}^{U+\pi/2} \int_{V-\pi/2}^{V+\pi/2} cos( u - v ) du dv  = 4 * cos( U - V )
		
		so that the correlation between two half-spheres
		C = \delta_c x S^+_U
		S = \delta_s x S^+_V
		is given by
		[ N(C) | N(S) ] = k_p( c-s ) * 4 * ( U . V ) .
		
		The bilinearity of this latter formula means that we can considerably simplify 
		calculations : instead of having to carry around the information of all 
		the hemispheres/cycles S^+_U at a given point p (directions U_i), 
		we can aggregate it into a simple vector U_p = \sum_i (w_i * U_i),
		which synthetizes in a practical way the behavior of the curve at the junction point.
		(Remember that typically, w_i = -1, while the "whole sphere" N({p}) collapses
		to a zero contribution (whole sphere = sum of two opposite hemispheres).
		
		This is a kind of "mean field" result.
		Note that the vector U_p is geometrically meaningful :
			- if p is an "end point" of the curve, U_p is the unit vector
			  which points in the direction "to be continued".
			- if p is a regular point of the curve, U_p is kind of equivalent
			  to the acceleration vector, but "outward pointing".
			  It is null if the curve is straight, and otherwise,
			  if it makes an angle t < pi,
			  it points in the direction of "concavity", 
			  bissecting the angle with a norm 2 * cos(t/2).
		
		In the end, using NormalCycles with a linear kernel on the normals
		is simply equivalent to using a Varifold data attachment term on the segments
		+ a Current one on the "pseudo -Acceleration field" U_p, located at the vertices of the curve.
		We just have to be careful when distributing the gradient dU_p to the U_i !
		"""
		
		# Cylindrical part
		Q_segments  = Varifold(  Q.cylinders.centers,  Q.cylinders.normals,  Q.cylinders.weights )
		Xt_segments = Varifold( Xt.cylinders.centers, Xt.cylinders.normals, Xt.cylinders.weights )
		
		(C_segments, dQ_segments) = Varifolds.kernel_matching(Q_segments, Xt_segments, s)
		
		# Spherical part
		Q_accelerations  = Current(  Q.spherical.points, 
		                             vstack( [ sum( atleast_2d(w_is).T * U_is, 0 ) 
		                             for (w_is, U_is) in zip(Q.spherical.weights, Q.spherical.directions) ] ) )
		Xt_accelerations = Current( Xt.spherical.points, 
		                             vstack( [ sum( atleast_2d(w_is).T * U_is, 0 ) 
		                             for (w_is, U_is) in zip(Xt.spherical.weights, Xt.spherical.directions) ] ) )
		
		(C_accelerations, dQ_accelerations) = Currents.kernel_matching(Q_accelerations, Xt_accelerations, s)
		# Back-propagation of the accelerations gradient on the weights + normalized directions
		dQ_directions = [ vstack( [ ( dU_p - dot(U_i, dU_p) * U_i ) / w_i
		                            for (w_i, U_i) in zip( w_is, U_is )  ] )
		                  for (w_is, U_is, dU_p) in zip(Q.spherical.weights, Q.spherical.directions, dQ_accelerations.normals) ]
		dQ_weights    = [ array( [  dot(U_i, dU_p)
		                            for (w_i, U_i) in zip( w_is, U_is )  ] )
		                  for (w_is, U_is, dU_p) in zip(Q.spherical.weights, Q.spherical.directions, dQ_accelerations.normals) ]
		
		
		# Overall
		C = C_segments + C_accelerations
		
		dQ = NormalCycle(
			Cylinders( 
				centers = dQ_segments.points,
				normals = dQ_segments.normals,
				weights = dQ_segments.weights
			),
			Spherical(
				points     = 0*dQ_accelerations.points,
				directions = dQ_directions,
				weights    = dQ_weights
			)
		)
		
		return (C, dQ)






















