# Import the relevant tools
import numpy as np          # standard array library
import torch

def _squared_distances(x, y) :
	"Returns the matrix of $\|x_i-y_j\|^2$."
	x_col = x.unsqueeze(1) # Theano : x.dimshuffle(0, 'x', 1)
	y_lin = y.unsqueeze(0) # Theano : y.dimshuffle('x', 0, 1)
	return torch.sum( (x_col - y_lin)**2 , 2 )

def _k(x, y, s) :
	"Returns the matrix of k(x_i,y_j)."
	sq = _squared_distances(x, y) / (s**2)
	#return torch.exp( -sq )
	return torch.pow( 1. / ( 1. + sq ), .25 )

def _cross_kernels(q, x, s) :
	"Returns the full k-correlation matrices between two point clouds q and x."
	K_qq = _k(q, q, s)
	K_qx = _k(q, x, s)
	K_xx = _k(x, x, s)
	return (K_qq, K_qx, K_xx)
