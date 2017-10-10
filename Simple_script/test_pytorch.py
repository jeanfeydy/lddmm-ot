# Import the relevant tools
import time                 # to measure performance
import numpy as np          # standard array library
import torch
from   torch.autograd import Variable
import torch.optim as optim


use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor



def _squared_distances(x, y) :
	"Returns the matrix of $\|x_i-y_j\|^2$."
	x_col = x.unsqueeze(1) #x.dimshuffle(0, 'x', 1)
	y_lin = y.unsqueeze(0) #y.dimshuffle('x', 0, 1)
	return torch.sum( (x_col - y_lin)**2 , 2 )

def _k(x, y, s) :
	"Returns the matrix of k(x_i,y_j)."
	sq = _squared_distances(x, y) / (s**2)
	return torch.exp(-sq) #torch.pow( 1. / ( 1. + sq ), .25 )

def _cross_kernels(q, x, s) :
	"Returns the full k-correlation matrices between two point clouds q and x."
	K_qq = _k(q, q, s)
	K_qx = _k(q, x, s)
	K_xx = _k(x, x, s)
	return (K_qq, K_qx, K_xx)

def _Hqp(q, p, sigma) :
	"The hamiltonian, or kinetic energy of the shape q with momenta p."
	pKqp =  _k(q, q, sigma) * (p @ p.t())     # Use a simple isotropic kernel
	return .5 * pKqp.sum()              # $H(q,p) = \frac{1}{2} * sum_{i,j} k(x_i,x_j) p_i.p_j$
    
    
# Part 2 : Geodesic shooting ====================================================================
# The partial derivatives of the Hamiltonian are automatically computed !
def _dq_Hqp(q,p,sigma) : 
	return torch.autograd.grad(_Hqp(q,p,sigma), q, create_graph=True)[0]
def _dp_Hqp(q,p,sigma) :
	return torch.autograd.grad(_Hqp(q,p,sigma), p, create_graph=True)[0]


q0    = Variable(torch.from_numpy(    Q0.points ).type(dtype), requires_grad=False)
p0    = Variable(torch.from_numpy( 0.*Q0.points ).type(dtype), requires_grad=True )
s     = Variable(torch.from_numpy( 1.).type(dtype), requires_grad=False)

_dp_Hqp(q,p,1.)

