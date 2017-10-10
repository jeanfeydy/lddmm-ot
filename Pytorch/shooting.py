# Import the relevant tools
import numpy as np          # standard array library
import torch

# No need for a ~/.theanorc file anymore !
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor

from kernel import _k

# Pytorch is a fantastic deep learning library : it transforms symbolic python code
# into highly optimized CPU/GPU binaries, which are called in the background seamlessly.
# It can be thought of as a "heir" to the legacy Theano library (RIP :'-( ):
# As you'll see, migrating a codebase from one to another is fairly simple !
#
# N.B. : On my Dell laptop, I have a GeForce GTX 960M with 640 Cuda cores and 2Gb of memory.
#
# We now show how to code a whole LDDMM pipeline into one (!!!) page of torch symbolic code.

# Part 1 : cometric on the space of landmarks, kinetic energy on the phase space (Hamiltonian)===

def _Hqp(q, p, sigma) :
	"The hamiltonian, or kinetic energy of the shape q with momenta p."
	pKqp =  _k(q, q, sigma) * (p @ p.t()) # Use a simple isotropic kernel
	return .5 * pKqp.sum()                # $H(q,p) = \frac{1}{2} * sum_{i,j} k(x_i,x_j) p_i.p_j$
    
# Part 2 : Geodesic shooting ====================================================================
# The partial derivatives of the Hamiltonian are automatically computed !
def _dq_Hqp(q,p,sigma) : 
	return torch.autograd.grad(_Hqp(q,p,sigma), q, create_graph=True)[0]
def _dp_Hqp(q,p,sigma) :
	return torch.autograd.grad(_Hqp(q,p,sigma), p, create_graph=True)[0]

def _hamiltonian_step(q,p, sigma) :
	"Simplistic euler scheme step with dt = .1."
	return [q + .1 * _dp_Hqp(q,p,sigma) ,  # See eq. 
			p - .1 * _dq_Hqp(q,p,sigma) ]

def _HamiltonianShooting(q, p, sigma) :
	"Shoots to time 1 a k-geodesic starting (at time 0) from q with momentum p."
	for t in range(10) :
		q,p = _hamiltonian_step(q, p, sigma) # Let's hardcode the "dt = .1"
	return [q,p]                             # and only return the final state + momentum

# Part 2bis : Geodesic shooting + deformation of the ambient space, for visualization ===========
def _HamiltonianCarrying(q, p, g, s) :
	"""
	Similar to _HamiltonianShooting, but also conveys information about the deformation of
	an arbitrary point cloud 'grid' in the ambient space.
	""" 
	for t in range(10) : # Let's hardcode the "dt = .1"
		q,p,g = [q + .1 * _dp_Hqp(q,p, s), 
		         p - .1 * _dq_Hqp(q,p, s), 
		         g + .1 * _k(g, q, s) @ p]
	return q,p,g         # return the final state + momentum + grid

















