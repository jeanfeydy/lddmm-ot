NPOINTS = 10000

# Variable definitions, etc. for this minimal working example. ----------------------------------
# You don't need to read this to understand the bottlneck.
# Import the relevant tools
import numpy as np          # standard array library
import torch
from   torch.autograd import Variable

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Computation of the Kernel Matrix :
def _squared_distances(x, y) :
	"Returns the matrix of |x_i-y_j|^2."
	x_col = x.unsqueeze(1) ; y_lin = y.unsqueeze(0)
	return torch.sum( (x_col - y_lin)**2 , 2 )
def _k(x, y, s) :
	"Returns the matrix of k(x_i,y_j)."
	sq = _squared_distances(x, y) / (s**2)
	return torch.exp( -sq )


t       = np.linspace(0, 2*np.pi, NPOINTS)
weights = np.ones( (NPOINTS,) ) /NPOINTS

X  = Variable(torch.from_numpy( np.vstack([np.cos(t),   np.sin(t)]).T ).type(dtype), requires_grad=True )
Y  = Variable(torch.from_numpy( np.vstack([np.cos(t)+1, np.sin(t)]).T ).type(dtype), requires_grad=True )
P  = Variable(torch.from_numpy( weights                               ).type(dtype), requires_grad=False)
Q  = Variable(torch.from_numpy( weights                               ).type(dtype), requires_grad=False)

# The meaningful part of the program, aka "The Sinkhorn Algorithm" ------------------------------
# Iterating this simple "for" loop is an efficient way of finding the optimal scaling
# factors A and B such that  Gamma = diag(A)*K*diag(B)  has :
# - row-wise    sums equal to P
# - column-wise sums equal to Q
# Since the publication of the milestone paper :
# "Sinkhorn Distances: Lightspeed Computation of Optimal Transport", Marco Cuturi (2013),
# who proposed to use this scheme to compute Optimal Transport plans (Monge-Kantorovitch theory),
# this algorithm has generated a considerable interest in some applied maths and
# image registration communities.
#
# This algorithm showcases the typical features of state-of-the-art methods in the field
# of medical image registration (major conference on the topic : MICCAI) :
#
# - "small" datasets, but each image has a large memory footprint 
#   (say, 256x256x256 images, or >50,000 triangles for a segmented brain surface).
#
# - "diffeomorphic" (that is, tearing-free) registration algorithms, which iterate a simple 
#   linear algebra operation on the data. 
#   Here, we iterate a matrix-vector product + pointwise division,
#   but other pipelines will flow a vector field through time [t, t+dt] using an ODE step 
#   such as Euler (simplistic) or Runge-Kutta, etc.
#
# - output is a matching score, whose gradient shall be used to update the registration.
#

K  = _k( X, Y, .5) # Gaussian Kernel matrix (Npoints-by-Npoints), regularization parameter = .5
A  = Variable(torch.ones( NPOINTS ).type(dtype), requires_grad = False) 
B  = Variable(torch.ones( NPOINTS ).type(dtype), requires_grad = False) 

for it in range(1000) :
	A = P / (K     @ B)
	B = Q / (K.t() @ A)

Gamma = (A.unsqueeze(1) @ B.unsqueeze(0)) * K  # Gamma_ij = a_i * b_j * k( x_i , y_j )
Cost  = torch.sum( Gamma * _squared_distances(X,Y) )
# The above cost can be used as a distance measure between the unlabeled points clouds X and Y.
print(Cost)

# In practice, target Y is fixed and we're looking for the X-gradient of the Cost.
dX = torch.autograd.grad( Cost , X )
print(dX)


















