import numpy as np
import theano
import theano.tensor as T


def _squared_distances(x, y) :
	return (x ** 2).sum(1).reshape((x.shape[0], 1)) \
		 + (y ** 2).sum(1).reshape((1, y.shape[0])) \
		- 2* x.dot(y.T)
		
def _gaussian_kernel(x, y, s) :
	if type(s) is not list :
		s = [(1., s)]
	res = s[0][0] * T.exp(- _squared_distances(x,y) / (2 * s[0][1]**2) )
	for (coeff,sigm) in s[1:] :
		res = res + coeff * T.exp(- _squared_distances(x,y) / (2 * sigm**2) )
	return res
	
def _gaussian_cross_kernels(q, x, s) :
	K_qq = _gaussian_kernel(q, q, s)
	K_qx = _gaussian_kernel(q, x, s)
	K_xx = _gaussian_kernel(x, x, s)
	
	return (K_qq, K_qx, K_xx)
	







