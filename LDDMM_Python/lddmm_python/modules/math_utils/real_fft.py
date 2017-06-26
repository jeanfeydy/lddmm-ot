import theano.tensor as T
from theano.tensor.fft import rfft, irfft
#from theano.gpuarray.fft import curfft as rfft
#from theano.gpuarray.fft import cuirfft as irfft
_rfft1d  = rfft
_irfft1d = irfft

def _rfft2d(x) :
    """
    Outputs a tensor3 of size (im.shape//2+1)+(4,).
    At each location, 4 real fourier transforms which respectively encode
    Re.Re, Re.Im, Im.Re, Im.Im.
    
    This routine is provided for you to understand its inverse the _irfft2d routine...
    """
    # First, take the real FFT along dimension 1
    f_x = _rfft1d(x).dimshuffle(1,0,2) # dimshuffle in order to take the FFT along dim 0
    f_x_re = _rfft1d(f_x[:,:,0]).dimshuffle(1,0,2)
    f_x_im = _rfft1d(f_x[:,:,1]).dimshuffle(1,0,2)
    return T.concatenate([f_x_re, f_x_im], axis = 2)
    
def _irfft2d(f_x) :
	"""
	Inverse of the _rfft2d(x) routine.
	"""
    f_x = f_x.dimshuffle(1,0,2)
    f_x_re = _irfft1d(f_x[:,:,0:2]).dimshuffle(1,0)
    f_x_im = _irfft1d(f_x[:,:,2:4]).dimshuffle(1,0)
    x      = _irfft1d( T.stack([f_x_re, f_x_im], axis = 2) )
    return(x)
    
    
