__author__ = 'Yangqing Jia'
__status__ = 'Development'
__email__ = 'jiayq [at] eecs [dot] berkeley [dot] edu'
import numpy as np
import ctypes as ct
import os.path

_loss_c = np.ctypeslib.load_library('libloss.so', os.path.dirname(__file__))

_loss_c.gL_bnll.restype = ct.c_int
_loss_c.gL_bnll.argtypes = [ct.POINTER(ct.c_double), \
                            ct.POINTER(ct.c_double), \
                            ct.POINTER(ct.c_double), \
                            ct.c_int]

_loss_c.LgL_bnll.restype = ct.c_int
_loss_c.LgL_bnll.argtypes = [ct.POINTER(ct.c_double), \
                            ct.POINTER(ct.c_double), \
                            ct.POINTER(ct.c_double), \
                            ct.POINTER(ct.c_double), \
                            ct.c_int]

def gL_bnll_c(y,f, gL = None):
    if gL is None:
        gL = np.empty(y.shape)
    _loss_c.gL_bnll(y.ctypes.data_as(ct.POINTER(ct.c_double)),\
                    f.ctypes.data_as(ct.POINTER(ct.c_double)),\
                    gL.ctypes.data_as(ct.POINTER(ct.c_double)),\
                    ct.c_int(y.size))
    return gL

def LgL_bnll_c(y,f, L = None, gL = None):
    if L is None:
        L = np.empty(y.shape)
        gL = np.empty(y.shape)
    _loss_c.LgL_bnll(y.ctypes.data_as(ct.POINTER(ct.c_double)),\
                    f.ctypes.data_as(ct.POINTER(ct.c_double)),\
                    L.ctypes.data_as(ct.POINTER(ct.c_double)),\
                    gL.ctypes.data_as(ct.POINTER(ct.c_double)),\
                    ct.c_int(y.size))
    return L,gL


if __name__ == "__main__":
    from jiayq.utils.timer import Timer
    EXP_MAX = np.float64(100.0) # exp(x) for any x value larger than this will return exp(EXP_MAX)
    
    def exp_safe(x):
        '''
        compute the safe exp 
        '''
        return np.exp(np.minimum(x,EXP_MAX))
    
    def gL_bnll(y,f):
        '''
        The BNLL gradient
        '''
        expnyf = exp_safe(-y*f+1)
        return -y*expnyf / (1.0+expnyf)
        
    def LgL_bnll(y,f):
        '''
        jointly computing the loss and gradient is usually faster
        '''
        expnyf = exp_safe(-y*f+1)
        return np.log(expnyf+1.0), -y*expnyf / (expnyf+1.0)
    
    y = np.random.rand(10,10001)
    f = np.random.rand(10,10001)
    timer = Timer()
    for i in range(100):
        L,gL = LgL_bnll(y,f)
    print L[0,:10], gL[0,:10]
    print timer.lap()
    for i in range(100):
        LgL_bnll_c(y,f, L, gL)
    print L[0,:10], gL[0,:10]
    print timer.lap()
