__author__ = 'Yangqing Jia'
__status__ = 'Development'
__email__ = 'jiayq [at] eecs [dot] berkeley [dot] edu'
import numpy as np
import ctypes as ct
import os.path

_fastmax_cpp = np.ctypeslib.load_library('libfastmax.so', os.path.dirname(__file__))

#int normalizev(double * a, int n, double mean, double std)
_fastmax_cpp.normalizev.restype = ct.c_int
_fastmax_cpp.normalizev.argtypes = [ct.POINTER(ct.c_double), \
                                    ct.c_int, \
                                    ct.c_double, \
                                    ct.c_double]

_fastmax_cpp.fastmaxm.restype = ct.c_int
_fastmax_cpp.fastmaxm.argtypes = [ct.POINTER(ct.c_double), \
                                  ct.POINTER(ct.c_double), \
                                  ct.POINTER(ct.c_int), \
                                  ct.c_int,\
                                  ct.c_int]

def normalizev(vector, mean, std):
    _fastmax_cpp.normalizev(vector.ctypes.data_as(ct.POINTER(ct.c_double)),\
                            ct.c_int(len(vector)),\
                            ct.c_double(mean),\
                            ct.c_double(std))

def fastmaxm(matrix, rowids, out):
    '''
    This function carries out the fast max for a subset of 
    the rows of a matrix. rowids indicates which rows we are
    going to compute the max over. out is a vector - make
    sure its memory is aligned! otherwise the program will
    simply segfault. If you create out using numpy, usually 
    it will be aligned already.
    '''
    rowids = rowids.astype(ct.c_int)
    _fastmax_cpp.fastmaxm(out.ctypes.data_as(ct.POINTER(ct.c_double)),\
                          matrix.ctypes.data_as(ct.POINTER(ct.c_double)),\
                          rowids.ctypes.data_as(ct.POINTER(ct.c_int)),\
                          ct.c_int(len(rowids)),\
                          ct.c_int(matrix.shape[1]))

    return out

if __name__ == "__main__":
    from jiayq.utils.timer import Timer
    # test fast max
    matrix = np.random.rand(10000,10001)
    rowids = np.random.randint(10000,size=10)
    timer = Timer()
    for i in range(1000):
        out = np.max(matrix[rowids],axis=0)
    print out
    print timer.lap()
    temp = np.zeros((2,10001))
    outmisalign = temp[0]
    for i in range(1000):
        fastmaxm(matrix,rowids,outmisalign)
    print outmisalign
    print timer.lap()

    # test normalize
    m = np.mean(out)
    std = np.std(out)
    timer.lap()
    out -= m
    out /= std
    print timer.lap()
    print out
    timer.lap()
    normalizev(outmisalign, m, std)
    print timer.lap()
    print outmisalign
