__author__ = 'Yangqing Jia'
__status__ = 'Development'
__email__ = 'jiayq [at] eecs [dot] berkeley [dot] edu'
import numpy as np
import ctypes as ct
import os.path

_fastop_cpp = np.ctypeslib.load_library('libfastop.so', os.path.dirname(__file__))

#int fastmeanstd(double *a, int n, double * pmean, double * pstd)
_fastop_cpp.fastmeanstd.restype = ct.c_int
_fastop_cpp.fastmeanstd.argtypes = [ct.POINTER(ct.c_double), \
                                    ct.c_int, \
                                    ct.POINTER(ct.c_double)]

_fastop_cpp.normalizev.restype = ct.c_int
_fastop_cpp.normalizev.argtypes = [ct.POINTER(ct.c_double), \
                                    ct.c_int, \
                                    ct.c_double, \
                                    ct.c_double]

_fastop_cpp.normalizev.restype = ct.c_int
_fastop_cpp.normalizev.argtypes = [ct.POINTER(ct.c_double), \
                                    ct.c_int, \
                                    ct.c_double, \
                                    ct.c_double]

_fastop_cpp.fastmaxm.restype = ct.c_int
_fastop_cpp.fastmaxm.argtypes = [ct.POINTER(ct.c_double), \
                                  ct.POINTER(ct.c_double), \
                                  ct.POINTER(ct.c_int), \
                                  ct.c_int,\
                                  ct.c_int]
_fastop_cpp.fastsumm.restype = ct.c_int
_fastop_cpp.fastsumm.argtypes = [ct.POINTER(ct.c_double), \
                                  ct.POINTER(ct.c_double), \
                                  ct.POINTER(ct.c_int), \
                                  ct.c_int,\
                                  ct.c_int]
_fastop_cpp.fastcenters.restype = ct.c_int
_fastop_cpp.fastcenters.argtypes = [ct.POINTER(ct.c_double), \
                                  ct.POINTER(ct.c_double), \
                                  ct.POINTER(ct.c_int), \
                                  ct.POINTER(ct.c_int), \
                                  ct.c_int,\
                                  ct.c_int,\
                                  ct.c_int]
_fastop_cpp.fastmaximums.restype = ct.c_int
_fastop_cpp.fastmaximums.argtypes = [ct.POINTER(ct.c_double), \
                                  ct.POINTER(ct.c_double), \
                                  ct.POINTER(ct.c_int), \
                                  ct.POINTER(ct.c_int), \
                                  ct.c_int,\
                                  ct.c_int,\
                                  ct.c_int]

def fastmeanstd(vector):
    meanstd = np.empty(2)
    _fastop_cpp.fastmeanstd(vector.ctypes.data_as(ct.POINTER(ct.c_double)),\
                            ct.c_int(vector.size),\
                            meanstd.ctypes.data_as(ct.POINTER(ct.c_double)))
    return meanstd[0],meanstd[1]

def normalizev(vector, mean, std):
    _fastop_cpp.normalizev(vector.ctypes.data_as(ct.POINTER(ct.c_double)),\
                            ct.c_int(len(vector)),\
                            ct.c_double(mean),\
                            ct.c_double(std))
                            
def fastop(matrix,rowids,cppfunc,out=None):
    if out is None:
        out = np.empty(matrix.shape[1])
    rowids = rowids.astype(ct.c_int)
    if rowids.size == matrix.shape[0]:
        # in this case, the rowids is specified as a 0-1 indicator
        cppfunc(out.ctypes.data_as(ct.POINTER(ct.c_double)),\
                          matrix.ctypes.data_as(ct.POINTER(ct.c_double)),\
                          rowids.ctypes.data_as(ct.POINTER(ct.c_int)),\
                          ct.c_int(-matrix.shape[0]),\
                          ct.c_int(matrix.shape[1]))
    else:
        cppfunc(out.ctypes.data_as(ct.POINTER(ct.c_double)),\
                          matrix.ctypes.data_as(ct.POINTER(ct.c_double)),\
                          rowids.ctypes.data_as(ct.POINTER(ct.c_int)),\
                          ct.c_int(len(rowids)),\
                          ct.c_int(matrix.shape[1]))
    
    return out
    
def fastmaxm(matrix, rowids, out=None):
    '''
    This function carries out the fast max for a subset of 
    the rows of a matrix. rowids indicates which rows we are
    going to compute the max over. out is a vector - make
    sure its memory is aligned! otherwise the program will
    simply segfault. If you create out using numpy, usually 
    it will be aligned already.
    '''
    return fastop(matrix,rowids,_fastop_cpp.fastmaxm, out)

def fastsumm(matrix, rowids, out=None):
    '''
    This function carries out the fast max for a subset of 
    the rows of a matrix. rowids indicates which rows we are
    going to compute the max over. out is a vector - make
    sure its memory is aligned! otherwise the program will
    simply segfault. If you create out using numpy, usually 
    it will be aligned already.
    '''
    return fastop(matrix,rowids,_fastop_cpp.fastsumm, out)
    
def fastcenters(matrix, idx, k, centers = None):
    '''
    this function is essentially the center computation step for k-means
    it returns two values: centers and counts, where centers are the 
    clustering centers and counts are the number of members per center.
    '''
    #int fastcenters(double* M, double* C, int* Ccounts, int* idx, int nrows, int ncols, int nctrs)
    if centers is None:
        centers = np.empty((k, matrix.shape[1]),dtype=np.float64)
    counts = np.empty(k, dtype=ct.c_int)
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    # just in case
    idx = idx.astype(ct.c_int)
    _fastop_cpp.fastcenters(matrix.ctypes.data_as(ct.POINTER(ct.c_double)),\
                            centers.ctypes.data_as(ct.POINTER(ct.c_double)),\
                            counts.ctypes.data_as(ct.POINTER(ct.c_int)),\
                            idx.ctypes.data_as(ct.POINTER(ct.c_int)),\
                            ct.c_int(matrix.shape[0]),\
                            ct.c_int(matrix.shape[1]),\
                            ct.c_int(k))
    return centers, counts

def fastmaximums(matrix, idx, k, centers = None):
    '''
    this function is essentially the maximum computation step for maxpooling
    it returns two values: centers and counts, where centers are the 
    clustering centers and counts are the number of members per center.
    '''
    #int fastcenters(double* M, double* C, int* Ccounts, int* idx, int nrows, int ncols, int nctrs)
    if centers is None:
        centers = np.empty((k, matrix.shape[1]),dtype=np.float64)
    counts = np.empty(k, dtype=ct.c_int)
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    # just in case
    idx = idx.astype(ct.c_int)
    _fastop_cpp.fastmaximums(matrix.ctypes.data_as(ct.POINTER(ct.c_double)),\
                            centers.ctypes.data_as(ct.POINTER(ct.c_double)),\
                            counts.ctypes.data_as(ct.POINTER(ct.c_int)),\
                            idx.ctypes.data_as(ct.POINTER(ct.c_int)),\
                            ct.c_int(matrix.shape[0]),\
                            ct.c_int(matrix.shape[1]),\
                            ct.c_int(k))
    return centers, counts
