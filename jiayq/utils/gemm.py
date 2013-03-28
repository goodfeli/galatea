import numpy as np

def mygemm(alpha,A,B,dtype=None,**kwargs):
    '''
    my gemm function that uses scipy fblas functions.
    '''
    from scipy.linalg.fblas import dgemm, sgemm
    if dtype is None:
        dtype=A.dtype
    if dtype != np.float32 and dtype != np.float64:
        print 'Error: this function cannot deal with such dtype.'
        exit()
    if not (A.flags['F_CONTIGUOUS'] or A.flags['C_CONTIGUOUS']) \
            or not (B.flags['F_CONTIGUOUS'] or B.flags['C_CONTIGUOUS']):
        print 'Matrices should either be C or F contiguous.'
        exit()
    if A.dtype != dtype:
        A=np.asarray(A,dtype=dtype)
    if B.dtype != dtype:
        B=np.asarray(B,dtype=dtype)
    if A.flags['F_CONTIGUOUS']:
        trans_a=0
    else:
        A=A.T
        trans_a=1
    if B.flags['F_CONTIGUOUS']:
        trans_b=0
    else:
        B=B.T
        trans_b=1
    if dtype==np.float32:
        return sgemm(alpha,A,B,trans_a=trans_a,trans_b=trans_b,**kwargs)
    else:
        return dgemm(alpha,A,B,trans_a=trans_a,trans_b=trans_b,**kwargs)

def mydot(A,B):
    '''
    a simple wrapper that mimics np.dot (if A and B are both matrices!)
    '''
    return mygemm(1.0,A,B)
