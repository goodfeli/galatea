import numpy as N


def pd_check(A):
    # Check whether a matrix is positive semi-definite

    """ Because it breaks down for large matrices, we check for 10 random
        100*100 submatrices.
        We have a pretty lenient PD check because we do not want to miss any
        kernel matrices.   """

    if 'kernelized' in dir(A) and A.kernelized:
        return True

    if 'X' in dir(A):
        return pd_check(A.X)
    #



    debug = False
    n, n1 = A.shape
    posdef = False

    if n != n1:
        if debug:
            print 'pd_check: Matrix not square!'
        #
        return False
    #

    if N.abs(A-A.T).max() > 0.0:
        if debug:
            print 'pd_check: Matrix not symmetric!'
        #
        return False
    #

    #print 'pd_check got through initial checks'
    #assert False

    if n > 100:
        RP = N.zeros( (10, 100)  ,dtype='uint32')
        for k in xrange(10): 
            print 'pd_check calling randperm'
            rp = make_learning_curve.randperm(n)
            RP[k,:] = rp[0:100]
        #
    else:
        RP = N.zeros( (1, n), dtype='uint32')
        RP[0,:] = range(n)
    #

    tol = -1e-15;

    for k in xrange(RP.shape[0]):
        idx=RP[k,:]
        #print 'A.shape: '+str(A.shape)
        a = N.zeros((idx.shape[0],idx.shape[0]))
        for i in xrange(a.shape[0]):
            for j in xrange(a.shape[0]):
                a[i,j] = A[idx[i],idx[j]]
            #
        #
        D = a.diagonal()
        M = D.max()
        if M<=0:
            return False
        #

        f = D.min()/M

        if f < tol:
            return False
        #
    #
    return True
#

import make_learning_curve

