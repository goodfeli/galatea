import numpy

def pd_check(A):
    """
    Check whether a matrix is positive definite.

    Because it breaks down for large matrices, we check for 10 random
    100*100 submatrices.
    We have a pretty lenient PD check because we do not want to miss any
    kernel matrices.

    :type  A: numpy.array
    :param A: matrix to test
    """
    (n,n1) = A.shape

    # Matrix not square!
    if n != n1:
        return False

    # Matrix not symmetric!
    if ((A - A.T) != 0).any():
        return False

    if n > 100:
        RP = numpy.zeros((10, 100))
        for k in numpy.arange(10):
            RP[k,:] = numpy.permutation(n)[:100]
    else:
        RP = numpy.array([numpy.arange(n)])

    tol = -1e-15

    for k in numpy.arange(RP.shape[0]):
        idx = RP[k,:]

        # With numpy, indexing this way
        # directly returns the diagonal
        D = A[idx,idx]

        m = numpy.max(D)
        if m <= 0: 
            return False

        if (min(D)/m) < tol:
            return False

    return True



        

 
