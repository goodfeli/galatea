import numpy as np
import theano.tensor as T
from theano.sandbox.linalg import alloc_diag
from theano.sandbox.linalg import matrix_inverse

raise NotImplementedError("Doesn't seem to work")

class GP(object):
    """ Works only on scalars """

    def __init__(self, beta):
        self.beta = beta
        self.kernel = BasicKernel()

    def expected_new_y(self, x, y, new_x):
        assert new_x.ndim == 0
        beta = alloc_diag(T.alloc(1., (x.shape[0],)) * self.beta)
        C = self.kernel.gram_matrix(x) + beta
        C_inv = matrix_inverse(C)
        k = self.kernel(x, new_x)
        return T.dot(k, T.dot(C_inv, y))

class BasicKernel(object):
    """

    K(x, y) = 1 + (x-y)^2

    """

    def gram_matrix(self, x):
        x_sq = T.sqr(x)
        return 1. + x_sq.dimshuffle(0,'x')+x_sq.dimshuffle('x',0) - 2. * T.outer(x,x)

    def __call__(self, x, new_x):
        assert x.ndim == 1
        assert new_x.ndim == 0
        return 1. + T.sqr(x-new_x)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from theano import function

    m = 10
    degree = 5
    rng = np.random.RandomState([2012, 10, 22])
    x = T.vector('x')
    x.tag.test_value = np.zeros((m,),dtype=x.dtype)
    w = rng.uniform(-1., 1., (degree+1,)).astype(x.dtype)
    X = [ x.dimshuffle(0, 'x') ** float(p) for p in xrange(degree + 1) ]
    X = T.concatenate(X, axis=1)
    y = T.dot(X,w)

    new_x = T.scalar('new_x')
    new_x.tag.test_value = np.zeros((),dtype=x.dtype)
    new_y = GP(100.).expected_new_y(x, y, new_x)

    f = function([x, new_x], new_y)
    yf = function([x], y)

    x = rng.randn(m).astype(x.dtype)
    y = yf(x)

    new_x = np.linspace(-2.,2.,100).astype(x.dtype)

    new_y = np.asarray([ f(x, nx) for nx in new_x ])

    print new_x.shape, new_y.shape


    plt.scatter(x,y)
    plt.hold(True)
    plt.plot(new_x, new_y)
    plt.show()


