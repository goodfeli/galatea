"""A Multivariate Normal Distribution."""
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from pylearn2.utils import sharedX
import numpy as np
N = np

class AdditiveMaskedDiagonalMND:

    def __init__(self, init_beta, nvis, prob):
        """ A conditional distribution that adds
        gaussian noise with diagonal precision
        matrix beta to another variable that it
        conditions on
        """

        self.__dict__.update(locals())
        del self.self

        self.beta = sharedX(np.ones((nvis,))*init_beta)
        assert self.beta.ndim == 1

        self.s_rng = RandomStreams(17)

    def random_design_matrix(self, X):
        """ X: a theano variable containing a design matrix
        of observations of the random vector to condition on."""
        Z = self.s_rng.normal(size=X.shape,
                              avg=0., std=1./T.sqrt(self.beta), dtype=config.floatX)

        mask = self.s_rng.binomial(size=X.shape, n = 1, p = self.prob, dtype=config.floatX)

        return X+mask*Z

    def is_symmetric(self):
        """ A property of conditional distributions
        P(Y|X)
        Return true if P(y|x) = P(x|y) for all x,y
        """

        return True

class BitFlip:

    def __init__(self,  nvis, prob):
        """ A conditional distribution that flips
        bits
        """

        self.__dict__.update(locals())
        del self.self

        self.s_rng = RandomStreams(17)

    def random_design_matrix(self, X):

        flip =  self.s_rng.binomial(size=X.shape, n = 1, p = self.prob, dtype=config.floatX)

        return X * (1-flip) + (1-X)*flip

    def is_symmetric(self):
        """ A property of conditional distributions
        P(Y|X)
        Return true if P(y|x) = P(x|y) for all x,y
        """

        return True
