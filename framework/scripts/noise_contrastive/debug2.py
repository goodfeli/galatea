import cPickle
import numpy as N
from theano import function, scan, shared
import theano.tensor as T
from theano.printing import Print
from theano.tensor.shared_randomstreams import RandomStreams
import theano
floatX = theano.config.floatX


class Model(object):
    def reset_rng(self):
        self.rng = N.random.RandomState([12.,9.,2.])
        self.theano_rng = RandomStreams(self.rng.randint(2**30))
        if self.initialized:
            self.redo_theano()

    def __init__(self):
        self.initialized = False
        self.reset_rng()
        self.sigma_sq = shared(N.cast[floatX] (1.0))

        self.redo_theano()
        
    def redo_theano(self):
        X = T.vector()
        X.name = 'X'

        X = Print('X')(X)

        corrupted = self.theano_rng.normal(size = X.shape, avg = X,
                                    std = 1.)

        corrupted = Print('corrupted')(corrupted)

        self.debug_func = function([X],corrupted )

    def debug(self,x):
        self.theano_rng.seed(5)
        return self.debug_func(x)

model = Model()

x = N.asarray([1])

first = model.debug(x)
second = model.debug(x)
assert first == second
            
model.redo_theano()
third = model.debug(x)

assert first == third
