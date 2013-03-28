import theano.tensor as T
x = T.scalar()
from theano.sandbox.rng_mrg import MRG_RandomStreams
theano_rng = MRG_RandomStreams(42)

