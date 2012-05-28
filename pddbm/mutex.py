import numpy as np
import theano.tensor as T
from pylearn2.models.rbm import RBM
from pylearn2.expr.nnet import inverse_sigmoid_numpy
from theano import function

D = 12

def int_to_bits(n, num_bits = D):
    assert isinstance(n,int)
    assert n >= 0
    assert n < 2 ** num_bits

    place = num_bits - 1

    bits = []

    while place >= 0:
        bits.append( int(((2 ** place) & n) > 0) )
        place -= 1

    return bits

all_states = []

for i in xrange(2 ** D):
    all_states.append(int_to_bits(i))

all_states = np.asarray(all_states, dtype = 'float32')

good_states = []

for i in xrange(D):
    good_states.append(int_to_bits(2 ** i) )

good_states = np.asarray(good_states, dtype = 'float32')


all_states_var = T.matrix()

rbm = RBM( nvis = D, nhid = D)

Z = T.exp( - rbm.free_energy_given_v(all_states_var) ).sum()

good_states_var = T.matrix()

good_prob = T.exp( - rbm.free_energy_given_v(good_states_var) ).sum() / Z

good_prob_func = function([all_states_var, good_states_var], good_prob)

def run_rbm( pos_weight = 1., neg_weight = 1., bias_hid = -1.,
             bias_vis = inverse_sigmoid_numpy( 1. / float(D) ) ):
    rbm.bias_vis.set_value( np.ones( (D,), dtype='float32') *  bias_vis)
    rbm.bias_hid.set_value( np.ones( (D,), dtype='float32') * bias_hid)
    rbm.transformer._W.set_value( np.identity(D, dtype='float32') * \
            (pos_weight + neg_weight) - np.ones( (D,D), dtype = 'float32') \
            * neg_weight)

    return float(good_prob_func(all_states, good_states))
