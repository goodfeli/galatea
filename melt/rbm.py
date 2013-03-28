"""
Implementations of Restricted Boltzmann Machines and associated sampling
strategies.
"""
# Standard library imports
from itertools import izip

# Third-party imports
import numpy as np
import theano
from theano import tensor
T = tensor
from theano.tensor import nnet

# Local imports
from pylearn2.base import Block, StackedBlocks
from pylearn2.utils import as_floatX, sharedX
from pylearn2.models.model import Model
theano.config.warn.sum_div_dimshuffle_bug = False

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams

from pylearn2.models.rbm import GaussianBinaryRBM


class GaussianBinaryEVRBM(GaussianBinaryRBM):
    """
    An RBM with Gaussian visible units and binary hidden units
    that learns transformations to which it is equivariant
    """

    def __init__(self, nvis, energy_function_class, irange=0.5, rng=None,
                 mean_vis=False, init_sigma=2., learn_sigma=False,
                 sigma_lr_scale=1., init_bias_hid=0.0, num_templates = 400,
                 num_compositions = []):
        """
        Allocate a GaussianBinaryRBM object.

        Parameters
        ----------
        nvis : int
            Number of visible units in the model.
        nhid : int
            Number of hidden units in the model.
        energy_function_class:
            TODO: finish comment
        irange : float, optional
            The size of the initial interval around 0 for weights.
        rng : RandomState object or seed
            NumPy RandomState object to use when initializing parameters
            of the model, or (integer) seed to use to create one.
        mean_vis : bool, optional
            Don't actually sample visibles; make sample method simply return
            mean.
        init_sigma : scalar (TODO: ?)
            Initial value of the sigma variable.
        init_bias_hid : scalar or 1-d array of length `nhid`
            Initial value for the biases on hidden units.
        """

        nhid = num_templates

        for comp in num_compositions:
            nhid *= comp

        if rng is None:
            rng = np.random.RandomState([1,2,3])

        self.templates = sharedX( rng.uniform(-irange,irange,(nvis,num_templates)))

        self.transforms = []

        weight_pieces = [ self.templates ]

        for comp  in num_compositions:
            transform = sharedX( rng.uniform(-irange, irange, (nvis, nvis) ) )

            for i in xrange(comp):
                new_weight_pieces = []

                for weight_piece in weight_pieces:
                    new_weight_pieces.append(T.dot(transform, weight_piece))

                for weight_piece in new_weight_pieces:
                    weight_pieces.append(weight_piece)

        weights = T.concatenate( weight_pieces, axis = 1)


        super(GaussianBinaryEVRBM, self).__init__(nvis = nvis, nhid = nhid,
                energy_function_class = energy_function_class,
                                                irange = irange, rng = rng,
                                                mean_vis = mean_vis, init_sigma = init_sigma, learn_sigma = learn_sigma,
                                                sigma_lr_scale = sigma_lr_scale,
                                                init_bias_hid = init_bias_hid, weights = weights)

        assert self.weights is weights

        found = False
        for i, param in enumerate(self.params):
            if param is weights:
                del self.params[i]
                found = True
                break
        assert found

        self.params.append(self.templates)

        for transform in self.transforms:
            self.transforms.append(transform)

