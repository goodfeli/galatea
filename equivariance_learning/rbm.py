"""
Implementations of Restricted Boltzmann Machines and associated sampling
strategies.
"""
# Standard library imports
from itertools import izip

# Third-party imports
import numpy
N = numpy
import theano
from theano import tensor
T = tensor
from theano.tensor import nnet

# Local imports
from pylearn2.base import Block, StackedBlocks
from pylearn2.utils import as_floatX, safe_update, sharedX
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

    def __init__(self, nvis, nhid, energy_function_class, irange=0.5, rng=None,
                 mean_vis=False, init_sigma=2., learn_sigma=False,
                 sigma_lr_scale=1., init_bias_hid=0.0):
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
        super(GaussianBinaryRBM, self).__init__(nvis, nhid,
                                                irange, rng,
                                                init_bias_hid)

        self.learn_sigma = learn_sigma
        self.init_sigma = init_sigma
        self.sigma_lr_scale = float(sigma_lr_scale)

        if energy_function_class.supports_vector_sigma():
            base = N.ones(nvis)
        else:
            base = 1

        self.sigma_driver = sharedX(
            base * init_sigma / self.sigma_lr_scale,
            name='sigma',
            borrow=True
        )

        self.sigma = self.sigma_driver * self.sigma_lr_scale

        if self.learn_sigma:
            self._params.append(self.sigma_driver)

        self.mean_vis = mean_vis

        self.energy_function = energy_function_class(
                    W=self.weights,
                    sigma=self.sigma,
                    bias_vis=self.bias_vis,
                    bias_hid=self.bias_hid
                )

    def censor_updates(self, updates):
        if self.sigma_driver in updates:
            assert self.learn_sigma
            updates[self.sigma_driver] = T.clip(
                updates[self.sigma_driver],
                1e-5 / self.sigma_lr_scale,
                1e5 / self.sigma_lr_scale
            )

    def score(self, V):
        return self.energy_function.score(V)

    """
    method made obsolete by switching to energy function objects

    def input_to_h_from_v(self, v):
        ""
        Compute the affine function (linear map plus bias) that serves as
        input to the hidden layer in an RBM.

        Parameters
        ----------
        v  : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing one or several
            minibatches on the visible units, with the first dimension indexing
            training examples and the second indexing data dimensions.

        Returns
        -------
        a : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input to each
            hidden unit for each training example.

        Notes
        -----
        In the Gaussian-binary case, each data dimension is scaled by a sigma
        parameter (which defaults to 1 in this implementation, but is
        nonetheless present as a shared variable in the model parameters).
        ""
        if isinstance(v, tensor.Variable):
            return self.bias_hid + tensor.dot(v / self.sigma, self.weights)
        else:
            return [self.input_to_h_from_v(vis) for vis in v]"""

    def P_H_given_V(self, V):
        return self.energy_function.P_H_given(V)

    def mean_v_given_h(self, h):
        """
        Compute the mean activation of the visibles given hidden unit
        configurations for a set of training examples.

        Parameters
        ----------
        h  : tensor_like
            Theano symbolic representing the hidden unit states for a batch of
            training examples, with the first dimension indexing training
            examples and the second indexing hidden units.

        Returns
        -------
        vprime : tensor_like
            Theano symbolic representing the mean (deterministic)
            reconstruction of the visible units given the hidden units.
        """

        return self.energy_function.mean_v_given_h(h)
        #return self.bias_vis + self.sigma * tensor.dot(h, self.weights.T)

    def free_energy_given_v(self, V):
        """
        Calculate the free energy of a visible unit configuration by
        marginalizing over the hidden units.

        Parameters
        ----------
        v : tensor_like
            Theano symbolic representing the hidden unit states for a batch of
            training examples, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        f : tensor_like
            1-dimensional tensor representing the
            free energy of the visible unit configuration
            for each example in the batch
        """

        """hid_inp = self.input_to_h_from_v(v)
        squared_term = ((self.bias_vis - v) ** 2.) / (2. * self.sigma)
        rval =  squared_term.sum(axis=1) - nnet.softplus(hid_inp).sum(axis=1)
        assert len(rval.type.broadcastable) == 1"""

        return self.energy_function.free_energy(V)

    def free_energy(self, V):
        return self.energy_function.free_energy(V)
    #

    def sample_visibles(self, params, shape, rng):
        """
        Stochastically sample the visible units given hidden unit
        configurations for a set of training examples.

        Parameters
        ----------
        params : list
            List of the necessary parameters to sample :math:`p(v|h)`. In the
            case of a Gaussian-binary RBM this is a single-element list
            containing the conditional mean.

        Returns
        -------
        vprime : tensor_like
            Theano symbolic representing stochastic samples from :math:`p(v|h)`

        Notes
        -----
        If `mean_vis` is specified as `True` in the constructor, this is
        equivalent to a call to `mean_v_given_h`.
        """
        v_mean = params[0]
        if self.mean_vis:
            return v_mean
        else:
            # zero mean, std sigma noise
            zero_mean = rng.normal(size=shape) * self.sigma
            return zero_mean + v_mean


