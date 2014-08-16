"""
MLP Layer objects related to the paper

Maxout Networks. Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
Courville, and Yoshua Bengio. ICML 2013.

If you use this code in your research, please cite this paper.

The objects in this module are Layer objects for use with
pylearn2.models.mlp.MLP. You need to make an MLP object in
order for these to do anything. For an example of how to build
an MLP with maxout hidden layers, see pylearn2/scripts/papers/maxout.

Note that maxout is designed for use with dropout, so you probably should
use dropout in your MLP when using these layers. If not using dropout, it
is best to use only 2 pieces per unit.

Note to developers / maintainers: when making changes to this module,
ensure that the changes do not break the examples in
pylearn2/scripts/papers/maxout.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"

import functools
import logging
import numpy as np
import warnings
from itertools import izip

from theano.compat.python2x import OrderedDict
from theano.sandbox import cuda
from theano import tensor as T

from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.mlp import Layer
from pylearn2.models.model import Model
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import py_integer_types
from pylearn2.utils import sharedX

from pylearn2.linear.conv2d_c01b import setup_detector_layer_c01b
from pylearn2.linear import local_c01b
if cuda.cuda_available:
    from pylearn2.sandbox.cuda_convnet.pool import max_pool_c01b
else:
    max_pool_c01b = None
from pylearn2.sandbox.cuda_convnet import check_cuda


logger = logging.getLogger(__name__)


class Discomax(Layer):
    """
    A hidden layer that does max pooling over groups of linear
    units. If you use this code in a research project, please
    cite

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013


    Parameters
    ----------
    layer_name : str
        A name for this layer that will be prepended to monitoring channels
        related to this layer. Each layer in an MLP must have a unique
        name.
    num_units : int
        The number of maxout units to use in this layer.
    num_pieces: int
        The number of linear pieces to use in each maxout unit.
    pool_stride : int, optional
        The distance between the start of each max pooling region. Defaults
        to num_pieces, which makes the pooling regions disjoint. If set to
        a smaller number, can do overlapping pools.
    randomize_pools : bool, optional
        If True, does max pooling over randomized subsets of the linear
        responses, rather than over sequential subsets.
    irange : float, optional
        If specified, initializes each weight randomly in
        U(-irange, irange)
    sparse_init : int, optional
        if specified, irange must not be specified.
        This is an integer specifying how many weights to make non-zero.
        All non-zero weights will be initialized randomly in
        N(0, sparse_stdev^2)
    sparse_stdev : float, optional
        WRITEME
    include_prob : float, optional
        probability of including a weight element in the set
        of weights initialized to U(-irange, irange). If not included
        a weight is initialized to 0. This defaults to 1.
    init_bias : float or ndarray, optional
        A value that can be broadcasted to a numpy vector.
        All biases are initialized to this number.
    W_lr_scale: float, optional
        The learning rate on the weights for this layer is multiplied by
        this scaling factor
    b_lr_scale: float, optional
        The learning rate on the biases for this layer is multiplied by
        this scaling factor
    max_col_norm: float, optional
        The norm of each column of the weight matrix is constrained to
        have at most this norm. If unspecified, no constraint. Constraint
        is enforced by re-projection (if necessary) at the end of each
        update.
    max_row_norm: float, optional
        Like max_col_norm, but applied to the rows.
    mask_weights: ndarray, optional
        A binary matrix multiplied by the weights after each update,
        allowing you to restrict their connectivity.
    min_zero: bool, optional
        If true, includes a zero in the set we take a max over for each
        maxout unit. This is equivalent to pooling over rectified
        linear units.
    """

    def __str__(self):
        """
        Returns
        -------
        rval : str
            A string representation of the object. In this case, just the
            class name.
        """
        return "Maxout"

    def __init__(self,
                 layer_name,
                 num_units,
                 num_pieces,
                 pool_stride=None,
                 randomize_pools=False,
                 irange=None,
                 sparse_init=None,
                 sparse_stdev=1.,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 max_col_norm=None,
                 max_row_norm=None,
                 mask_weights=None,
                 min_zero=False):

        super(Discomax, self).__init__()

        detector_layer_dim = num_units * num_pieces
        pool_size = num_pieces

        if pool_stride is None:
            pool_stride = pool_size

        self.__dict__.update(locals())
        del self.self

        self.b = sharedX(np.zeros((self.detector_layer_dim,)) + init_bias,
                         name=(layer_name + '_b'))
        self.ofs = sharedX(np.zeros((self.detector_layer_dim,)),
                         name=(layer_name + '_ofs'))

        if max_row_norm is not None:
            raise NotImplementedError()

    @functools.wraps(Model.get_lr_scalers)
    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """
        Tells the layer to use the specified input space.

        This resets parameters! The weight matrix is initialized with the
        size needed to receive input from this space.

        Parameters
        ----------
        space : Space
            The Space that the input will lie in.
        """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        if not (0 == ((self.detector_layer_dim - self.pool_size) %
                      self.pool_stride)):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. "
                                 "Should be divisible but remainder is %d" %
                                 (self.detector_layer_dim,
                                  self.pool_size,
                                  self.detector_layer_dim % self.pool_size))
            raise ValueError()

        self.h_space = VectorSpace(self.detector_layer_dim)
        self.pool_layer_dim = ((self.detector_layer_dim - self.pool_size) /
                               self.pool_stride + 1)
        self.output_space = VectorSpace(self.pool_layer_dim)

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.detector_layer_dim)) * \
                (rng.uniform(0., 1., (self.input_dim, self.detector_layer_dim))
                 < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.detector_layer_dim))

            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.

            for i in xrange(self.detector_layer_dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W, = self.transformer.get_params()
        assert W.name is not None

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            permute = np.zeros((self.detector_layer_dim,
                                self.detector_layer_dim))
            for j in xrange(self.detector_layer_dim):
                i = rng.randint(self.detector_layer_dim)
                permute[i, j] = 1
            self.permute = sharedX(permute)

        if self.mask_weights is not None:
            expected_shape = (self.input_dim, self.detector_layer_dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape " +
                                 str(expected_shape) +
                                 " but got " +
                                 str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def _modify_updates(self, updates):
        """
        Replaces the values in `updates` if needed to enforce the options set
        in the __init__ method, including `mask_weights` and `max_col_norm`.

        Parameters
        ----------
        updates : OrderedDict
            A dictionary mapping parameters (including parameters not
            belonging to this model) to updated values of those parameters.
            The dictionary passed in contains the updates proposed by the
            learning algorithm. This function modifies the dictionary
            directly. The modified version will be compiled and executed
            by the learning algorithm.
        """

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None

        if self.mask_weights is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

        if self.ofs in updates:
            updates[self.ofs] = T.clip(updates[self.ofs], 0., 1e6)

    @functools.wraps(Model.get_params)
    def get_params(self):
        assert self.b.name is not None
        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        rval.append(self.ofs)
        return rval

    @functools.wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @functools.wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.abs_(W).sum()

    @functools.wraps(Model.get_weights)
    def get_weights(self):
        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W, = self.transformer.get_params()
        W = W.get_value()

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            warnings.warn("randomize_pools makes get_weights multiply by the "
                          "permutation matrix. If you call set_weights(W) and "
                          "then call get_weights(), the return value will "
                          "WP not W.")
            P = self.permute.get_value()
            return np.dot(W, P)

        return W

    @functools.wraps(Layer.set_weights)
    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    @functools.wraps(Layer.set_biases)
    def set_biases(self, biases):
        self.b.set_value(biases)

    @functools.wraps(Layer.get_biases)
    def get_biases(self):
        return self.b.get_value()

    @functools.wraps(Model.get_weights_format)
    def get_weights_format(self):
        return ('v', 'h')

    @functools.wraps(Model.get_weights_view_shape)
    def get_weights_view_shape(self):
        total = self.detector_layer_dim
        cols = self.pool_size
        if cols == 1:
            # Let the PatchViewer decide how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total // cols
        if rows * cols < total:
            rows = rows + 1
        return rows, cols

    @functools.wraps(Model.get_weights_topo)
    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        # There was an implementation of this, but it was broken
        raise NotImplementedError()

    @functools.wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):
        warnings.warn("Layer.get_monitoring_channels is " +
                      "deprecated. Use get_layer_monitoring_channels " +
                      "instead. Layer.get_monitoring_channels " +
                      "will be removed on or after september 24th 2014",
                      stacklevel=2)

        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        row_norms_min = row_norms.min()
        row_norms_min.__doc__ = ("The smallest norm of any row of the "
                                 "weight matrix W. This is a measure of the "
                                 "least influence any visible unit has.")

        return OrderedDict([('row_norms_min',  row_norms_min),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

    @functools.wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state):
        warnings.warn("Layer.get_monitoring_channels_from_state is " +
                      "deprecated. Use get_layer_monitoring_channels " +
                      "instead. Layer.get_monitoring_channels_from_state " +
                      "will be removed on or after september 24th 2014",
                      stacklevel=2)

        P = state

        rval = OrderedDict()

        if self.pool_size == 1:
            vars_and_prefixes = [(P, '')]
        else:
            vars_and_prefixes = [(P, 'p_')]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over
            # e*x*amples" The x and u are included in the name because
            # otherwise its hard to remember which axis is which when reading
            # the monitor I use inner.outer rather than outer_of_inner or
            # something like that because I want mean_x.* to appear next to
            # each other in the alphabetical list, as these are commonly
            # plotted together
            for key, val in [('max_x.max_u', v_max.max()),
                             ('max_x.mean_u', v_max.mean()),
                             ('max_x.min_u', v_max.min()),
                             ('min_x.max_u', v_min.max()),
                             ('min_x.mean_u', v_min.mean()),
                             ('min_x.min_u', v_min.min()),
                             ('range_x.max_u', v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u', v_range.min()),
                             ('mean_x.max_u', v_mean.max()),
                             ('mean_x.mean_u', v_mean.mean()),
                             ('mean_x.min_u', v_mean.min())]:
                rval[prefix+key] = val

        return rval

    @functools.wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):

        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        row_norms_min = row_norms.min()
        row_norms_min.__doc__ = ("The smallest norm of any row of the "
                                 "weight matrix W. This is a measure of the "
                                 "least influence any visible unit has.")

        rval = OrderedDict([('row_norms_min',  row_norms_min),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

        if (state is not None) or (state_below is not None):
            if state is None:
                state = self.fprop(state_below)

            P = state
            if self.pool_size == 1:
                vars_and_prefixes = [(P, '')]
            else:
                vars_and_prefixes = [(P, 'p_')]

            for var, prefix in vars_and_prefixes:
                v_max = var.max(axis=0)
                v_min = var.min(axis=0)
                v_mean = var.mean(axis=0)
                v_range = v_max - v_min

                # max_x.mean_u is "the mean over *u*nits of the max over
                # e*x*amples" The x and u are included in the name because
                # otherwise its hard to remember which axis is which when
                # reading the monitor I use inner.outer
                # rather than outer_of_inner or
                # something like that because I want mean_x.* to appear next to
                # each other in the alphabetical list, as these are commonly
                # plotted together
                for key, val in [('max_x.max_u', v_max.max()),
                                 ('max_x.mean_u', v_max.mean()),
                                 ('max_x.min_u', v_max.min()),
                                 ('min_x.max_u', v_min.max()),
                                 ('min_x.mean_u', v_min.mean()),
                                 ('min_x.min_u', v_min.min()),
                                 ('range_x.max_u', v_range.max()),
                                 ('range_x.mean_u', v_range.mean()),
                                 ('range_x.min_u', v_range.min()),
                                 ('mean_x.max_u', v_mean.max()),
                                 ('mean_x.mean_u', v_mean.mean()),
                                 ('mean_x.min_u', v_mean.min())]:
                    rval[prefix+key] = val

        return rval

    @functools.wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        z = T.switch(z > 0., z + self.ofs, z)


        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = 0.
        else:
            p = None

        last_start = self.detector_layer_dim - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:, i:last_start+i+1:self.pool_stride]
            if p is None:
                p = cur
            else:
                p = T.maximum(cur, p)

        p.name = self.layer_name + '_p_'

        return p


