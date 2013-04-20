from collections import OrderedDict
import numpy as np

from theano import tensor as T

from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.mlp import Layer
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX


class MLP_GatedRectifier(Layer):
    """
    A hidden layer that does max pooling over groups of linear
    units. If you use this code in a research project, please
    cite

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. arXiv 2013
    """

    def __init__(self,
                 layer_name,
                 num_units,
                 num_pieces,
                 gater = None,
                 pool_stride = None,
                 randomize_pools = False,
                 irange = None,
                 sparse_init = None,
                 sparse_stdev = 1.,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 max_col_norm = None,
                 max_row_norm = None,
                 mask_weights = None,
        ):
        """
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            num_units: The number of maxout units to use in this layer.
            num_pieces: The number of linear pieces to use in each maxout
                        unit.
            pool_stride: The distance between the start of each max pooling
                        region. Defaults to num_pieces, which makes the
                        pooling regions disjoint. If set to a smaller number,
                        can do overlapping pools.
            randomize_pools: Does max pooling over randomized subsets of
                        the linear responses, rather than over sequential
                        subsets.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            sparse_init: if specified, irange must not be specified.
                        This is an integer specifying how many weights to make
                        non-zero. All non-zero weights will be initialized
                        randomly in N(0, sparse_stdev^2)
            include_prob: probability of including a weight element in the set
               of weights initialized to U(-irange, irange). If not included
               a weight is initialized to 0. This defaults to 1.
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            max_col_norm: The norm of each column of the weight matrix is
                constrained to have at most this norm. If unspecified, no
                constraint. Constraint is enforced by re-projection (if
                necessary) at the end of each update.
            max_row_norm: Like max_col_norm, but applied to the rows.
            mask_weights: A binary matrix multiplied by the weights after each
                         update, allowing you to restrict their connectivity.
            min_zero: If true, includes a zero in the set we take a max over
                    for each maxout unit. This is equivalent to pooling over
                    rectified linear units.
        """

        detector_layer_dim = num_units * num_pieces
        pool_size = num_pieces

        if pool_stride is None:
            pool_stride = pool_size

        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((self.detector_layer_dim,)) + init_bias, name = layer_name + '_b')


        if max_row_norm is not None:
            raise NotImplementedError()

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
        """ Note: this resets parameters! """

        self.input_space = space

        assert self.gater.get_input_space() == space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)


        if not ((self.detector_layer_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        self.h_space = VectorSpace(self.detector_layer_dim)
        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size)/ self.pool_stride + 1
        self.output_space = VectorSpace(self.pool_layer_dim)

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.detector_layer_dim)) * \
                (rng.uniform(0.,1., (self.input_dim, self.detector_layer_dim))
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

        W ,= self.transformer.get_params()
        assert W.name is not None

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            permute = np.zeros((self.detector_layer_dim, self.detector_layer_dim))
            for j in xrange(self.detector_layer_dim):
                i = rng.randint(self.detector_layer_dim)
                permute[i,j] = 1
            self.permute = sharedX(permute)

        if self.mask_weights is not None:
            expected_shape =  (self.input_dim, self.detector_layer_dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def censor_updates(self, updates):

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None

        if self.mask_weights is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval + self.gater.get_params()

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.abs(W).sum()

    def get_weights(self):

        print 'Which weights? '
        print 'g) gater'
        print 'm) main network'

        x = raw_input()

        if x == 'g':
            return self.gater.get_weights()
        assert x == 'm'

        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W ,= self.transformer.get_params()
        W = W.get_value()

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            warnings.warn("randomize_pools makes get_weights multiply by the permutation matrix. "
                    "If you call set_weights(W) and then call get_weights(), the return value will "
                    "WP not W.")
            P = self.permute.get_value()
            return np.dot(W,P)

        print W.shape

        return W

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

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
        print total, rows, cols
        return rows, cols


    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        # There was an implementation of this, but it was broken
        raise NotImplementedError()

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])


    def get_monitoring_channels_from_state(self, state):

        P = state

        rval = OrderedDict()

        if self.pool_size == 1:
            vars_and_prefixes = [ (P,'') ]
        else:
            vars_and_prefixes = [ (P, 'p_') ]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            # The x and u are included in the name because otherwise its hard
            # to remember which axis is which when reading the monitor
            # I use inner.outer rather than outer_of_inner or something like that
            # because I want mean_x.* to appear next to each other in the alphabetical
            # list, as these are commonly plotted together
            for key, val in [
                             ('max_x.max_u', v_max.max()),
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
                             ('mean_x.min_u', v_mean.min())
                             ]:
                rval[prefix+key] = val

        return rval

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        z = T.clip(z, 0., 1e30)
        p = None

        gate = self.gater.fprop(state_below)

        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride] * gate[:, i].dimshuffle(0, 'x')
            if p is None:
                p = cur
            else:
                p = cur + p

        p.name = self.layer_name + '_p_'

        return p

