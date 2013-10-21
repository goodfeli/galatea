__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"


import numpy as np
from theano.compat.python2x import OrderedDict
from theano import function
from theano.gof.op import get_debug_values
from theano import tensor as T

from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.mlp import Layer
from pylearn2.models.mlp import Linear
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX

from pylearn2.sandbox.cuda_convnet import check_cuda
from pylearn2.linear.conv2d_c01b import setup_detector_layer_c01b
from pylearn2.utils import py_integer_types
from theano.sandbox import cuda
if cuda.cuda_available:
    from pylearn2.sandbox.cuda_convnet.pool import max_pool_c01b


class RestrictedMaxout(Layer):
    """
    A hidden layer that does max pooling over groups of linear
    units. If you use this code in a research project, please
    cite

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013
    """

    def __init__(self,
                 layer_name,
                 num_units,
                 num_pieces,
                 irange = None,
                 islope = .05,
                 sparse_init = None,
                 sparse_stdev = 1.,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 max_row_norm = None,
                 slope_lr_scale = None,
                 min_zero = False
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
            min_zero: If true, includes a zero in the set we take a max over
                    for each maxout unit. This is equivalent to pooling over
                    rectified linear units.
        """

        detector_layer_dim = num_units
        pool_size = num_pieces


        self.__dict__.update(locals())
        del self.self

        self.b = [sharedX( np.zeros((self.detector_layer_dim,)) + init_bias, name = layer_name + '_b(%d)' % i) \
                for i in xrange(self.num_pieces)]


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

        if self.slope_lr_scale is not None:
            for slope in self.slopes:
                rval[slope] = self.slope_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)


        self.h_space = VectorSpace(self.detector_layer_dim)
        self.output_space = VectorSpace(self.detector_layer_dim)

        rng = self.mlp.rng

        self.slopes = [sharedX(rng.uniform(-self.islope, self.islope, (self.detector_layer_dim)),
            name = self.layer_name + '_slopes(%d)' % i) for i in xrange(self.num_pieces)]

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
                return False
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

        # Apply constraints from start
        identity_updates = OrderedDict()

        for param in self.get_params():
            identity_updates[param] = param

        self.censor_updates(identity_updates)

        f = function([], updates=identity_updates)
        f()

    def censor_updates(self, updates):

        W ,= self.transformer.get_params()

        if W in updates:
            updated_W = updates[W]
            col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
            desired_norms = 1.
            updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

    def get_params(self):
        assert not any([b.name is None for b in self.b])
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval = rval + self.b + self.slopes
        return rval

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
        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W ,= self.transformer.get_params()
        W = W.get_value()

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
        raise NotImplementedError()


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

        slopes = T.concatenate(self.slopes)

        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ('slope_min', slopes.min()),
                            ('slope_max', slopes.max())
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

        z = self.transformer.lmul(state_below)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = T.zeros_like(z)
        else:
            p = None

        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.num_pieces):
            cur = z * self.slopes[i] + self.b[i]
            if p is None:
                p = cur
            else:
                p = T.maximum(cur, p)

        p.name = self.layer_name + '_p_'

        return p

class DSI_ReLU(Linear):
    """
    Direction-Slope-Intercept parameterization of ReLU
    """

    def __init__(self, islope, **kwargs):
        assert 'max_col_norm' not in kwargs
        assert 'max_row_norm' not in kwargs
        super(DSI_ReLU, self).__init__(**kwargs)
        self.islope = islope


    def get_params(self):
        return super(DSI_ReLU, self).get_params() + [self.slopes]

    def set_input_space(self, space):

        super(DSI_ReLU, self).set_input_space(space)

        rng = self.mlp.rng

        self.slopes = sharedX(rng.uniform(-self.islope, self.islope, (self.dim)), 'slopes')

        # Apply constraints from start
        identity_updates = OrderedDict()

        for param in self.get_params():
            identity_updates[param] = param

        self.censor_updates(identity_updates)

        f = function([], updates=identity_updates)
        f()

    def censor_updates(self, updates):

        W ,= self.transformer.get_params()

        if W in updates:
            updated_W = updates[W]
            col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
            desired_norms = 1.
            updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

    def fprop(self, state_below):
        assert not self.copy_input
        assert not self.softmax_columns
        z = self.transformer.lmul(state_below) * self.slopes + self.b
        h = z * (z > 0.)
        return h

    def cost(self, *args, **kwargs):
        raise NotImplementedError()


class ConvReMU(Layer):
    """
    This uses the C01B ("channels", topological axis 0,
    topological axis 1, "batch") format of tensors for input
    and output.

    The back-end is Alex Krizhevsky's cuda-convnet library,
    so it is extremely fast, but requires a GPU.
    """

    def __init__(self,
                 num_channels,
                 num_pieces,
                 kernel_shape,
                 pool_shape,
                 pool_stride,
                 layer_name,
                 islope,
                 slope_lr_scale = None,
                 irange = None,
                 sparse_init = None,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 pad = 0,
                 fix_pool_shape = False,
                 fix_pool_stride = False,
                 fix_kernel_shape = False,
                 partial_sum = 1,
                 tied_b = False,
                 input_normalization = None,
                 detector_normalization = None,
                 min_zero = False,
                 output_normalization = None,
                 kernel_stride=(1, 1)):
        """
            num_channels: The number of output channels the layer should have.
                          Note that it must internally compute num_channels * num_pieces
                          convolution channels.
            num_pieces:   The number of linear pieces used to make each maxout unit.
            kernel_shape: The shape of the convolution kernel.
            pool_shape:   The shape of the spatial max pooling. A two-tuple of ints.
                          This is redundant as cuda-convnet requires the pool shape to
                          be square.
            pool_stride:  The stride of the spatial max pooling. Also must be square.
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            pad: The amount of zero-padding to implicitly add to the boundary of the
                image when computing the convolution. Useful for making sure pixels
                at the edge still get to influence multiple hidden units.
            fix_pool_shape: If True, will modify self.pool_shape to avoid having
                pool shape bigger than the entire detector layer.
                If you have this on, you should probably also have
                fix_pool_stride on, since the pool shape might shrink
                smaller than the stride, even if the stride was initially
                valid.
                The "fix" parameters are useful for working with a hyperparameter
                optimization package, which might often propose sets of hyperparameters
                that are not feasible, but can easily be projected back into the feasible
                set.
            fix_kernel_shape: if True, will modify self.kernel_shape to avoid
            having the kernel shape bigger than the implicitly
            zero padded input layer

            partial_sum: a parameter that controls whether to prefer runtime savings
                        or memory savings when computing the gradient with respect to
                        the kernels. See pylearn2.sandbox.cuda_convnet.weight_acts.py
                        for details. The default is to prefer high speed.
                        Note that changing this setting may change the value of computed
                        results slightly due to different rounding error.
            tied_b: If true, all biases in the same channel are constrained to be the same
                    as each other. Otherwise, each bias at each location is learned independently.
            max_kernel_norm: If specifed, each kernel is constrained to have at most this norm.
            input_normalization, detector_normalization, output_normalization:
                if specified, should be a callable object. the state of the network is optionally
                replaced with normalization(state) at each of the 3 points in processing:
                    input: the input the layer receives can be normalized right away
                    detector: the maxout units can be normalized prior to the spatial pooling
                    output: the output of the layer, after sptial pooling, can be normalized as well
            kernel_stride: vertical and horizontal pixel stride between
                           each detector.
        """
        check_cuda(str(type(self)))

        detector_channels = num_channels

        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            for b in self.b:
                rval[b] = self.b_lr_scale

        if self.slope_lr_scale is not None:
            for slope in self.slopes:
                rval[slope] = self.slope_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        setup_detector_layer_c01b(layer=self,
                input_space=space,
                rng=self.mlp.rng)

        rng = self.mlp.rng

        detector_shape = self.detector_space.shape

        def handle_pool_shape(idx):
            if self.pool_shape[idx] < 1:
                raise ValueError("bad pool shape: " + str(self.pool_shape))
            if self.pool_shape[idx] > detector_shape[idx]:
                if self.fix_pool_shape:
                    assert detector_shape[idx] > 0
                    self.pool_shape[idx] = detector_shape[idx]
                else:
                    raise ValueError("Pool shape exceeds detector layer shape on axis %d" % idx)

        map(handle_pool_shape, [0, 1])

        assert self.pool_shape[0] == self.pool_shape[1]
        assert self.pool_stride[0] == self.pool_stride[1]
        assert all(isinstance(elem, py_integer_types) for elem in self.pool_stride)
        if self.pool_stride[0] > self.pool_shape[0]:
            if self.fix_pool_stride:
                warnings.warn("Fixing the pool stride")
                ps = self.pool_shape[0]
                assert isinstance(ps, py_integer_types)
                self.pool_stride = [ps, ps]
            else:
                raise ValueError("Stride too big.")
        assert all(isinstance(elem, py_integer_types) for elem in self.pool_stride)

        dummy_detector = sharedX(self.detector_space.get_origin_batch(2)[0:16,:,:,:])

        dummy_p = max_pool_c01b(c01b=dummy_detector, pool_shape=self.pool_shape,
                                pool_stride=self.pool_stride,
                                image_shape=self.detector_space.shape)
        dummy_p = dummy_p.eval()
        self.output_space = Conv2DSpace(shape=[dummy_p.shape[1], dummy_p.shape[2]],
                                        num_channels = self.num_channels, axes = ('c', 0, 1, 'b') )

        print 'Output space: ', self.output_space.shape


        if self.tied_b:
            self.b = [sharedX(np.zeros((self.detector_space.num_channels)) + self.init_bias,
                name = 'bias_%d' % i)
                    for i in xrange(self.num_pieces)]
        else:
            self.b = [sharedX(self.detector_space.get_origin() + self.init_bias,
                name = 'bias_%d' % i)
                    for i in xrange(self.num_pieces)]

        rng = self.mlp.rng
        self.slopes = [sharedX(rng.uniform(-self.islope, self.islope, (self.detector_space.num_channels)),
            name='slope_%d' % i) for i in xrange(self.num_pieces)]


        # Apply constraints from start
        identity_updates = OrderedDict()

        for param in self.get_params():
            identity_updates[param] = param

        self.censor_updates(identity_updates)

        f = function([], updates=identity_updates)
        f()

    def censor_updates(self, updates):

        W ,= self.transformer.get_params()
        if W in updates:
            updated_W = updates[W]
            row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(0,1,2)))
            desired_norms = 1
            updates[W] = updated_W * (desired_norms / (1e-7 + row_norms)).dimshuffle('x', 'x', 'x', 0)

    def get_params(self):
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval) + self.b + self.slopes
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_topo(self):
        raw = self.transformer.get_weights_topo()
        slopes = [slope.get_value() for slope in self.slopes]
        shape = list(raw.shape)
        num_pieces = len(slopes)
        shape[0] *= num_pieces

        raw_T = np.transpose(raw, (1, 2, 3, 0))
        stack_T = [raw_T * slope for slope in slopes]
        stack = [np.transpose(elem, (3, 0, 1, 2)) for elem in stack_T]

        rval = np.concatenate(stack, axis=2)

        return rval

    def get_weights_view_shape(self):
        raise NotImplementedError()

        return (self.num_channels, self.num_pieces)

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(0,1,2)))

        all_slopes = T.concatenate(self.slopes)

        slope_abs = abs(all_slopes)
        slope_abs_min = slope_abs.min()
        slope_abs_mean = slope_abs.mean()
        slope_abs_max = slope_abs.max()

        return OrderedDict([
                            ('kernel_norms_min'  , row_norms.min()),
                            ('kernel_norms_mean' , row_norms.mean()),
                            ('kernel_norms_max'  , row_norms.max()),
                            ('slope_abs_min', slope_abs_min),
                            ('slope_abs_mean', slope_abs_mean),
                            ('slope_abs_max', slope_abs_max)
                            ])

    def fprop(self, state_below):
        check_cuda(str(type(self)))

        self.input_space.validate(state_below)

        if not hasattr(self, 'input_normalization'):
            self.input_normalization = None

        if self.input_normalization:
            state_below = self.input_normalization(state_below)

        # Alex's code requires # input channels to be <= 3 or a multiple of 4
        # so we add dummy channels if necessary
        if not hasattr(self, 'dummy_channels'):
            self.dummy_channels = 0
        if self.dummy_channels > 0:
            state_below = T.concatenate((state_below,
                                         T.zeros_like(state_below[0:self.dummy_channels, :, :, :])),
                                        axis=0)

        z = self.transformer.lmul(state_below)


        self.detector_space.validate(z)

        assert self.detector_space.num_channels % 16 == 0

        def cross_channel_pool(lin):
            rval = None
            for i in xrange(self.num_pieces):
                b = self.b[i]
                slope = self.slopes[i]

                slope = slope.dimshuffle(0, 'x', 'x', 'x')

                if self.tied_b:
                    b = b.dimshuffle(0, 'x', 'x', 'x')
                else:
                    b = b.dimshuffle(0, 1, 2, 'x')

                z = lin * slope + b

                if rval is None:
                    rval = z
                else:
                    rval = T.maximum(rval, z)
            return rval


        if self.output_space.num_channels % 16 == 0:
            # alex's max pool op only works when the number of channels
            # is divisible by 16. we can only do the cross-channel pooling
            # first if the cross-channel pooling preserves that property
            z = cross_channel_pool(z)

            if self.detector_normalization:
                z = self.detector_normalization(z)

            p = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
        else:

            if self.detector_normalization is not None:
                raise NotImplementedError("We can't normalize the detector "
                        "layer because the detector layer never exists as a "
                        "stage of processing in this implementation.")
            z = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
            p = cross_channel_pool(z)


        self.output_space.validate(p)

        if hasattr(self, 'min_zero') and self.min_zero:
            p = p * (p > 0.)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p



    def get_monitoring_channels_from_state(self, state):

        P = state

        rval = OrderedDict()

        vars_and_prefixes = [ (P,'') ]

        for var, prefix in vars_and_prefixes:
            assert var.ndim == 4
            v_max = var.max(axis=(1,2,3))
            v_min = var.min(axis=(1,2,3))
            v_mean = var.mean(axis=(1,2,3))
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

