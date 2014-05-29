
class MaxoutConvC01B(Layer):
    """
    Maxout units arranged in a convolutional layer, with
    spatial max pooling on top of the maxout. If you use this
    code in a research project, please cite

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013


    This uses the C01B ("channels", topological axis 0,
    topological axis 1, "batch") format of tensors for input
    and output.

    The back-end is Alex Krizhevsky's cuda-convnet library,
    so it is extremely fast, but requires a GPU.

    Parameters
    ----------
    num_channels : int
        The number of output channels the layer should have.
        Note that it must internally compute num_channels * num_pieces
        convolution channels.
    num_pieces : int
        The number of linear pieces used to make each maxout unit.
    kernel_shape : tuple
        The shape of the convolution kernel.
    pool_shape : tuple
        The shape of the spatial max pooling. A two-tuple of ints.
        This is redundant as cuda-convnet requires the pool shape to
        be square.
    pool_stride : tuple
        The stride of the spatial max pooling. Also must be square.
    layer_name : str
        A name for this layer that will be prepended to
        monitoring channels related to this layer.
    irange : float, optional
        if specified, initializes each weight randomly in
        U(-irange, irange)
    init_bias : float, optional
        All biases are initialized to this number
    W_lr_scale : float, optional
        The learning rate on the weights for this layer is
        multiplied by this scaling factor
    b_lr_scale : float, optional
        The learning rate on the biases for this layer is
        multiplied by this scaling factor
    pad : int, optional
        The amount of zero-padding to implicitly add to the boundary of the
        image when computing the convolution. Useful for making sure pixels
        at the edge still get to influence multiple hidden units.
    fix_pool_shape : bool, optional
        If True, will modify self.pool_shape to avoid having
        pool shape bigger than the entire detector layer.
        If you have this on, you should probably also have
        fix_pool_stride on, since the pool shape might shrink
        smaller than the stride, even if the stride was initially
        valid.
        The "fix" parameters are useful for working with a hyperparameter
        optimization package, which might often propose sets of
        hyperparameters that are not feasible, but can easily be projected
        back into the feasible set.
    fix_pool_stride : bool, optional
        WRITEME
    fix_kernel_shape : bool, optional
        if True, will modify self.kernel_shape to avoid having the kernel
        shape bigger than the implicitly zero padded input layer
    partial_sum : int, optional
        a parameter that controls whether to prefer runtime savings
        or memory savings when computing the gradient with respect to
        the kernels. See pylearn2.sandbox.cuda_convnet.weight_acts.py
        for details. The default is to prefer high speed.
        Note that changing this setting may change the value of computed
        results slightly due to different rounding error.
    tied_b : bool, optional
        If true, all biases in the same channel are constrained to be the
        same as each other. Otherwise, each bias at each location is
        learned independently.
    max_kernel_norm : float, optional
        If specified, each kernel is constrained to have at most this norm.
    input_normalization : callable, optional
        see output normalization
    detector_normalization : callable, optional
        see output normalization
    min_zero : bool, optional
        WRITEME
    output_normalization : callable, optional
        if specified, should be a callable object. the state of the
        network is optionally replaced with normalization(state) at each
        of the 3 points in processing:

          - input: the input the layer receives can be normalized right
            away
          - detector: the maxout units can be normalized prior to the
            spatial pooling
          - output: the output of the layer, after sptial pooling,
            can be normalized as well
    kernel_stride : tuple, optional
        vertical and horizontal pixel stride between each detector.
    """

    def __init__(self,
                 num_channels,
                 num_pieces,
                 kernel_shape,
                 pool_shape,
                 pool_stride,
                 layer_name,
                 irange=None,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 pad=0,
                 fix_pool_shape=False,
                 fix_pool_stride=False,
                 fix_kernel_shape=False,
                 partial_sum=1,
                 tied_b=False,
                 max_kernel_norm=None,
                 input_normalization=None,
                 detector_normalization=None,
                 min_zero=False,
                 output_normalization=None,
                 kernel_stride=(1, 1)):
        check_cuda(str(type(self)))
        super(MaxoutConvC01B, self).__init__()

        detector_channels = num_channels * num_pieces

        self.__dict__.update(locals())
        del self.self

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

        This resets parameters! The kernel tensor is initialized with the
        size needed to receive input from this space.

        Parameters
        ----------
        space : Space
            The Space that the input will lie in.
        """

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
                    raise ValueError("Pool shape exceeds detector layer shape "
                                     "on axis %d" % idx)

        map(handle_pool_shape, [0, 1])

        assert self.pool_shape[0] == self.pool_shape[1]
        assert self.pool_stride[0] == self.pool_stride[1]
        assert all(isinstance(elem, py_integer_types)
                   for elem in self.pool_stride)
        if self.pool_stride[0] > self.pool_shape[0]:
            if self.fix_pool_stride:
                warnings.warn("Fixing the pool stride")
                ps = self.pool_shape[0]
                assert isinstance(ps, py_integer_types)
                self.pool_stride = [ps, ps]
            else:
                raise ValueError("Stride too big.")
        assert all(isinstance(elem, py_integer_types)
                   for elem in self.pool_stride)

        dummy_detector = sharedX(self.detector_space.get_origin_batch(2)[0:16,
                                                                         :,
                                                                         :,
                                                                         :])

        dummy_p = max_pool_c01b(c01b=dummy_detector,
                                pool_shape=self.pool_shape,
                                pool_stride=self.pool_stride,
                                image_shape=self.detector_space.shape)
        dummy_p = dummy_p.eval()
        self.output_space = Conv2DSpace(shape=[dummy_p.shape[1],
                                               dummy_p.shape[2]],
                                        num_channels=self.num_channels,
                                        axes=('c', 0, 1, 'b'))

        logger.info('Output space: {0}'.format(self.output_space.shape))

    def _modify_updates(self, updates):
        """
        Replaces the values in `updates` if needed to enforce the options set
        in the __init__ method, including `max_kernel_norm`.

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

        if self.max_kernel_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(0, 1, 2)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                scales = desired_norms / (1e-7 + row_norms)
                updates[W] = (updated_W * scales.dimshuffle('x', 'x', 'x', 0))

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
        return rval

    @functools.wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

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

    @functools.wraps(Model.get_weights_topo)
    def get_weights_topo(self):
        return self.transformer.get_weights_topo()

    @functools.wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):

        W, = self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(0, 1, 2)))

        return OrderedDict([('kernel_norms_min',  row_norms.min()),
                            ('kernel_norms_mean', row_norms.mean()),
                            ('kernel_norms_max',  row_norms.max()), ])

    @functools.wraps(Layer.fprop)
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
            zeros = T.zeros_like(state_below[0:self.dummy_channels, :, :, :])
            state_below = T.concatenate((state_below, zeros), axis=0)

        z = self.transformer.lmul(state_below)
        if not hasattr(self, 'tied_b'):
            self.tied_b = False
        if self.tied_b:
            b = self.b.dimshuffle(0, 'x', 'x', 'x')
        else:
            b = self.b.dimshuffle(0, 1, 2, 'x')

        z = z + b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        self.detector_space.validate(z)

        assert self.detector_space.num_channels % 16 == 0

        if self.output_space.num_channels % 16 == 0:
            # alex's max pool op only works when the number of channels
            # is divisible by 16. we can only do the cross-channel pooling
            # first if the cross-channel pooling preserves that property
            if self.num_pieces != 1:
                s = None
                for i in xrange(self.num_pieces):
                    t = z[i::self.num_pieces, :, :, :]
                    if s is None:
                        s = t
                    else:
                        s = T.maximum(s, t)
                z = s

            if self.detector_normalization:
                z = self.detector_normalization(z)

            p = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
        else:

            if self.detector_normalization is not None:
                raise NotImplementedError("We can't normalize the detector "
                                          "layer because the detector layer "
                                          "never exists as a stage of "
                                          "processing in this implementation.")
            z = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
            if self.num_pieces != 1:
                s = None
                for i in xrange(self.num_pieces):
                    t = z[i::self.num_pieces, :, :, :]
                    if s is None:
                        s = t
                    else:
                        s = T.maximum(s, t)
                z = s
            p = z

        self.output_space.validate(p)

        if hasattr(self, 'min_zero') and self.min_zero:
            p = p * (p > 0.)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p

    @functools.wraps(Model.get_weights_view_shape)
    def get_weights_view_shape(self):
        total = self.detector_channels
        cols = self.num_pieces
        if cols == 1:
            # Let the PatchViewer decide how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total // cols
        if rows * cols < total:
            rows = rows + 1
        return rows, cols

    @functools.wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state):

        P = state

        rval = OrderedDict()

        vars_and_prefixes = [(P, '')]

        for var, prefix in vars_and_prefixes:
            assert var.ndim == 4
            v_max = var.max(axis=(1, 2, 3))
            v_min = var.min(axis=(1, 2, 3))
            v_mean = var.mean(axis=(1, 2, 3))
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over
            # e*x*amples" The x and u are included in the name because
            # otherwise its hard to remember which axis is which when reading
            # the monitor I use inner.outer rather than outer_of_inner or
            # something like that because I want mean_x.* to appear next to
            # each other in the alphabetical list, as these are commonly
            # plotted together
            for key, val in [('max_x.max_u',    v_max.max()),
                             ('max_x.mean_u',   v_max.mean()),
                             ('max_x.min_u',    v_max.min()),
                             ('min_x.max_u',    v_min.max()),
                             ('min_x.mean_u',   v_min.mean()),
                             ('min_x.min_u',    v_min.min()),
                             ('range_x.max_u',  v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u',  v_range.min()),
                             ('mean_x.max_u',   v_mean.max()),
                             ('mean_x.mean_u',  v_mean.mean()),
                             ('mean_x.min_u',   v_mean.min())]:
                rval[prefix+key] = val

        return rval
