import theano
from pylearn2.space import Conv2DSpace
import theano.tensor as T
import warnings
io = None
from pylearn2.models.dbm import DBM
from pylearn2.models.dbm import VisibleLayer
from pylearn2.models import dbm


class SuperDBM(DBM):

    # Constructor is handled by DBM superclass

    def add_polyak_channels(self, param_to_mean, monitoring_dataset):
        """
            Hack to make Polyak averaging work.
        """

        for param in param_to_mean:
            if type(param) != type(param_to_mean[param]):
                print type(param)
                print type(param_to_mean[param])
                assert False

        X = self.get_input_space().make_batch_theano()
        if isinstance(self.hidden_layers[-1], dbm.Softmax):
            Y = T.matrix()

            Y_hat = self.mf(X)[-1]

            for var in theano.gof.graph.ancestors([Y_hat]):
                if var.owner is not None:
                    node = var.owner
                    inputs = []
                    for inp in node.inputs:
                        if inp in param_to_mean:
                            inputs.append(param_to_mean[inp])
                        else:
                            inputs.append(inp)
                    node.inputs = inputs

            new_ancestors = theano.gof.graph.ancestors([Y_hat])

            for param in param_to_mean:
                assert param not in new_ancestors

            pred = T.argmax(Y_hat, axis=1)
            true = T.argmax(Y, axis=1)

            err = T.cast(T.neq(pred, true).mean(), X.dtype)

            assert isinstance(monitoring_dataset, dict)

            for dataset_name in monitoring_dataset:
                d = monitoring_dataset[dataset_name]

                if dataset_name == '':
                    channel_name = 'polyak_err'
                else:
                    channel_name = dataset_name + '_polyak_err'

                self.monitor.add_channel(name = channel_name,
                        val = err,
                        ipt = (X,Y),
                        dataset = d)

    def expected_energy(self, V, mf_hidden):
        """
            V: a theano batch of visible unit observations
                (must be SAMPLES, not mean field parameters:
                    the random variables in the expectation are
                    the hiddens only)

            mf_hidden: a list, one element per hidden layer, of
                      batches of variational parameters
                (must be VARIATIONAL PARAMETERS, not samples.
                Layers with analytically determined variance parameters
                for their mean field parameters will use those to integrate
                over the variational distribution, so it's not generally
                the same thing as measuring the energy at a point.)

            returns: a vector containing the expected energy of
                    each example under the corresponding variational
                    distribution.
        """

        self.visible_layer.space.validate(V)
        assert isinstance(mf_hidden, (list, tuple))
        assert len(mf_hidden) == len(self.hidden_layers)

        terms = []

        terms.append(self.visible_layer.expected_energy_term(state = V, average=False))

        assert len(self.hidden_layers) > 0 # this could be relaxed, but current code assumes it

        terms.append(self.hidden_layers[0].expected_energy_term(
            state_below=self.visible_layer.upward_state(V), average_below=False,
            state=mf_hidden[0], average=True))

        for i in xrange(1, len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            layer_below = self.hidden_layers[i-1]
            mf_below = mf_hidden[i-1]
            mf_below = layer_below.upward_state(mf_below)
            mf = mf_hidden[i]
            terms.append(layer.expected_energy_term(state_below=mf_below, state=mf,
                average_below=True, average=True))

        assert len(terms) > 0

        rval = reduce(lambda x, y: x + y, terms)

        assert rval.ndim == 1
        return rval

    def setup_inference_procedure(self):
        if not hasattr(self, 'inference_procedure') or \
                self.inference_procedure is None:
            self.inference_procedure = SuperWeightDoubling()
            self.inference_procedure.set_dbm(self)


    def mf(self, *args, **kwargs):
        self.setup_inference_procedure()
        return self.inference_procedure.mf(*args, **kwargs)

    def do_inpainting(self, *args, **kwargs):
        self.setup_inference_procedure()
        return self.inference_procedure.do_inpainting(*args, **kwargs)



    def rao_blackwellize(self, layer_to_state):
        """
        layer_to_state:
            dict mapping layers to samples of that layer

        returns:
            layer_to_rao_blackwellized
                dict mapping layers to either samples or
                distributional samples.
                all hidden numbers with an even-numbered index
                become distributional.
        """

        layer_to_rao_blackwellized = {}

        # Copy over visible layer
        layer_to_rao_blackwellized[self.visible_layer] = layer_to_state[self.visible_layer]

        for i, layer in enumerate(self.hidden_layers):
            if i % 2 == 0:
                # Even numbered layer, make distributional
                if i == 0:
                    layer_below = self.visible_layer
                else:
                    layer_below = self.hidden_layers[i-1]
                state_below = layer_to_state[layer_below]
                state_below = layer_below.upward_state(state_below)

                if i + 1 < len(self.hidden_layers):
                    layer_above = self.hidden_layers[i+1]
                    state_above = layer_to_state[layer_above]
                    state_above = layer_above.downward_state(state_above)
                else:
                    layer_above = None
                    state_above = None

                distributional = layer.mf_update(state_below = state_below,
                                                state_above = state_above,
                                                layer_above = layer_above,
                                                iter_name = 'rao_blackwell')
                layer_to_rao_blackwellized[layer] = distributional
            else:
                # Odd numbered layer, copy over
                layer_to_rao_blackwellized[layer] = layer_to_state[layer]

        assert all([layer in layer_to_rao_blackwellized for layer in layer_to_state])
        assert all([layer in layer_to_state for layer in layer_to_rao_blackwellized])

        return layer_to_rao_blackwellized



    def reconstruct(self, V):

        H = self.mf(V)[0]

        downward_state = self.hidden_layers[0].downward_state(H)

        recons = self.visible_layer.inpaint_update(
                layer_above = self.hidden_layers[0],
                state_above = downward_state,
                drop_mask = None, V = None)

        return recons

    def score_matching(self, X):
        """

        Returns the score matching objective for this model on a batch of
        examples X.

        Note:
        Score matching is also implemented in pylearn2.costs.ebm_estimation.
        However, that implementation is generic and based on scan. This
        method tries to look up an efficient model-specific implementation.
        Also, the version in ebm_estimation assumes X is a matrix, but here
        it is a theano batch in the visible layer's Space.

        Note:
        This method is used in conjunction with the
        pylearn2.costs.cost.make_method_cost function, so it should
        mimic the call signature of UnsupervisedCost.__call__
        (there is no "model" argument here because "model" is now "self").
        This means that it is important that the variable be named X and
        not V in order for named argument calls to MethodCost objects to work.
        """

        # eventually we should try to look up the right implementation
        # for now we just check if it is the only case I have implemented

        if isinstance(self.visible_layer, GaussianConvolutionalVisLayer) and \
            len(self.hidden_layers) == 1 and \
            isinstance(self.hidden_layers[0], ConvMaxPool):

            # see uper_dbm_score_matching.lyx

            warnings.warn("super_dbm score matching is untested."
                    "the math in the lyx file has not been verified."
                    "there is no test that the code matches the lyx file."
                    "Bad results on CIFAR-10 full image and CIFAR-10 patches "
                    "suggest there is a bug. See galatea/sparsity/cifar_grbm_super_sm.yaml for easiest case to debug.")
            assert False # really not a good idea to run this except to debug it

            vis = self.visible_layer
            hid, = self.hidden_layers

            V = X
            assert V.ndim == 4

            P, H = hid.mf_update(state_below = V,
                    state_above = None,
                    double_weights = False,
                    iter_name = 'rbm')

            assert H is hid.downward_state( (P,H) )

            recons = hid.downward_message(H) + vis.mu
            assert recons.ndim == 4

            beta = vis.beta

            # this should be non-negative
            hid_stuff = H * (1. - H)
            #hid_stuff = Print('hid_stuff',attrs=['min'])(hid_stuff)

            # this should be non-negative
            vis_stuff =  hid.transformer.lmul_sq_T(hid_stuff)
            #vis_stuff = Print('vis_stuff',attrs=['min'])(vis_stuff)

            sq_beta = T.sqr(beta)

            # this should be non-negative
            first_term_presum = sq_beta *(0.5* T.square(V-recons)+vis_stuff)
            #first_term_presum = Print('first_term_presum',attrs=['min'])(first_term_presum)
            first_term = first_term_presum.sum(axis=(1,2,3)).mean()
            assert first_term.ndim == 0

            second_term = - beta.sum()
            #second_term = Print('second_term')(second_term)

            return first_term + second_term
        #end if gconv + convmaxpool
        raise NotImplementedError()
    #end score matching



class GaussianConvolutionalVisLayer(VisibleLayer):
    def __init__(self,
            rows,
            cols,
            channels,
            init_beta = 1.,
            min_beta = 1.,
            init_mu = 0.,
            tie_beta = None,
            tie_mu = None):
        """
            Implements a visible layer that is conditionally gaussian with
            diagonal variance. The layer lives in a Conv2DSpace.

            rows, cols, channels: the shape of the space

            init_beta: the initial value of the precision parameter
            min_beta: clip beta so it is at least this big (default 1)

            init_mu: the initial value of the mean parameter

            tie_beta: None or a string specifying how to tie beta
                      'locations' = tie beta across locations, ie
                                    beta should be a vector with one
                                    elem per channel
            tie_mu: None or a string specifying how to tie mu
                    'locations' = tie mu across locations, ie
                                  mu should be a vector with one
                                  elem per channel

        """

        self.__dict__.update(locals())
        del self.self

        self.space = Conv2DSpace(shape = [rows,cols], nchannels = channels)
        self.input_space = self.space

        origin = self.space.get_origin()
