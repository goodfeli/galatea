'''
PredNets, denoising autoencoders, and stacked DAEs.
'''
# Standard library imports
import functools
from itertools import izip
from collections import OrderedDict
import operator
import ipdb as pdb

# Third-party imports
import numpy
import theano
from theano import tensor
#from theano.tensor.nnet.conv import conv2d
#from theano.tensor.nnet.conv3d2d import conv3d

# Local imports
from pylearn2.base import Block, StackedBlocks
from pylearn2.models import Model
from pylearn2.models.mlp import MLP, SpaceConverter
from pylearn2.models.autoencoder import Autoencoder, SpaceAwareAutoencoderWrapper
from pylearn2.utils import sharedX
from pylearn2.utils.theano_graph import is_pure_elemwise
from pylearn2.utils.misc import Container
from pylearn2.space import Space, VectorSpace, Conv2DSpace, Conv3DSpace

#from pylearn2.costs.prednet import PredNetCost

theano.config.warn.sum_div_dimshuffle_bug = False
theano.config.compute_test_value = 'raise'
theano.config.exception_verbosity = 'high'


if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams


##################
# JBY: Copied from autoencoder.py / pred_convnet.py
##################

class PredNet(Model):
    '''
    Base class implementing Predictive Nets.
    '''

    def __init__(self, past_space, future_space, rep_space,
                 layers_up = None, layers_down = None,
                 layers_ae = None, layers_pred = None, seed = 1):
        '''
        Create a PredNet object.

        Parameters
        ----------
        input_space : Space
            input_space.shape = (time slices, ii size, jj size, channels) of data used for prediction.
        output_space : Space
            output_space.shape = (time slices (1?), ii size, jj size, channels) of data that will be predicted.
        layers_up : mlp
            An MLP representing the transformations on the way up. Note on denormalization: the input to the mlp must be of size prod(input_shape)!
        layers_down : mlp
            Ditto, donw. Denorm: ditto for output space.
        '''

        super(PredNet, self).__init__()   # does nothing

        assert isinstance(past_space, Conv3DSpace), 'past_space must be a Space'
        assert isinstance(future_space, Conv3DSpace), 'future_space must be a Space'
        past_dims = past_space.get_dim_dict()
        fut_dims = future_space.get_dim_dict()
        assert past_dims[1] == fut_dims[1], 'ii dimension must match'
        assert past_dims[2] == fut_dims[2], 'jj dimension must match'
        assert past_dims['c'] == fut_dims['c'], 'channels must match'

        # Set up spaces
        self.past_space = past_space
        self.future_space = future_space
        self.rep_space = rep_space

        self.past_timeslices = self.past_space.shape[0]
        self.future_timeslices = self.future_space.shape[0]
        self.total_timeslices = self.past_timeslices + self.future_timeslices
        # Input space is the entire timeline (i.e if we use 2
        # timesteps to predict 1, then input space spans all 3
        # timesteps)
        self.input_space = Conv3DSpace(shape = (self.total_timeslices,
                                                past_dims[1],
                                                past_dims[2]),
                                       num_channels = past_dims['c'])
        self.ouput_space = self.rep_space    # just a synonym

        # Set up layers
        # Figure out which mode we're in: 'mlp' or 'ae'
        if layers_up is not None:
            self.mode = 'mlp'
            assert isinstance(layers_up, MLP), 'layers_up must be an MLP'
            assert isinstance(layers_down, MLP), 'layers_down must be an MLP'
            assert layers_ae is None, 'Must specify up+down xor ae+pred'
            assert layers_pred is None, 'Must specify up+down xor ae+pred'
            self.layers_up = layers_up
            self.layers_down = layers_down
            # Add a layer to the down mlp to output in Conv3DSpace
            self.layers_down.add_layers([SpaceConverter('out_converter',
                                                       self.future_space)])
        else:
            self.mode = 'ae'
            assert isinstance(layers_ae, Autoencoder), 'layers_ae must be an Autoencoder'
            assert isinstance(layers_pred, MLP), 'layers_pred must be an MLP'
            assert layers_up is None, 'Must specify up+down xor ae+pred'
            assert layers_down is None, 'Must specify up+down xor ae+pred'
            assert layers_ae.input_space.dim == numpy.prod(self.past_space.shape), (
                'layers_ae.input_space.dim != numpy.prod(self.past_space.shape). You'
                ' must manually set this to be the same in the YAML file.'
                )
            self.layers_ae = SpaceAwareAutoencoderWrapper(
                layers_ae,
                self.past_space if not isinstance(self.past_space, VectorSpace) else None,
                self.rep_space if not isinstance(self.rep_space, VectorSpace) else None)
            self.layers_pred = layers_pred
            #self.ae_converter_in = SpaceConverter('ae_converter_in',
            #                                      self.layers_ae.input_space)
            #self.ae_converter_in.set_input_space(self.past_space)
            #self.ae_converter_out = SpaceConverter('ae_converter_out',
            #                                       self.past_space)
            #self.ae_converter_out.set_input_space(self.layers_ae.input_space)

        # Other...
        #self.rng = RandomStreams(seed)
        self.rng = numpy.random.RandomState(seed)

        assert self.mlp_mode != self.ae_mode, 'only one should be true'
        if self.mlp_mode:
            self._params = self.layers_up.get_params() + self.layers_down.get_params()
        else:
            self._params = self.layers_ae.get_params() + self.layers_pred.get_params()

        self._init_fns()

    def _init_fns(self):
        '''Here we create compiled versions of all relevant functions'''

        self.fns = Container()

        # 0. Create example input zero matrices
        # Size: ('b', 0, 1, 2, 'c')
        example_batch_size = 4
        example_past_shape = ((example_batch_size,) +              # e.g. 4
                              self.past_space.shape +              # e.g. (2,10,10)
                              (self.past_space.num_channels,))     # e.g. 1
        example_past = numpy.zeros(example_past_shape,
                                   dtype=theano.config.floatX)
        example_rep = numpy.zeros((example_batch_size, self.rep_space.dim),
                                  dtype=theano.config.floatX)

        # 1. Construct computation graph (internal to this instance)
        # Note: this part is messy. We should eventually encapsulate
        # scope not via ugly unique names, but by defining within a
        # function
        _past0 = theano.tensor.TensorType(theano.config.floatX,[False]*5)('_past0')
        _past0.tag.test_value = example_past
        _rep0 = self.past_to_representation(_past0)
        _rep0.name = '_rep0'
        self.fns.past_to_representation = theano.function(
            [_past0], _rep0,
            name = 'fns.past_to_representation')

        _rep1 = theano.tensor.matrix('_rep1')
        _rep1.tag.test_value = example_rep
        _fut1 = self.representation_to_future(_rep1)
        _fut1.name = '_fut1'
        self.fns.representation_to_future = theano.function(
            [_rep1], _fut1,
            name = 'fns.representation_to_future')

        _past2 = theano.tensor.TensorType(theano.config.floatX,[False]*5)('_past2')
        _past2.tag.test_value = example_past
        _fut2 = self.past_to_future(_past2)
        _fut2.name = '_fut2'
        self.fns.past_to_future = theano.function(
            [_past2], _fut2,
            name = 'fns.past_to_future')

        # 2. At this point, if compute_test_value mode is on, all
        # functions should have already been tested (when they were
        # defined). However, we can also test them ourselves (maybe
        # needed for GPU-only code too?)
        run_extra_tests = True
        if run_extra_tests:
            junk = self.fns.past_to_representation(example_past)
            junk = self.fns.representation_to_future(example_rep)
            junk = self.fns.past_to_future(example_past)

    def past_to_representation(self, past):
        '''
        Compute representation given past.

        Parameters
        ----------
        past : tensor_like
            Theano symbolic representing the input
            minibatch(es) to be encoded and reconstructed. Assumed to be
            5-tensors in the space self.past_space

        Returns
        -------
        representation : tensor_like
            Theano symbolic representing the corresponding
            representation for the past.
        '''

        if self.mlp_mode:
            representation = self.layers_up.fprop(past)
        else:
            #past_vec = self.ae_converter_in.fprop(past)
            representation = self.layers_ae.encode(past)
        representation.name = 'representation'

        return representation

        
    def representation_to_future(self, representation):
        '''
        Compute future given a representation of the past.

        Parameters
        ----------
        representation : tensor_like
            Theano symbolic representing the input
            minibatch(es) to be encoded and reconstructed. Assumed to be
            2-tensors in the space self.rep_space

        Returns
        -------
        future : tensor_like
            Theano symbolic representing the corresponding prediction of
            the future.
        '''

        if self.mlp_mode:
            future = self.layers_down.fprop(representation)
        else:
            future_representation = self.layers_pred.fprop(representation)
            future_representation.name = 'future_representation'

            #future_wide_vec = self.layers_ae.decode(future_representation)
            #future_wide_vec.name = 'future_wide_vec'
            #future_wide = self.ae_converter_out.fprop(future_wide_vec)
            #future_wide.name = 'future_wide'

            future_wide = self.layers_ae.decode(future_representation)
            future_wide.name = 'future_wide'

            future = self.slice_future_from_block(future_wide)
        future.name = 'future'

        return future

        
    def past_to_future(self, past):
        '''
        Predict future of input space from past input space.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded and reconstructed. Assumed to be
            2-tensors, with the first dimension indexing training examples and
            the second indexing data dimensions.

        Returns
        -------
        reconstructed : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            reconstructed minibatch(es) after encoding/decoding.
        '''

        representation = self.past_to_representation(past)
        future = self.representation_to_future(representation)
        
        return future
        

    def slice_future_from_block(self, data):
        '''Slice just the "future" segment from a wider block of time'''

        # data is (b, 0, 1, 2, c).  Time is dim 0.
        future = data[:,-self.future_timeslices:]
        future.name = 'future'

        return future
        
        
    def get_default_cost(self):
        raise Exception('Manually specify cost instead.')
        #return PredNetCost()


    def get_monitoring_channels(self, data):
        #print 'CALLED: PredNet.get_monitoring_channels'
        #pdb.set_trace()
        #ret = super(PredNet, self).get_monitoring_channels(data)
        #return ret

        #monitor_rng = RandomStreams(self._default_seed)

        return OrderedDict((
            ('mean_data_channel', tensor.mean(data)),
        ))


    def get_monitoring_data_specs(self):
        #print 'CALLED: PredNet.get_monitoring_data_specs'
        #pdb.set_trace()
        #ret = super(PredNet, self).get_monitoring_data_specs()
        ret = (self.get_input_space(), self.get_input_source())
        return ret


    @property
    def mlp_mode(self):
        return self.mode == 'mlp'

    
    @property
    def ae_mode(self):
        return self.mode == 'ae'

        

    # Use version defined in Model, rather than Block (which raises
    # NotImplementedError).
    get_input_space = Model.get_input_space
    get_output_space = Model.get_output_space


