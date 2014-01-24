#! /usr/bin/env python

# Standard library imports
#import functools
#from itertools import izip
#from collections import OrderedDict
#import operator
import ipdb as pdb

# Third-party imports
import numpy
import theano
#from theano import tensor
#from theano.tensor.nnet.conv import conv2d
#from theano.tensor.nnet.conv3d2d import conv3d

# Local imports
#from pylearn2.models import Model
from pylearn2.models.mlp import MLP
#from pylearn2.models.mlp import MLP, SpaceConverter
#from pylearn2.models.autoencoder import Autoencoder, SpaceAwareAutoencoderWrapper
#from pylearn2.utils import sharedX
#from pylearn2.utils.theano_graph import is_pure_elemwise
#from pylearn2.utils.misc import Container
#from pylearn2.space import Space, VectorSpace, Conv2DSpace

#from pylearn2.costs.prednet import PredNetCost

#theano.config.warn.sum_div_dimshuffle_bug = False
#theano.config.compute_test_value = 'raise'
#theano.config.exception_verbosity = 'high'


class Container(object):
    pass



class FishMLP(MLP):
    '''
    A slight extension of MLP.
    '''

    def __init__(self, layers, num_frames, batch_size=None,
                 input_space=None, nvis=None, seed=None):
        '''Takes the same arguments as MLP, except also requires
        num_frames: the number of consecutive frames to expect in each
        batch.
        '''

        super(FishMLP, self).__init__(layers,
                                      batch_size = batch_size,
                                      input_space = input_space,
                                      nvis = nvis,
                                      seed = seed)

        self.num_frames = num_frames
        assert num_frames > 0, 'num_frames must be positive'

        self._init_fns()

    def get_target_source(self):
        '''Override default Model.get_target_source'''
        return ('targets1', 'targets2')
    
    def _init_fns(self):
        '''Create compiled versions of all relevant functions'''

        # HACK
        self.names_to_del = getattr(self, 'names_to_del', set())
        self._test_batch_size = getattr(self, '_test_batch_size', 2)
        
        self.register_names_to_del('fns')

        self.fns = Container()

        # 0. Create example inputs/outputs
        # Size: ('c', 0, 1, 'b')
        example_batch_size = self.get_test_batch_size()  # e.g. 2
        id_space, xy_space = self.get_output_space().components      # Hard coded order here!!
        assert id_space.dim in (10,25), 'maybe wrong order of spaces?'
        assert xy_space.dim in (4,6),   'maybe wrong order of spaces?'

        # 0.1 Shapes
        example_feat_shape = ((self.input_space.num_channels,) +  # e.g. 1
                              self.input_space.shape +            # e.g. (156,156)
                              (example_batch_size,))              # e.g. 2
        example_id_shape = (example_batch_size, id_space.dim)    # e.g. (2,25)
        example_xy_shape = (example_batch_size, xy_space.dim)    # e.g. (2,4)
        # 0.2 Make zero matrices
        example_feat = numpy.zeros(example_feat_shape,
                                   dtype=theano.config.floatX)
        example_id = numpy.zeros(example_id_shape,
                                 dtype=theano.config.floatX)
        example_xy = numpy.zeros(example_xy_shape,
                                 dtype=theano.config.floatX)

        # 1. Construct computation graph (internal to this instance)
        # Note: this part is messy. We should eventually encapsulate
        # scope not via ugly unique names, but by defining within a
        # function
        _feat0 = theano.tensor.tensor4('_feat0')
        _feat0.tag.test_value = example_feat ############
        _id0, _xy0 = self.feat_to_idxy(_feat0)
        _id0.name = '_id0'
        _xy0.name = '_xy0'
        self.fns.feat_to_idxy = theano.function(
            [_feat0], [_id0, _xy0],
            name = 'fns.feat_to_idxy')

        # 2. At this point, if compute_test_value mode is on, all
        # functions should have already been tested (when they were
        # defined). However, we can also test them ourselves (maybe
        # needed for GPU-only code too?)
        run_extra_tests = True
        if run_extra_tests:
            junk = self.fns.feat_to_idxy(example_feat)

    def feat_to_idxy(self, feat):
        '''
        Compute (id,xy) given input features.

        Parameters
        ----------
        feat : tensor_like
            Theano symbolic representing the input
            minibatch(es) to be encoded and reconstructed. Assumed to be
            4-tensors in the space self.input_space

        Returns
        -------
        idxy_tuple : 2-tuple
            Tuple of two Theano symbolics representing the
            corresponding (id,xy) given then input features.
        '''

        idxy_tuple = self.fprop(feat)

        return idxy_tuple
