'''
Convnet PredNets, denoising autoencoders, and stacked DAEs.
'''


# Standard library imports
#import functools
#from itertools import izip
#from collections import OrderedDict
#import operator
import ipdb as pdb

# Third-party imports
#import numpy
#import theano
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



class FishMLP(MLP):
    '''
    A slight extension of MLP.
    '''

    def __init__(self, *args, **kwargs):
        super(FishMLP, self).__init__(*args, **kwargs)

    def get_target_source(self):
        '''Override default Model.get_target_source'''
        return ('targets1', 'targets2')

