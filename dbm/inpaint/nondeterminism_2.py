from pylearn2.config import yaml_parse
import sys
import numpy as np
from pylearn2.utils import sharedX

_, replay = sys.argv
replay = int(replay)

def allocate_random():
    # Allocate a time-dependent amount of objects to increase
    # chances of all subsequent objects' ids changing from run
    # to run
    global l
    from datetime import datetime
    now = datetime.now()
    ms = now.microsecond
    ms = int(ms)
    n = ms % 1000
    m = ms / 1000
    l = [[0]*m for i in xrange(n)]
allocate_random()

yaml_src = """
    !obj:galatea.dbm.inpaint.super_dbm.MLP_Wrapper {
                        decapitate: 0,
                        super_dbm: !obj:galatea.dbm.inpaint.super_dbm.SuperDBM {
                              batch_size : 100,
                              niter: 6, #note: since we have to backprop through the whole thing, this does
                                         #increase the memory usage
                              visible_layer: !obj:galatea.dbm.inpaint.super_dbm.BinaryVisLayer {
                                nvis: 784,
                                #bias_from_marginals: *train,
                              },
              hidden_layers: [
                !obj:galatea.dbm.inpaint.super_dbm.DenseMaxPool {
                        detector_layer_dim: 100,
                        pool_size: 1,
                        sparse_init: 15,
                        layer_name: 'h0',
                        init_bias: 0.
               },
                !obj:galatea.dbm.inpaint.super_dbm.DenseMaxPool {
                        detector_layer_dim: 100,
                        pool_size: 1,
                        sparse_init: 15,
                        layer_name: 'h1',
                        init_bias: 0.
               },
               !obj:galatea.dbm.inpaint.super_dbm.Softmax {
                        sparse_init: 15,
                        layer_name: 'c',
                        n_classes: 10
               }
              ]
    },
    }"""
model =  yaml_parse.load(yaml_src)

from pylearn2.models.model import Model
from pylearn2.space import VectorSpace

class DummyModel(Model):
    def __init__(self, model):
        param_spec = {"vishid" : (784, 100), "hidbias" : (100,), "hidpen" : (100, 100), "penhid" : (100, 100), "penbias" : (100,), "softmax_b" : (10,), "softmax_W" : (100, 10)}
        self._params = [sharedX(np.zeros(param_spec[name]), name) for name in sorted(param_spec.keys())]
        """
        self._params = model.get_params()
        param_rep = []
        for param in self._params:
            param_rep.append('"'+param.name+'" : '+str(param.get_value().shape))
        param_rep = '{' + ', '.join(param_rep) + '}'
        print param_rep
        quit(-1)
        """
        self.input_space = VectorSpace(28*28)

#DummyModel(model)
model = DummyModel(model)

from pylearn2.training_algorithms.bgd import BGD
from pylearn2.devtools.record import Record
allocate_random()
from pylearn2.costs.cost import Cost

class DummyCost(Cost):
    supervised = True
    def __call__(self, model, X, Y, **kwargs):
        return sum([x.sum() for x in (model.get_params()+[X, Y])])


algorithm =  BGD( **{
               'theano_function_mode': Record(
                        path = 'nondeterminism_2_record.txt',
                        replay = replay
               ),
               'line_search_mode': 'exhaustive',
               'batch_size': 100,
               'set_batch_size': 1,
               'updates_per_batch': 1,
               'reset_alpha': 0,
               'conjugate': 1,
               'reset_conjugate': 0,
               'cost' : DummyCost()
})

algorithm.setup(model=model, dataset=None)
algorithm.optimizer._cache_values()

