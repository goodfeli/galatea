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

from pylearn2.models.model import Model
from pylearn2.space import VectorSpace

class DummyModel(Model):
    def __init__(self):
        param_spec = {"vishid" : (784, 100), "hidbias" : (100,), "hidpen" : (100, 100), "penhid" : (100, 100), "penbias" : (100,), "softmax_b" : (10,), "softmax_W" : (100, 10)}
        self._params = [sharedX(np.zeros(param_spec[name]), name) for name in sorted(param_spec.keys())]
        self.input_space = VectorSpace(28*28)

model = DummyModel()

from pylearn2.training_algorithms.bgd import BGD
from pylearn2.devtools.record import Record
allocate_random()
from pylearn2.costs.cost import Cost

class DummyCost(Cost):
    supervised = True
    def __call__(self, model, X, Y, **kwargs):
        return sharedX(0.)
        return sum([x.sum() for x in (model.get_params()+[X, Y])])


algorithm =  BGD( **{
               'theano_function_mode': Record(
                        path = 'nondeterminism_2_record.txt',
                        replay = replay
               ),
               'batch_size': 100,
               'cost' : DummyCost()
})

algorithm.setup(model=model, dataset=None)
algorithm.optimizer._cache_values()

