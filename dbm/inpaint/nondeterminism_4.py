from pylearn2.devtools.record import RecordMode
from collections import OrderedDict
from pylearn2.devtools import disturb_mem
import numpy as np
from pylearn2.monitor import Monitor
import theano
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX

class DummyModel(Model):
    def __init__(self):
        self.param = sharedX(np.zeros((2,)))
        self._params = [self.param]
        self.input_space = VectorSpace(2)

def run(replay):
    model = DummyModel()
    disturb_mem.disturb_mem()

    mode = RecordMode(file_path= "nondeterminism_4.txt",
                      replay=replay)

    monitor = Monitor.get_monitor(model)
    monitor.set_theano_function_mode(mode)

    b = model.param
    channels = OrderedDict()

    disturb_mem.disturb_mem()

    v_max = b.max(axis=0)
    v_min = b.min(axis=0)
    v_mean = b.mean(axis=0)
    v_range = v_max - v_min

    for key, val in [
            ('max_x.max_u', v_max.max()),
            ('max_x.min_u', v_max.min()),
            ('min_x.max_u', v_min.max()),
            ('range_x.max_u', v_range.max()),
            ('mean_x.max_u', v_mean.max()),
            ]:
        disturb_mem.disturb_mem()
        channels[key] = val

    updates = []
    for key in channels:
        s = sharedX(0.)
        updates.append((s, channels[key]))
    X = theano.tensor.matrix()
    f = theano.function([X], mode=mode, updates=updates, on_unused_input='ignore', name='f')
    disturb_mem.disturb_mem()
    f(np.zeros((2,2)).astype(X.dtype))

    mode.record.f.flush()
    mode.record.f.close()

run(0)
run(1)
