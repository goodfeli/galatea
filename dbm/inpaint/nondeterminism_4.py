from pylearn2.devtools.record import RecordMode
from collections import OrderedDict
from pylearn2.devtools import disturb_mem
import numpy as np
from pylearn2.monitor import Monitor
import theano
from pylearn2.models.dbm import DBM
from pylearn2.models.dbm import BinaryVector
from pylearn2.models.dbm import BinaryVectorMaxPool
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
import theano.tensor as T

def prereq(*args):
    disturb_mem.disturb_mem()

class DummyModel(Model):
    def __init__(self):
        self.param = sharedX(np.zeros((2,)))
        self._params = [self.param]
        self.input_space = VectorSpace(2)

def run(replay):
    X = np.zeros((2,2))
    X[0,0] = 1.
    train = DenseDesignMatrix(X=X)

    model = DBM(
            batch_size = 2,
            niter= 2,
            visible_layer= BinaryVector(
                nvis= 2,
                bias_from_marginals = train,
            ),
            hidden_layers= [
                # removing this removes the bug. not sure if I just need to disturb mem though
                BinaryVectorMaxPool(
                    detector_layer_dim= 2,
                            pool_size= 1,
                            sparse_init= 1,
                            layer_name= 'h0',
                            init_bias= 0.
                   )
                  ]
        )
    model = DummyModel()
    disturb_mem.disturb_mem()

    theano_function_mode = RecordMode(
                            file_path= "nondeterminism_4.txt",
                            replay=replay
                   )

    monitor = Monitor.get_monitor(model)
    monitor.set_theano_function_mode(theano_function_mode)


    b = model.param #hidden_layers[0].b
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


    monitor.add_dataset(dataset=train, mode="sequential",
                                    batch_size=2,
                                    num_batches=1)
    ipt = theano.tensor.matrix()
    prereqs = [prereq]

    for name in channels:
        J = channels[name]
        monitor.add_channel(name=name,
                                 ipt=ipt,
                                 val=J, dataset=train,
                                 prereqs=prereqs)
    monitor()

    theano_function_mode.record.f.flush()
    theano_function_mode.record.f.close()

run(0)
run(1)
