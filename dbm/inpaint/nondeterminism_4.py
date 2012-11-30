from pylearn2.devtools.record import RecordMode
from collections import OrderedDict
from pylearn2.datasets.dataset import Dataset
from pylearn2.devtools import disturb_mem
import numpy as np
from pylearn2.monitor import Monitor
import theano
from pylearn2.models.dbm import DBM
from pylearn2.models.dbm import BinaryVector
from pylearn2.models.dbm import BinaryVectorMaxPool
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def get_monitoring_channels(model, X):

    rval = OrderedDict()

    layer = model.hidden_layers[0]
    state = layer.mf_update(
        state_above = None,
        state_below = X,
        iter_name = '0')

    d = layer.get_monitoring_channels_from_state(state)
    for key in d:
        mod_key = '_' + key
        assert mod_key not in rval
        rval[mod_key] = d[key]

    return rval

def prereq(*args):
    disturb_mem.disturb_mem()


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
    disturb_mem.disturb_mem()

    theano_function_mode = RecordMode(
                            file_path= "nondeterminism_4.txt",
                            replay=replay
                   )

    monitor = Monitor.get_monitor(model)
    monitor.set_theano_function_mode(theano_function_mode)

    X = theano.tensor.matrix()

    channels = get_monitoring_channels(model, X = X)

    monitor.add_dataset(dataset=train, mode="sequential",
                                    batch_size=2,
                                    num_batches=1)
    ipt = X

    for name in channels:
        J = channels[name]
        if isinstance(J, tuple):
            assert False
            assert len(J) == 2
            J, prereqs = J
        else:
            prereqs = []

        prereqs = list(prereqs)
        prereqs.append(prereq)

        monitor.add_channel(name=name,
                                 ipt=ipt,
                                 val=J, dataset=train,
                                 prereqs=prereqs)
    monitor()

    theano_function_mode.record.f.flush()
    theano_function_mode.record.f.close()

run(0)
run(1)
