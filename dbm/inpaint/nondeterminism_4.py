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
import theano.tensor as T

def get_monitoring_channels(model, X):

    rval = OrderedDict()

    layer = model.hidden_layers[0]
    b  = layer.b
    H = b
    state = (H, H)

    def get_monitoring_channels_from_state():
        self = layer

        rval = OrderedDict()

        vars_and_prefixes = [ (H,'') ]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            for key, val in [
                    ('max_x.max_u', v_max.max()),
                    ('max_x.mean_u', v_max.mean()),
                    ('max_x.min_u', v_max.min()),
                    ('min_x.max_u', v_min.max()),
                    ('range_x.max_u', v_range.max()),
                    ('mean_x.max_u', v_mean.max()),
                    ]:
                rval[prefix+key] = val

        return rval

    d = get_monitoring_channels_from_state()
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
