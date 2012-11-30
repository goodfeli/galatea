from pylearn2.devtools.record import RecordMode
from collections import OrderedDict
from pylearn2.datasets.dataset import Dataset
from pylearn2.devtools import disturb_mem
import numpy as np
from pylearn2.monitor import Monitor
from pylearn2.utils import sharedX
from pylearn2.utils import safe_izip
from pylearn2.models.dbm import DBM
from pylearn2.models.dbm import WeightDoubling
from pylearn2.models.dbm import BinaryVector
from pylearn2.models.dbm import BinaryVectorMaxPool
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.costs.cost import Cost

class A(Cost):
    def __init__(self,
                    noise = False,
                    both_directions = False,
                    l1_act_coeffs = None,
                    l1_act_targets = None,
                    l1_act_eps = None,
                    range_rewards = None,
                    stdev_rewards = None,
                    robustness = None,
                    supervised = False,
                    niter = None,
                    block_grad = None,
                    vis_presynaptic_cost = None,
                    hid_presynaptic_cost = None,
                    reweighted_act_coeffs = None,
                    reweighted_act_targets = None,
                    toronto_act_targets = None,
                    toronto_act_coeffs = None
                    ):
        self.__dict__.update(locals())
        del self.self


    def get_monitoring_channels(self, model, X, Y = None, drop_mask = None, drop_mask_Y = None):

        rval = OrderedDict()

        if drop_mask is not None and drop_mask.ndim < X.ndim:
            if X.ndim != 4:
                raise NotImplementedError()
            drop_mask = drop_mask.dimshuffle(0,1,2,'x')

        H = foo(model, X)

        layers = [ model.visible_layer ] + model.hidden_layers
        states = [ X ] + H

        for layer, state in safe_izip(layers, states):
            d = layer.get_monitoring_channels_from_state(state)
            for key in d:
                mod_key = 'final_inpaint_' + layer.layer_name + '_' + key
                assert mod_key not in rval
                rval[mod_key] = d[key]

        return rval

    def __call__(self, model, X,
            return_locals = False):
        assert not return_locals

        dbm = model

        history = foo(dbm, X)
        final_state = history[-1]


        total_cost = X.sum()

        if return_locals:
            return locals()

        return total_cost

def foo(dbm, V, drop_mask = None):

    H_hat = []
    H_hat.append(dbm.hidden_layers[0].mf_update(
        state_above = None,
        state_below = V,
        iter_name = '0'))

    return H_hat

def prereq(*args):
    disturb_mem.disturb_mem()

class InpaintAlgorithm(object):
    def __init__(self, cost, batch_size=None, batches_per_iter=None,
                 monitoring_batches=None, monitoring_dataset=None,
                 theano_function_mode=None):

        self.__dict__.update(locals())
        if isinstance(monitoring_dataset, Dataset):
            self.monitoring_dataset = { '': monitoring_dataset }

    def setup(self, model, dataset):

        self.monitor = Monitor.get_monitor(model)
        self.monitor.set_theano_function_mode(self.theano_function_mode)

        space = model.get_input_space()
        X = sharedX( space.get_origin_batch(model.batch_size) , 'BGD_X')
        self.space = space
        rng = np.random.RandomState([2012,7,20])
        test_mask = space.get_origin_batch(model.batch_size)
        test_mask = rng.randint(0,2,test_mask.shape)
        drop_mask = sharedX( np.cast[X.dtype] ( test_mask), name = 'drop_mask')
        self.drop_mask = drop_mask
        assert drop_mask.ndim == test_mask.ndim

        Y = None
        drop_mask_Y = None

        obj = self.cost(model,X)

        if self.monitoring_dataset is not None:
            if not any([dataset.has_targets() for dataset in self.monitoring_dataset.values()]):
                Y = None
            assert X.name is not None
            channels = {}
            assert X.name is not None
            cost_channels = self.cost.get_monitoring_channels(model, X = X, Y = Y, drop_mask = drop_mask,
                    drop_mask_Y = drop_mask_Y)
            for key in cost_channels:
                channels[key] = cost_channels[key]

            for dataset_name in self.monitoring_dataset:

                monitoring_dataset = self.monitoring_dataset[dataset_name]
                self.monitor.add_dataset(dataset=monitoring_dataset,
                                    mode="sequential",
                                    batch_size=2,
                                    num_batches=1)
                ipt = X
                if Y is not None:
                    ipt = [X,Y]
                self.monitor.add_channel('objective',ipt=ipt,val=obj, dataset=monitoring_dataset, prereqs =  [ prereq ])

                for name in channels:
                    J = channels[name]
                    if isinstance(J, tuple):
                        assert len(J) == 2
                        J, prereqs = J
                    else:
                        prereqs = []

                    prereqs = list(prereqs)
                    prereqs.append(prereq)

                    if Y is not None:
                        ipt = (X,Y)
                    else:
                        ipt = X

                    self.monitor.add_channel(name=name,
                                             ipt=ipt,
                                             val=J, dataset=monitoring_dataset,
                                             prereqs=prereqs)


        self.inputs = None

        self.X = X

def run(replay):
    X = np.zeros((2,2))
    X[0,0] = 1.
    raw_train = DenseDesignMatrix(X=X)

    train = raw_train

    model = DBM(
            batch_size = 2,
            niter= 2,
            visible_layer= BinaryVector(
                nvis= 2,
                bias_from_marginals = raw_train,
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

    algorithm = InpaintAlgorithm(
        theano_function_mode = RecordMode(
                            file_path= "nondeterminism_4.txt",
                            replay=replay
                   ),
                   monitoring_dataset = OrderedDict([
                            ('train', train)
                            ]
                   ),
                   cost= A(),
            )

    algorithm.setup(model=model, dataset=train)
    model.monitor()

    algorithm.theano_function_mode.record.f.flush()
    algorithm.theano_function_mode.record.f.close()

run(0)
run(1)
