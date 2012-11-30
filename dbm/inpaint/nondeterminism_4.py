from pylearn2.datasets.binarizer import Binarizer
import galatea.dbm.inpaint.super_dbm
import galatea.dbm.inpaint.super_inpaint
import pylearn2.costs.cost
from pylearn2.devtools.record import RecordMode
from collections import OrderedDict
from pylearn2.datasets.dataset import Dataset
from pylearn2.devtools import disturb_mem
import numpy as np
from pylearn2.monitor import Monitor
from pylearn2.utils import sharedX
from pylearn2.utils import function
import warnings
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent
from pylearn2.models.dbm import DBM
from pylearn2.models.dbm import WeightDoubling
from pylearn2.models.dbm import BinaryVector
import theano
import theano.tensor as T
from pylearn2.utils import block_gradient
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class BinaryVisLayer(BinaryVector):
    def recons_cost(self, V, V_hat_unmasked, drop_mask = None):
        return V_hat_unmasked.sum()

class SuperWeightDoubling(WeightDoubling):
    def do_inpainting(self, V, Y = None, drop_mask = None, drop_mask_Y = None,
            return_history = False, noise = False, niter = None, block_grad = None):
        dbm = self.dbm

        niter = 2

        history = []

        V_hat = V
        V_hat_unmasked = V

        H_hat = []
        H_hat.append(dbm.hidden_layers[0].mf_update(
            state_above = None,
            state_below = V_hat,
            iter_name = '0'))

        def update_history():
            assert V_hat_unmasked.ndim > 1
            d =  { 'V_hat' :  V_hat, 'H_hat' : H_hat, 'V_hat_unmasked' : V_hat_unmasked }
            history.append( d )

        update_history()

        V_hat_unmasked = dbm.hidden_layers[0].downward_message(H_hat[0][0])
        V_hat = V_hat_unmasked
        V_hat.name = 'V_hat[%d](V_hat = %s)' % (1, V_hat.name)

        update_history()

        if return_history:
            return history
        else:
            return V_hat

class ADBM(DBM):
    def setup_inference_procedure(self):
        if not hasattr(self, 'inference_procedure') or \
                self.inference_procedure is None:
            self.inference_procedure = SuperWeightDoubling()
            self.inference_procedure.set_dbm(self)

    def do_inpainting(self, *args, **kwargs):
        self.setup_inference_procedure()
        return self.inference_procedure.do_inpainting(*args, **kwargs)

    def mf(self, *args, **kwargs):
        self.setup_inference_procedure()
        return self.inference_procedure.mf(*args, **kwargs)

class SetupBatch:
    def __init__(self,alg):
        self.alg = alg

    def __call__(self, * args):
        if len(args) > 1:
            X, Y = args
            self.alg.setup_batch(X, Y)
        else:
            X = args
            self.alg.setup_batch(X)

class InpaintAlgorithm(object):
    def __init__(self, mask_gen, cost, batch_size=None, batches_per_iter=None,
                 monitoring_batches=None, monitoring_dataset=None,
                 max_iter = 5, suicide = False, init_alpha = None,
                 reset_alpha = True, conjugate = False, reset_conjugate = True,
                 termination_criterion = None, set_batch_size = False,
                 line_search_mode = None, min_init_alpha = 1e-3,
                 duplicate = 1, combine_batches = 1, scale_step = 1.,
                 theano_function_mode=None):

        if line_search_mode is None and init_alpha is None:
            init_alpha = ( .001, .005, .01, .05, .1 )

        self.__dict__.update(locals())
        del self.self
        if monitoring_dataset is None:
            assert monitoring_batches == None
        if isinstance(monitoring_dataset, Dataset):
            self.monitoring_dataset = { '': monitoring_dataset }
        self.bSetup = False
        self.rng = np.random.RandomState([2012,10,17])

    def setup_batch(self, X, Y = None):
        self.X.set_value(X)

    def get_setup_batch_object(self):
        return SetupBatch(self)

    def setup(self, model, dataset):
        if self.set_batch_size:
            model.set_batch_size(self.batch_size)

        if self.batch_size is None:
            self.batch_size = model.force_batch_size

        #model.cost = self.cost
        #model.mask_gen = self.mask_gen

        self.monitor = Monitor.get_monitor(model)
        self.monitor.set_theano_function_mode(self.theano_function_mode)
        prereq = self.get_setup_batch_object()
        space = model.get_input_space()
        X = sharedX( space.get_origin_batch(model.batch_size) , 'BGD_X')
        self.space = space
        rng = np.random.RandomState([2012,7,20])
        test_mask = space.get_origin_batch(model.batch_size)
        test_mask = rng.randint(0,2,test_mask.shape)
        #if hasattr(self.mask_gen,'sync_channels') and self.mask_gen.sync_channels:
        #    if test_mask.ndim != 4:
        #        raise NotImplementedError()
        #    test_mask = test_mask[:,:,:,0]
        #    assert test_mask.ndim == 3
        drop_mask = sharedX( np.cast[X.dtype] ( test_mask), name = 'drop_mask')
        self.drop_mask = drop_mask
        assert drop_mask.ndim == test_mask.ndim

        Y = None
        drop_mask_Y = None
        updates = OrderedDict([( drop_mask, self.mask_gen(X) )])

        obj = self.cost(model,X, Y, drop_mask = drop_mask, drop_mask_Y = drop_mask_Y)


        if self.monitoring_dataset is not None:
            if not any([dataset.has_targets() for dataset in self.monitoring_dataset.values()]):
                Y = None
            assert X.name is not None
            channels = model.get_monitoring_channels(X,Y)
            assert X.name is not None
            wtf = self.cost.get_monitoring_channels(model, X = X, Y = Y, drop_mask = drop_mask,
                    drop_mask_Y = drop_mask_Y)
            for key in wtf:
                channels[key] = wtf[key]

            for dataset_name in self.monitoring_dataset:

                if dataset_name == '':
                    prefix = ''
                else:
                    prefix = dataset_name + '_'

                monitoring_dataset = self.monitoring_dataset[dataset_name]
                self.monitor.add_dataset(dataset=monitoring_dataset,
                                    mode="sequential",
                                    batch_size=self.batch_size,
                                    num_batches=self.monitoring_batches)
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

    model = ADBM(
            batch_size = 2,
            niter= 2,
            visible_layer= BinaryVisLayer(
                nvis= 2,
                bias_from_marginals = raw_train,
            ),
            hidden_layers= [
                # removing this removes the bug. not sure if I just need to disturb mem though
                galatea.dbm.inpaint.super_dbm.DenseMaxPool(
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
                   line_search_mode= 'exhaustive',
                   init_alpha= [0.0256, .128, .256, 1.28, 2.56],
                   reset_alpha= 0,
                   conjugate= 1,
                   reset_conjugate= 0,
                   max_iter= 5,
                   cost= galatea.dbm.inpaint.super_inpaint.SuperInpaint(
                                            both_directions = 0,
                                            noise =  0,
                                            supervised =  0,
                                   ),
                   mask_gen = galatea.dbm.inpaint.super_inpaint.MaskGen (
                            drop_prob= 0.1,
                            balance= 0,
                            sync_channels= 0
                   )
            )

    algorithm.setup(model=model, dataset=train)
    model.monitor()

    algorithm.theano_function_mode.record.f.flush()
    algorithm.theano_function_mode.record.f.close()

run(0)
run(1)
