from pylearn2.datasets.binarizer import Binarizer
from pylearn2.datasets.mnist import MNIST
import galatea.dbm.inpaint.super_dbm
import galatea.dbm.inpaint.super_inpaint
import pylearn2.costs.cost
from pylearn2.devtools.record import RecordMode
from collections import OrderedDict
from pylearn2.datasets.dataset import Dataset
import numpy as np
from pylearn2.monitor import Monitor
from pylearn2.utils import sharedX
from pylearn2.utils import function
import warnings
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent
from pylearn2.models.dbm import DBM
from pylearn2.models.dbm import WeightDoubling
import theano


class SuperWeightDoubling(WeightDoubling):
    def do_inpainting(self, V, Y = None, drop_mask = None, drop_mask_Y = None,
            return_history = False, noise = False, niter = None, block_grad = None):
        dbm = self.dbm

        niter = 2

        orig_V = V
        orig_drop_mask = drop_mask

        history = []

        V_hat, V_hat_unmasked = dbm.visible_layer.init_inpainting_state(V,drop_mask,noise, return_unmasked = True)
        assert V_hat_unmasked.ndim > 1

        H_hat = []
        for i in xrange(0,len(dbm.hidden_layers)-1):
            if i == 0:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.visible_layer.upward_state(V_hat),
                    iter_name = '0'))
            else:
                H_hat.append(dbm.hidden_layers[i].mf_update(
                    state_above = None,
                    double_weights = True,
                    state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                    iter_name = '0'))
        if len(dbm.hidden_layers) > 1:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = dbm.hidden_layers[-2].upward_state(H_hat[-1])))
        else:
            H_hat.append(dbm.hidden_layers[-1].mf_update(
                state_above = None,
                state_below = dbm.visible_layer.upward_state(V_hat)))

        if Y is not None:
            Y_hat_unmasked = dbm.hidden_layers[-1].init_inpainting_state(Y, noise)
            Y_hat = drop_mask_Y * Y_hat_unmasked + (1 - drop_mask_Y) * Y
            H_hat[-1] = Y_hat
            if len(dbm.hidden_layers) > 1:
                i = len(dbm.hidden_layers) - 2
                if i == 0:
                    H_hat[i] = dbm.hidden_layers[i].mf_update(
                        state_above = Y_hat,
                        layer_above = dbm.hidden_layers[-1],
                        state_below = dbm.visible_layer.upward_state(V_hat),
                        iter_name = '0')
                else:
                    H_hat[i] = dbm.hidden_layers[i].mf_update(
                        state_above = Y_hat,
                        layer_above = dbm.hidden_layers[-1],
                        state_below = dbm.hidden_layers[i-1].upward_state(H_hat[i-1]),
                        iter_name = '0')


        def update_history():
            assert V_hat_unmasked.ndim > 1
            d =  { 'V_hat' :  V_hat, 'H_hat' : H_hat, 'V_hat_unmasked' : V_hat_unmasked }
            history.append( d )

        update_history()

        for j in xrange(0, len(H_hat), 2):
            if j == 0:
                state_below = dbm.visible_layer.upward_state(V_hat)
            else:
                state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
            if j == len(H_hat) - 1:
                state_above = None
                layer_above = None
            else:
                state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                layer_above = dbm.hidden_layers[j+1]
            H_hat[j] = dbm.hidden_layers[j].mf_update(
                    state_below = state_below,
                    state_above = state_above,
                    layer_above = layer_above)

        V_hat, V_hat_unmasked = dbm.visible_layer.inpaint_update(
                state_above = dbm.hidden_layers[0].downward_state(H_hat[0]),
                layer_above = dbm.hidden_layers[0],
                V = V,
                drop_mask = drop_mask, return_unmasked = True)
        V_hat.name = 'V_hat[%d](V_hat = %s)' % (1, V_hat.name)

        for j in xrange(1,len(H_hat),2):
            state_below = dbm.hidden_layers[j-1].upward_state(H_hat[j-1])
            if j == len(H_hat) - 1:
                state_above = None
                layer_above = None
            else:
                state_above = dbm.hidden_layers[j+1].downward_state(H_hat[j+1])
                layer_above = dbm.hidden_layers[j+1]
            #end if j
            H_hat[j] = dbm.hidden_layers[j].mf_update(
                    state_below = state_below,
                    state_above = state_above,
                    layer_above = layer_above)
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

    def __getstate__(self):
        return {}

class InpaintAlgorithm(object):
    def __init__(self, mask_gen, cost, batch_size=None, batches_per_iter=None,
                 monitoring_batches=None, monitoring_dataset=None,
                 max_iter = 5, suicide = False, init_alpha = None,
                 reset_alpha = True, conjugate = False, reset_conjugate = True,
                 termination_criterion = None, set_batch_size = False,
                 line_search_mode = None, min_init_alpha = 1e-3,
                 duplicate = 1, combine_batches = 1, scale_step = 1.,
                 theano_function_mode=None):
        """
        if batch_size is None, reverts to the force_batch_size field of the
        model
        """

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
        assert not isinstance(X,tuple)
        self.X.set_value(X)
        assert self.cost.supervised == (Y is not None)
        if Y is not None:
            assert Y.ndim == 2
            assert self.Y.ndim == 2
            self.Y.set_value(Y)
        self.update_mask()

    def get_setup_batch_object(self):
        return SetupBatch(self)

    def setup(self, model, dataset):
        if self.set_batch_size:
            model.set_batch_size(self.batch_size)

        if self.batch_size is None:
            self.batch_size = model.force_batch_size

        model.cost = self.cost
        model.mask_gen = self.mask_gen

        self.monitor = Monitor.get_monitor(model)
        self.monitor.set_theano_function_mode(self.theano_function_mode)
        prereq = self.get_setup_batch_object()
        space = model.get_input_space()
        X = sharedX( space.get_origin_batch(model.batch_size) , 'BGD_X')
        self.space = space
        rng = np.random.RandomState([2012,7,20])
        test_mask = space.get_origin_batch(model.batch_size)
        test_mask = rng.randint(0,2,test_mask.shape)
        if hasattr(self.mask_gen,'sync_channels') and self.mask_gen.sync_channels:
            if test_mask.ndim != 4:
                raise NotImplementedError()
            test_mask = test_mask[:,:,:,0]
            assert test_mask.ndim == 3
        drop_mask = sharedX( np.cast[X.dtype] ( test_mask), name = 'drop_mask')
        self.drop_mask = drop_mask
        assert drop_mask.ndim == test_mask.ndim

        Y = None
        drop_mask_Y = None
        updates = OrderedDict([( drop_mask, self.mask_gen(X) )])


        obj = self.cost(model,X, Y, drop_mask = drop_mask, drop_mask_Y = drop_mask_Y)
        gradients, gradient_updates = self.cost.get_gradients(model, X, Y, drop_mask = drop_mask,
                drop_mask_Y = drop_mask_Y)

        self.update_mask = function([], updates = updates)

        if self.monitoring_dataset is not None:
            if not any([dataset.has_targets() for dataset in self.monitoring_dataset.values()]):
                Y = None
            assert X.name is not None
            channels = model.get_monitoring_channels(X,Y)
            if not isinstance(channels, dict):
                raise TypeError("model.get_monitoring_channels must return a "
                                "dictionary, but it returned " + str(channels))
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
                warnings.warn("This is weird-- ipt=(X,Y)=tell the monitor to replace X, Y with the givens dict, "
                        " but you don't actually want them to be replaced.")
                ipt = X
                if Y is not None:
                    ipt = [X,Y]
                self.monitor.add_channel(prefix+'objective',ipt=ipt,val=obj, dataset=monitoring_dataset, prereqs =  [ prereq ])

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

                    self.monitor.add_channel(name=prefix+name,
                                             ipt=ipt,
                                             val=J, dataset=monitoring_dataset,
                                             prereqs=prereqs)


        self.inputs = None

        self.X = X

def run(replay):
    raw_train = MNIST(
        which_set="train",
        shuffle=0,
        one_hot=1,
        start=0,
        stop=2)

    train = raw_train

    model = ADBM(
            batch_size = 2,
            niter= 2,
            visible_layer= galatea.dbm.inpaint.super_dbm.BinaryVisLayer(
                nvis= 784,
                bias_from_marginals = raw_train,
            ),
            hidden_layers= [
                # removing this removes the bug. not sure if I just need to disturb mem though
                galatea.dbm.inpaint.super_dbm.DenseMaxPool(
                    detector_layer_dim= 500,
                            pool_size= 1,
                            sparse_init= 15,
                            layer_name= 'h0',
                            init_bias= 0.
                   )
                  ]
        )

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
