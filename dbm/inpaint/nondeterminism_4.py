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
        """
        Allows the training algorithm to do some preliminary configuration
        *before* we actually start training the model. The dataset is provided
        in case other derived training algorithms need to modify model based on
        the dataset.

        Parameters
        ----------
        model: a Python object representing the model to train loosely
        implementing the interface of models.model.Model.

        dataset: a pylearn2.datasets.dataset.Dataset object used to draw
        training data
        """
        self.model = model

        if self.set_batch_size:
            model.set_batch_size(self.batch_size)

        if self.batch_size is None:
            self.batch_size = model.force_batch_size

        model.cost = self.cost
        model.mask_gen = self.mask_gen

        self.monitor = Monitor.get_monitor(model)
        self.monitor.set_theano_function_mode(self.theano_function_mode)
        prereq = self.get_setup_batch_object()
        #We want to use big batches. We need to make several theano calls on each
        #batch. To avoid paying the GPU latency every time, we use a shared variable
        #but the shared variable needs to stay allocated during the time that the
        #monitor is working, and we don't want the monitor to increase the memory
        #overhead. So we make the monitor work off of the same shared variable
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
        if self.cost.supervised:
            Y = sharedX(model.get_output_space().get_origin_batch(model.batch_size), 'BGD_Y')
            self.Y = Y
            test_mask_Y = rng.randint(0,2,(model.batch_size,))
            drop_mask_Y = sharedX( np.cast[Y.dtype](test_mask_Y), name = 'drop_mask_Y')
            self.drop_mask_Y = drop_mask_Y
            dmx, dmy = self.mask_gen(X, Y)
            updates = OrderedDict([ (drop_mask, dmx),\
                    (drop_mask_Y, dmy)] )
        else:
            updates = OrderedDict([( drop_mask, self.mask_gen(X) )])


        obj = self.cost(model,X, Y, drop_mask = drop_mask, drop_mask_Y = drop_mask_Y)
        gradients, gradient_updates = self.cost.get_gradients(model, X, Y, drop_mask = drop_mask,
                drop_mask_Y = drop_mask_Y)

        if hasattr(model.inference_procedure, 'V_dropout'):
            include_prob = model.inference_procedure.include_prob
            theano_rng = MRG_RandomStreams(2012+11+20)
            for elem in flatten([model.inference_procedure.V_dropout, model.inference_procedure.H_dropout]):
                updates[elem] =  theano_rng.binomial(p=include_prob, size=elem.shape, dtype=elem.dtype, n=1) / include_prob
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
                #we only need to put the prereq in once to make sure it gets run
                #adding it more times shouldn't hurt, but be careful
                #each time you say "self.setup_batch" you get a new object with a
                #different id, and if you install n of those the prereq will run n
                #times. It won't cause any wrong results, just a big slowdown
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


        self.accumulate = self.combine_batches > 1
        if self.accumulate:
            self.inputs = [elem for elem in [X, Y, drop_mask, drop_mask_Y] if elem is not None]
        else:
            self.inputs = None

        self.optimizer = BatchGradientDescent(
                            objective = obj,
                            inputs = self.inputs,
                            verbose = 1,
                            gradients = gradients,
                            gradient_updates = gradient_updates,
                            params = model.get_params(),
                            lr_scalers = model.get_lr_scalers(),
                            param_constrainers = [ model.censor_updates ],
                            max_iter = self.max_iter,
                            tol = 3e-7,
                            init_alpha = self.init_alpha,
                            reset_alpha = self.reset_alpha,
                            conjugate = self.conjugate,
                            reset_conjugate = self.reset_conjugate,
                            min_init_alpha = self.min_init_alpha,
                            line_search_mode = self.line_search_mode,
                            accumulate = self.accumulate,
                            theano_function_mode = self.theano_function_mode)
        self.X = X

        self.monitor.add_channel(name='ave_step_size',
                ipt=ipt, val = self.optimizer.ave_step_size, dataset=self.monitoring_dataset.values()[0])
        self.monitor.add_channel(name='ave_grad_size',
                ipt=ipt, val = self.optimizer.ave_grad_size, dataset=self.monitoring_dataset.values()[0])
        self.monitor.add_channel(name='ave_grad_mult',
                ipt=ipt, val = self.optimizer.ave_grad_mult, dataset=self.monitoring_dataset.values()[0])

        self.first = True
        self.bSetup = True

def run(replay):
    raw_train = MNIST(
        which_set="train",
        shuffle=0,
        one_hot=1,
        start=0,
        stop=2)

    train = raw_train

    model = galatea.dbm.inpaint.super_dbm.SuperDBM(
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
                   cost=\
                                   galatea.dbm.inpaint.super_inpaint.SuperInpaint(
                                            both_directions = 0,
                                            noise =  0,
                                            supervised =  0,
                                   )
                   ,
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
